"""자유 텍스트(ex_want / ex_have)를 8개 기준으로 추출하고, 추출 결과(ex_trait)를 읽는다.

- 추출은 사람(registration)당 side(want/have)당 한 번만 한다. 쌍마다 하지 않는다.
- 기준마다 LLM이 두 가지를 돌려준다.
    statement  원문을 정리해 다시 쓴 한 구절. 채점(비교)에는 이것만 쓴다. ex_trait의 기준 컬럼에 저장한다.
    evidence   statement의 근거가 된 원문 구절(원문 그대로 복사). 사람이 확인하는 용도라 raw_response에만 둔다.
- evidence가 원문의 부분 문자열인지는 코드가 검증한다. 틀리면 이유를 알려 주고 다시 시도한다.
  (statement는 재작성이라 글자 검증은 못 하지만, 길이와 거절 표현이 남았는지는 검증한다.)
- "~한 사람은 싫어요" 같은 거절은 "~하지 않는 사람"으로, "너무 ~한 사람은 말고" 같은 한도는
  "적당히 ~한"처럼 정도를 나타내는 말로 바꿔 쓴다. 확신 없는 말("~맞죠?", "~잘 모르겠어요", "~일까요?")은 넣지 않는다. "~같아요."와 같은 표현은 허용한다.
- 같은 기준이 원문 여러 곳에 흩어져 있어도 한 기준에 한 구절로 합친다. (행을 나누지 않는다.)
"""

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

from psycopg import errors as pg_errors
from pydantic import BaseModel, Field

from score_functions.llm import parse_chat

CRITERIA = (
    "impression",
    "appearance",
    "personality",
    "vibe_style",
    "interests",
    "relationship_values",
    "lifestyle",
    "background",
)
CRITERION_KO = {
    "impression": "인상",
    "appearance": "외모",
    "personality": "성격/가치관",
    "vibe_style": "분위기/스타일",
    "interests": "취미/관심사",
    "relationship_values": "연애관",
    "lifestyle": "생활습관",
    "background": "배경",
}
CRITERION_HINT = {
    "impression": "얼굴상·첫인상·이미지. 예) 강아지상, 순한 인상",
    "appearance": "키·체형·이목구비 등 눈에 보이는 외형. 예) 잘생긴, 키 큰, 이목구비 뚜렷한",
    "personality": (
        "성격·성향·가치관(인생관·가족관 등)·상대를 대하는 태도. "
        "예) 배려심 많은, 내성적인, 활발한, 잘 맞춰주는, 잘 챙겨주는"
    ),
    "vibe_style": "전체적인 분위기·옷차림·꾸밈 스타일. 예) 깔끔한 스타일, 단정한 옷차림",
    "interests": "취미·관심사·좋아하는 활동. 예) 영화 감상, 전시회, 카페 투어",
    "relationship_values": "연애관·연애 방식·애정표현·스킨십 같은 연애에 관한 것만. 예) 연락을 자주 하는 편, 잘 안아주는, 애정표현이 많은",
    "lifestyle": "흡연·음주·운동·기상 시간·식습관 같은 생활 습관. 예) 담배 안 피는, 아침형, 운동하는",
    "background": "군필 여부·직업·학력·소속·거주지 같은 사실 정보. 예) 군필, 대학원생, 개발자, 서울 거주",
}
SIDES = ("want", "have")
MAX_RETRIES = 2  # 첫 시도 뒤 최대 2번 더 → 최대 3회
MAX_STATEMENT_CHARS = 100  # 한 기준의 재작성 구절 최대 길이. 길면 요약이 아니라 원문 복사에 가깝다.
# 재작성 구절에 거절 표현이 그대로 남아 있으면 다시 쓰게 한다. ("~하지 않는", "적당히 ~한"으로 바꿔야 한다.)
REJECTION_MARKERS = ("싫", "별로", "부담스", "말고", "꺼려", "상대하기 어려")


class TraitValue(BaseModel):
    # evidence를 먼저 쓰게 해서, 재작성이 원문 근거에서 벗어나지 않게 한다.
    evidence: list[str] = Field(
        description="statement의 근거가 되는 원문 구절. 원문에 적힌 글자를 그대로 복사한다. 떨어져 있으면 여러 개로 나눈다."
    )
    statement: str = Field(
        description=(
            "evidence를 정리해 다시 쓴 한 구절. 의미를 바꾸거나 덧붙이지 않는다. "
            "거절은 '~하지 않는', 한도는 '적당히 ~한'처럼 정도를 나타내는 말로 쓴다. "
            "'~맞죠?', '~잘 모르겠어요', '~일까요?' 같은 확신 없는 말은 쓰지 않는다. '~같아요'는 허용한다."
        )
    )


class TraitExtraction(BaseModel):
    impression: Optional[TraitValue] = Field(description=CRITERION_HINT["impression"] + ". 없으면 null.")
    appearance: Optional[TraitValue] = Field(description=CRITERION_HINT["appearance"] + ". 없으면 null.")
    personality: Optional[TraitValue] = Field(description=CRITERION_HINT["personality"] + ". 없으면 null.")
    vibe_style: Optional[TraitValue] = Field(description=CRITERION_HINT["vibe_style"] + ". 없으면 null.")
    interests: Optional[TraitValue] = Field(description=CRITERION_HINT["interests"] + ". 없으면 null.")
    relationship_values: Optional[TraitValue] = Field(
        description=CRITERION_HINT["relationship_values"] + ". 없으면 null."
    )
    lifestyle: Optional[TraitValue] = Field(description=CRITERION_HINT["lifestyle"] + ". 없으면 null.")
    background: Optional[TraitValue] = Field(description=CRITERION_HINT["background"] + ". 없으면 null.")


_CRITERIA_LINES = "\n".join(f"- {key} ({CRITERION_KO[key]}): {CRITERION_HINT[key]}" for key in CRITERIA)

# 프롬프트에 싣는 예시. 테스트가 같은 데이터를 검증한다.
# (방향, 제목, 글, {기준: (statement, [evidence, ...])}, 덧붙이는 설명)
EXAMPLES: list[tuple[str, str, str, dict[str, tuple[str, list[str]]], str]] = [
    (
        "want",
        "거절을 '~하지 않는'으로",
        "잘 웃고 배려심 있는 사람이면 좋겠어요. 담배 안 피는 사람이 좋고 영화 같이 보러 다니고 싶어요. 술 많이 마시는 사람은 싫어요.",
        {
            "personality": ("잘 웃고 배려심 있는 사람", ["잘 웃고 배려심 있는 사람이면 좋겠어요"]),
            "lifestyle": (
                "담배를 피우지 않고 술을 많이 마시지 않는 사람",
                ["담배 안 피는 사람이 좋고", "술 많이 마시는 사람은 싫어요"],
            ),
            "interests": ("영화를 같이 보러 다니는 것", ["영화 같이 보러 다니고 싶어요"]),
        },
        "같은 기준(lifestyle)의 내용은 떨어져 있어도 한 구절로 합친다.",
    ),
    (
        "want",
        "한도를 정도 표현으로, 사실 정보",
        "다정한 사람 좋아하고 그렇다고 너무 느끼한 사람은 말고요. 외모는 아이유 느낌이 좋은데 너무 압도적으로 예쁜 사람은 부담스러워요. "
        "군필이면 좋겠고 잘 안아주는 사람이었으면 해요. 책 읽는 건 별로예요.",
        {
            "personality": (
                "과하게 느끼하지 않은 다정한 사람",
                ["다정한 사람 좋아하고 그렇다고 너무 느끼한 사람은 말고요"],
            ),
            "appearance": (
                "아이유 느낌이되 과하게 예쁘지 않은 외모",
                ["외모는 아이유 느낌이 좋은데 너무 압도적으로 예쁜 사람은 부담스러워요"],
            ),
            "background": ("군필", ["군필이면 좋겠고"]),
            "relationship_values": ("잘 안아주는 사람", ["잘 안아주는 사람이었으면 해요"]),
        },
        '"책 읽는 건 별로예요"는 상대에 대한 요구인지 본인 취향인지 알 수 없어 넣지 않는다.',
    ),
    (
        "want",
        "비표준 강조도 정도 표현으로 옮긴다",
        "말 잘 통하고 성격이 진짜x9999 다정한 사람!! 감성적인데 개웃기면 좋겠구 낭만을 아는 사람ㅎㅎ 키는 컸으면 좋겠어요.",
        {
            "personality": (
                "말이 잘 통하고 무엇보다 다정하며, 감성적이고 매우 웃기고 낭만을 아는 사람",
                ["말 잘 통하고 성격이 진짜x9999 다정한 사람", "감성적인데 개웃기면 좋겠구 낭만을 아는 사람"],
            ),
            "appearance": ("키 큰 사람", ["키는 컸으면 좋겠어요"]),
        },
        '"진짜x9999"는 가장 세게 강조한 것이라 "무엇보다", "개웃기면"은 "매우"로 옮겼다. 같은 기준의 특성(감성적·웃긴·낭만)도 빠뜨리지 않는다.',
    ),
    (
        "want",
        "글쓴이 본인의 태도는 상대의 조건으로 바꾸지 않는다",
        "키 큰 사람이 좋아요. 취미는 상대가 좋아하면 뭐든 같이 해요. 요리는 빼고요. 저는 낯을 좀 가려요.",
        {
            "appearance": ("키 큰 사람", ["키 큰 사람이 좋아요"]),
            "interests": (
                "상대의 취미는 가리지 않음(요리 제외)",
                ["취미는 상대가 좋아하면 뭐든 같이 해요", "요리는 빼고요"],
            ),
        },
        '"뭐든 같이 해요"는 글쓴이 본인의 태도라서 "뭐든 즐기는 사람"처럼 상대의 조건으로 바꾸지 않는다. '
        '"저는 낯을 좀 가려요"는 본인을 소개하는 말이라 want에는 넣지 않는다.',
    ),
    (
        "have",
        "",
        "강아지상이라는 말 자주 들어요. 운동 좋아하고 아침형 인간이에요.",
        {
            "impression": ("강아지상", ["강아지상이라는 말 자주 들어요"]),
            "lifestyle": ("운동을 좋아하는 아침형 인간", ["운동 좋아하고 아침형 인간이에요"]),
        },
        "",
    ),
    (
        "have",
        "사실 정보, 상대를 대하는 태도",
        "군필이고 IT 회사 다녀요. 상대에게 잘 맞춰주는 편이에요.",
        {
            "background": ("군필, IT 회사 재직", ["군필이고 IT 회사 다녀요"]),
            "personality": ("상대에게 잘 맞춰주는 편", ["상대에게 잘 맞춰주는 편이에요"]),
        },
        "",
    ),
    (
        "have",
        "확신 없는 말",
        "제 매력을 모르겠어요. 굳이 적자면 직진남이에요. 쉬는 날엔 혼자 산책하면서 감성을 느껴요. 근데 키가 작은 것도 매력 맞죠..?",
        {
            "personality": ("직진남", ["굳이 적자면 직진남이에요"]),
            "interests": ("혼자 산책하며 감성을 느끼는 것", ["쉬는 날엔 혼자 산책하면서 감성을 느껴요"]),
        },
        '"키가 작은 것도 매력 맞죠..?"는 확신 없는 질문이라 넣지 않는다.',
    ),
    (
        "have",
        "추측형 어미는 허용, 잘 모르겠다는 말은 제외",
        "저는 좀 내향적인 것 같아요. 주말엔 집에서 쉬는 편인 것 같아요. 제 장점이 뭔지는 잘 모르겠어요.",
        {
            "personality": ("내향적인 편", ["저는 좀 내향적인 것 같아요"]),
            "lifestyle": ("주말엔 집에서 쉬는 편", ["주말엔 집에서 쉬는 편인 것 같아요"]),
        },
        '"~같아요"는 추측형 어미일 뿐 내용을 말하는 구절이라 넣고, "장점이 뭔지는 잘 모르겠어요"는 확신 없는 말이라 넣지 않는다.',
    ),
]


def _render_examples() -> str:
    blocks = []
    for side, title, text, items, note in EXAMPLES:
        header = f"## 예시 ({side}{', ' + title if title else ''})"
        lines = [header, f'글: "{text}"']
        for key in CRITERIA:
            if key in items:
                statement, evidence = items[key]
                shown = ", ".join(f'"{piece}"' for piece in evidence)
                lines.append(f'- {key}: statement "{statement}" / evidence [{shown}]')
        lines.append("나머지 기준은 null." + (f" ({note})" if note else ""))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


EXTRACT_SYSTEM = f"""
당신은 소개팅 매칭을 위한 텍스트 구조화 도구다. 사용자가 자유롭게 쓴 글을 읽고, 8개 기준마다 해당하는 내용을 짧은 한 구절로 정리한다.
기준마다 evidence(원문에서 그대로 복사한 근거)와 statement(정리해 다시 쓴 구절)를 함께 돌려준다. 채점에는 statement만 쓰고, evidence는 사람이 원문과 대조해 확인하는 데 쓴다.

## 규칙
1. evidence는 원문에 적힌 글자를 그대로 복사한다. 요약, 단어 바꾸기, 문법 고치기를 하지 않는다.
   - statement의 근거가 되는 부분만 필요한 만큼 짧게 복사한다. 한 문장 전체일 필요는 없다.
   - 같은 기준의 내용이 떨어져 있으면 evidence를 여러 개로 나눠 담는다.
   - 거절·한도 표현("싫어요", "말고요")이 statement의 근거라면 그것도 evidence에 원문 그대로 복사한다.
2. statement는 evidence를 정리해 다시 쓴 짧은 한 구절(100자 이내)이다.
   - 원문에 없는 사실이나 성향을 덧붙이거나 의미를 바꾸지 않는다.
   - 같은 기준의 내용이 여러 곳에 있으면 한 구절로 합친다.
   - 강조는 지우지 않고 정도를 나타내는 말로 옮긴다. "진짜", "완전", "개", "ㄹㅇ", "겁나", "엄청", "짱"은 물론이고,
     "x100000", "진짜진짜", "다정다정", "다정!!!"처럼 글자나 숫자를 붙이거나 늘이거나 반복해서 표준적이지 않게 강조한 표현도 뜻을 읽어서 반영한다.
     예) "성격이 진짜x100000 다정한 사람" → "무엇보다 다정한 사람" (× "다정한 사람" — 강조가 사라진 오류다.)
     - 같은 기준 안에서 가장 세게 강조한 특성은 "무엇보다", "특히"로, 보통 강조("진짜", "완전")는 "매우", "아주"로 쓴다. 강조하지 않은 특성은 정도 표현 없이 쓴다.
   - 같은 기준에 속한 특성은 빠뜨리지 않고 모두 담는다. 길어지면 표현을 줄이되 강조와 정도는 남긴다.
   - "ㅎ", "ㅠ", 줄임말, 말버릇은 정리하고, 주어 없이 특성 중심으로 쓴다. "나도", "저는" 같은 1인칭은 쓰지 않는다. want이면 상대에게 원하는 모습, have이면 본인이 가진 모습이 드러나게 쓴다.
3. 거절과 한도는 긍정형으로 바꿔 쓴다. statement에는 싫다, 별로다, 부담스럽다, 말고요 같은 거절 표현을 남기지 않는다.
   - 완전한 거절: "OOO한 사람은 싫어요/별로예요" → "OOO하지 않는 사람". 예) "담배 피는 사람 싫어요" → "담배를 피우지 않는 사람"
   - 취향의 한도: "너무 ~한 사람은 말고요", "~하면 부담스러워요" → "적당히 ~한"처럼 정도를 나타내는 말로 바꿔 쓴다. 예) "진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고" → "적당히 진지한 사람"
   - 범위에서 빼는 말: "독서는 빼고요" → 앞 구절의 의미 뒤에 "(독서 제외)"를 붙여 쓴다. 예) 규칙 7의 "상대의 취미는 가리지 않음(독서 제외)"
   - "안", "못"이 들어 있어도 원하는 모습·가진 모습을 말하는 구절은 그대로 쓴다. 예) "담배 안 피는 사람이 좋아요" → "담배를 피우지 않는 사람"
   - 거절하는 대상이 상대에 대한 요구인지 본인 취향인지 알 수 없으면 넣지 않는다.
4. 확신 없는 말은 넣지 않는다. "~맞죠?", "~잘 모르겠어요", "~일까요?"처럼 확인을 구하거나 스스로도 확신하지 못하는 구절은 evidence에도 statement에도 쓰지 않는다.
   - "~같아요", "~인 것 같아요"는 허용한다. 추측형 어미로 말해도 자기 모습이나 바라는 모습을 말하는 구절이면 넣는다. 예) "내향적인 것 같아요" → "내향적인 편"
   - "굳이 적자면 직진남이에요"처럼 망설이는 말이 앞에 붙어도 내용을 단정하는 구절이면 넣는다.
5. 글에 언급이 없는 기준은 null이다. 없는 내용을 추측해서 채우지 않는다.
6. 한 구절은 가장 알맞은 기준 하나에만 넣는다.
   - 딱 맞는 기준이 없어도 의미가 가장 가까운 기준에 넣는다. 예) "잘 맞춰줘요"(상대를 대하는 태도) → personality, "잘 안아줄 수 있는 사람"(애정표현·스킨십) → relationship_values.
   - 가치관(인생관·가족관 등)은 personality에 넣는다. relationship_values에는 연락 빈도·애정표현·스킨십 같은 연애관만 넣는다.
     예) "가족이 가장 중요해요" → personality, "연락은 자주 하는 편이에요", "스킨십이 많아요" → relationship_values
   - 결혼관(결혼에 대한 생각·결혼 계획)은 어느 기준에도 넣지 않는다.
   - 군필 여부·직업·학력·소속·거주지 같은 사실 정보는 background에 넣는다. 예) "군필", "대학원생", "개발자".
   - 어느 기준에도 가깝지 않은 말은 넣지 않는다.
7. 구절의 주체를 바꾸지 않는다. 재작성하기 전에 그 구절이 "상대의 모습"을 말하는지 "글쓴이 본인의 행동·태도"를 말하는지 먼저 가른다.
   - want 글에서 상대의 모습을 말하는 구절은 상대에게 원하는 조건으로 쓴다. ("저는 ~한 사람이 좋아요", "저는 ~한 사람은 어려워요"도 상대의 모습을 말하므로 조건이다.)
   - want 글에서 글쓴이 본인의 너그러움·허용("상대가 좋다면 어떤 취미라도 즐겨요", "뭐든 같이 해요")은 상대가 어떤 사람이길 바라는 말이 아니다.
     "~한 사람"처럼 상대의 조건으로 바꾸지 않고, 그 항목에 조건이 없다는 뜻으로 "상대의 ~는 가리지 않음"이라고 쓴다.
     예) "상대가 좋다면 어떤 취미라도 즐깁니다. 독서빼고요." → interests: "상대의 취미는 가리지 않음(독서 제외)"
          (× "독서를 제외한 어떤 취미라도 함께 즐기는 사람" — 본인의 태도를 상대가 갖춰야 할 조건으로 바꾼 오류다.)
   - want 글에서 본인을 소개하는 말("저는 키가 작아요")은 상대에게 원하는 모습이 아니므로 넣지 않는다.
   - have 글에서 상대에게 바라는 말은 본인이 가진 모습이 아니므로 넣지 않는다.
8. 방향이 want이면 "상대에게 원하는 모습", have이면 "본인이 가진 모습"을 같은 키에 담는다.

## 기준
{_CRITERIA_LINES}

{_render_examples()}
""".strip()


def _user_message(side: str, text: str, feedback: str = "") -> str:
    label = "상대에게 원하는 모습(want)" if side == "want" else "본인이 가진 모습(have)"
    body = f"방향: {label}\n\n## 원문\n{text}"
    if feedback:
        body += (
            f"\n\n## 이전 시도의 문제\n{feedback}\n"
            "evidence는 원문에 실제로 적힌 글자 그대로, statement는 규칙에 맞게 다시 답한다."
        )
    return body


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


# 프롬프트·기준·스키마 중 무엇이든 바뀌면 버전이 바뀌어 재추출 대상이 된다.
EXTRACT_PROMPT_VERSION = "ext-v2-" + _hash(
    EXTRACT_SYSTEM,
    _user_message("want", "{text}", "{feedback}"),
    json.dumps(TraitExtraction.model_json_schema(), sort_keys=True, ensure_ascii=False),
)[:10]


# --------------------------------------------------------------------------- 검증


def normalize_for_match(text: str) -> str:
    """공백·구두점·기호를 지우고 유니코드와 대소문자를 맞춘다. 글자(L)·숫자(N)·결합 문자(M)만 남는다."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in folded if unicodedata.category(ch)[0] in "LNM")


def clean_value(value: Optional[str]) -> Optional[str]:
    """빈 문자열은 허용하지 않는다. 비었으면 None."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def find_violations(values: dict[str, Optional[dict]], source: str) -> list[str]:
    """기준별 {evidence, statement} 응답의 문제를 찾는다. 위반이 없으면 빈 리스트.

    - evidence: 비어 있지 않고, 각 구절이 원문의 부분 문자열이어야 한다. (공백·구두점·대소문자는 무시)
    - statement: 비어 있지 않고, MAX_STATEMENT_CHARS 이하이며, 거절 표현이 남아 있지 않아야 한다.
    """
    source_norm = normalize_for_match(source)
    violations = []
    for key in CRITERIA:
        value = values.get(key)
        if value is None:
            continue
        evidence = [piece for piece in (clean_value(p) for p in value.get("evidence") or []) if piece]
        if not evidence:
            violations.append(f"{key}: evidence가 비어 있다")
        for piece in evidence:
            norm = normalize_for_match(piece)
            if not norm:
                violations.append(f"{key}: evidence에 의미 있는 글자가 없다 {piece!r}")
            elif norm not in source_norm:
                violations.append(f"{key}: evidence가 원문에 없는 표현이다 {piece!r}")
        statement = clean_value(value.get("statement"))
        if statement is None:
            violations.append(f"{key}: statement가 비어 있다")
            continue
        if len(statement) > MAX_STATEMENT_CHARS:
            violations.append(f"{key}: statement가 {MAX_STATEMENT_CHARS}자를 넘는다 ({len(statement)}자). 짧은 한 구절로 정리한다")
        left = [marker for marker in REJECTION_MARKERS if marker in statement]
        if left:
            violations.append(
                f"{key}: statement에 거절 표현({', '.join(left)})이 남아 있다 {statement!r}. "
                "'~하지 않는 사람'이나 '적당히 ~한' 같은 정도 표현으로 바꿔 쓴다"
            )
    return violations


# --------------------------------------------------------------------------- 추출


@dataclass
class ExtractionOutcome:
    ok: bool
    traits: dict[str, Optional[str]]  # 기준 -> statement. ex_trait의 기준 컬럼에 저장되고 채점에 쓰인다.
    attempts: int
    error: Optional[str]
    raw: dict = field(default_factory=dict)
    evidence: dict[str, list[str]] = field(default_factory=dict)  # 기준 -> 원문 근거(사람 확인용). 채점에 쓰지 않는다.


def extract_traits(
    side: str,
    text: str,
    model: str,
    parse: Callable = parse_chat,
    max_retries: int = MAX_RETRIES,
) -> ExtractionOutcome:
    """사람 한 명의 한 side 텍스트에서 8개 기준을 추출한다. 검증에 실패하면 최대 max_retries번 다시 시도한다."""
    empty = {key: None for key in CRITERIA}
    attempt_errors: list[str] = []
    last_response: Optional[dict] = None
    feedback = ""

    for attempt in range(1, max_retries + 2):
        try:
            parsed, _raw = parse(
                model,
                EXTRACT_SYSTEM,
                [{"role": "user", "content": _user_message(side, text, feedback)}],
                TraitExtraction,
            )
        except Exception as err:  # API 오류·응답 거부 등. 다음 시도로 넘어간다.
            attempt_errors.append(f"시도 {attempt}: 호출 실패 — {type(err).__name__}: {err}")
            feedback = ""
            continue

        last_response = parsed.model_dump()
        violations = find_violations(last_response, text)
        if not violations:
            traits = {
                key: clean_value(last_response[key]["statement"]) if last_response.get(key) else None
                for key in CRITERIA
            }
            evidence = {
                key: [piece for piece in (clean_value(p) for p in last_response[key]["evidence"]) if piece]
                for key in CRITERIA
                if last_response.get(key)
            }
            return ExtractionOutcome(
                ok=True,
                traits=traits,
                attempts=attempt,
                error=None,
                raw={"response": last_response, "attempt_errors": attempt_errors},
                evidence=evidence,
            )
        attempt_errors.append(f"시도 {attempt}: " + "; ".join(violations))
        feedback = "\n".join(violations)

    return ExtractionOutcome(
        ok=False,
        traits=empty,
        attempts=max_retries + 1,
        error=" | ".join(attempt_errors),
        raw={"response": last_response, "attempt_errors": attempt_errors},
    )


# --------------------------------------------------------------------------- 읽기 (채점 쪽)


@dataclass(frozen=True)
class TraitRow:
    registration_id: str
    side: str
    extraction_ok: bool
    source_text: str
    prompt_version: str
    traits: dict


class TraitBook:
    """ex_trait 행 모음. (registration_id, side)로 찾는다."""

    def __init__(self, rows: dict[tuple[str, str], TraitRow]):
        self.rows = rows

    def get(self, registration_id: str, side: str) -> Optional[TraitRow]:
        return self.rows.get((registration_id, side))


def load_trait_book(conn, registration_ids: list[str]) -> TraitBook:
    columns = ", ".join(CRITERIA)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT registration_id, side, {columns}, extraction_ok, source_text, prompt_version
                FROM ex_trait
                WHERE registration_id = ANY(%s)
                """,
                (list(registration_ids),),
            )
            fetched = cur.fetchall()
    except pg_errors.UndefinedTable as err:
        conn.rollback()
        raise RuntimeError(
            "ex_trait 테이블이 없습니다. QRious/migrations/aurora/02_ex_trait.sql을 적용한 뒤 "
            "extract_ex_traits.py --execute로 추출해 주세요."
        ) from err
    except pg_errors.UndefinedColumn as err:
        conn.rollback()
        raise RuntimeError(
            "ex_trait에 기준 컬럼이 빠져 있습니다. QRious/migrations/aurora/04_ex_background.sql을 적용해 주세요."
        ) from err
    rows = {}
    for record in fetched:
        registration_id, side = record[0], record[1]
        traits = {key: record[2 + i] for i, key in enumerate(CRITERIA)}
        extraction_ok, source_text, prompt_version = record[2 + len(CRITERIA) :]
        rows[(registration_id, side)] = TraitRow(
            registration_id, side, extraction_ok, source_text, prompt_version, traits
        )
    return TraitBook(rows)


def usable_traits(text: Optional[str], row: Optional[TraitRow]) -> Optional[dict]:
    """채점에 쓸 수 있는 추출 결과. 원문이 없거나 추출이 유효하지 않으면 None."""
    if not text or not text.strip():
        return None
    if row is None or not row.extraction_ok:
        return None
    if row.source_text != text or row.prompt_version != EXTRACT_PROMPT_VERSION:
        return None
    return row.traits


def trait_problems(students: dict, book: TraitBook) -> list[str]:
    """원문은 있는데 유효한 추출 결과가 없는 (registration, side) 목록."""
    problems = []
    for registration_id, student in students.items():
        for side in SIDES:
            text = student.get(f"ex_{side}")
            if not text or not text.strip():
                continue
            row = book.get(registration_id, side)
            if usable_traits(text, row) is not None:
                continue
            if row is None:
                reason = "추출 안 됨"
            elif not row.extraction_ok:
                reason = "추출 실패"
            elif row.source_text != text:
                reason = "원문이 바뀜"
            else:
                reason = "프롬프트 버전이 다름"
            problems.append(f"{registration_id} {side}: {reason}")
    return problems
