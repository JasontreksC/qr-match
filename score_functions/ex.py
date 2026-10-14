"""ex 점수 (자유 텍스트 매력 비교) — 원문 통째 평가 v3.

흐름
  1. 방향마다 A의 이상형 원문(ex_want)과 B의 매력 원문(ex_have)을 그대로 LLM에 넣는다.
     사람마다 기준별 구절을 미리 뽑아 두지 않는다. (추출하면서 원문 정보가 빠지는 일을 없앤다.)
  2. LLM은 8개 기준마다 이상형 원문의 요구 구절(want_quote)과 매력 원문의 관련 구절(have_quote)을
     원문 그대로 인용하고, 이유(reason)를 쓴 뒤 라벨을 고른다.
       요구 없음  이상형 원문에 그 기준의 요구가 없음 → n에서 뺀다
       정보 없음  요구는 있는데 매력 원문에 관련 내용이 없음 → 0점
       동일 1 / 거의 일치 0.75 / 부분 일치 0.5 / 약한 관련 0.25 / 무관/충돌 0
     인용이 원문에 있는지, 라벨이 인용과 맞는지는 코드가 검증하고, 틀리면 이유를 알려 주고 다시 묻는다.
  3. 같은 두 원문은 독립 호출 3회로 기준별 합의를 낸다. (요구 여부는 과반, 점수는 중앙값)
     의견이 갈린 기준이 있으면 2회 더(총 5회) 뽑는다. 결과는 캐시(ex_pair_score)에 남아 다시 돌려도 같다.
  4. 방향 점수:  dir(A→B) = Σ(요구한 기준의 점수) / (n + k)   (n = 요구 없음이 아닌 기준 수)
  5. ex 점수와 가중치 규칙(weight_mode):
       both          양방향 모두 정의됨   ex = (dir_AB + dir_BA) / 2
       one_direction 한쪽만 정의됨        ex = 그 점수 / 2   (정의되지 않은 방향은 0으로 보고 평균)
       none          양방향 모두 없음      ex 항을 빼고 최종 점수를 다시 계산
"""

import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Optional

from psycopg import errors as pg_errors
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from score_functions.ex_traits import CRITERIA, CRITERION_HINT, CRITERION_KO, normalize_for_match
from score_functions.llm import parse_chat

EX_DETAIL_VERSION = "ex-v3"

NOT_REQUESTED = "요구 없음"
NO_INFO = "정보 없음"
MATCH_LABELS = ("동일", "거의 일치", "부분 일치", "약한 관련", "무관/충돌")
Label = Literal["요구 없음", "정보 없음", "동일", "거의 일치", "부분 일치", "약한 관련", "무관/충돌"]

# 모델은 점수를 고르지 않는다. 숫자는 여기서 고정한다. 요구 없음은 점수가 없고 n에서 빠진다.
SCORE_BY_LABEL: dict[str, float] = {
    NO_INFO: 0.0,
    "동일": 1.0,
    "거의 일치": 0.75,
    "부분 일치": 0.5,
    "약한 관련": 0.25,
    "무관/충돌": 0.0,
}

SAMPLES = 3  # 원문 쌍마다 독립 호출 수
RESAMPLE_EXTRA = 2  # 의견이 갈리면 추가로 뽑는 수 (총 5회)
RESAMPLE_SPREAD = 0.5  # 요구가 있다고 본 호출들의 max - min 이 이 값 이상이면 재샘플
MAX_CALL_RETRIES = 2  # 응답 검증에 실패한 호출을 다시 부르는 최대 횟수
DEFAULT_K = 1.0  # 분모 보정 합계/(n+k). 검증 세트로 확정하기 전의 시작값
DEFAULT_WORKERS = 6
EMPTY_HAVE = "(적지 않음)"


class ExScoringError(RuntimeError):
    pass


# --------------------------------------------------------------------------- 프롬프트

# 라벨 앵커: (이상형 구절, 매력 구절, 라벨). 실제 텍스트를 본 뒤 보강한다.
# 이 목록은 프롬프트에 들어가므로, 바꾸면 PAIR_PROMPT_VERSION이 바뀌어 캐시 키가 달라진다.
ANCHORS: dict[str, list[tuple[str, str, str]]] = {
    "impression": [
        ("강아지상", "순하고 귀여운 인상", "동일"),
        ("강아지상", "다정한 인상", "부분 일치"),
        ("강아지상", "고양이상", "무관/충돌"),
    ],
    "appearance": [
        ("잘생긴 사람", "이목구비 뚜렷함", "거의 일치"),
    ],
    "personality": [
        ("배려심 많은", "잘 챙겨주는", "거의 일치"),
        ("내성적인", "활발한", "무관/충돌"),
        ("잘 맞춰주는", "상대에게 잘 맞춰주는 편", "동일"),
        ("성격이 진짜x100 다정한 사람", "다정하고 배려심이 깊다는 말을 들어요", "거의 일치"),
        ("성격이 진짜x100 다정한 사람", "다정한 편이에요", "부분 일치"),
        ("진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고", "진중하지만 유머도 있어요", "거의 일치"),
        ("진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고", "항상 엄청 진지하고 무거운 편", "부분 일치"),
        ("진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고", "장난기 많고 가벼운 편", "무관/충돌"),
    ],
    "vibe_style": [
        ("깔끔한 스타일", "단정한 옷차림", "동일"),
    ],
    "interests": [
        ("영화", "영화 감상, 넷플릭스", "동일"),
        ("영화", "전시회", "약한 관련"),
        ("상대가 좋다면 어떤 취미라도 즐겨요. 독서빼고요", "영화 감상, 전시회", "거의 일치"),
        ("상대가 좋다면 어떤 취미라도 즐겨요. 독서빼고요", "독서, 글쓰기", "무관/충돌"),
    ],
    "relationship_values": [
        ("연락 자주 하는 편", "연락을 자주 해요", "동일"),
        ("잘 안아주는 사람", "스킨십을 좋아하고 애정표현이 많은 편", "거의 일치"),
    ],
    "lifestyle": [
        ("담배 안 피는", "비흡연자", "동일"),
        ("술 자주 안 마시는", "가끔만 마시는", "거의 일치"),
        ("아침형", "일찍 일어나는", "동일"),
        ("운동 좋아하는", "헬스 다니는", "거의 일치"),
        ("야행성", "아침형", "무관/충돌"),
    ],
    "background": [
        ("군필", "군필이고 IT 회사 다녀요", "동일"),
        ("군필", "미필", "무관/충돌"),
        ("개발자", "IT 회사 다녀요", "부분 일치"),
        ("대학원생", "서울 소재 대학 석사 과정", "동일"),
    ],
}

# 프롬프트에 싣는 전체 예시. 테스트가 같은 데이터를 검증한다.
EXAMPLE_WANT = "잘 웃고 배려심 있는 사람이면 좋겠어요. 담배 안 피는 사람이 좋고 키 큰 사람이면 좋겠어요. 저는 낯을 좀 가려요."
EXAMPLE_HAVE = "배려심 깊다는 말 자주 들어요. 비흡연자고 운동 좋아해요."
# 기준 -> (want_quote, have_quote, reason, label). 적지 않은 기준은 요구 없음.
EXAMPLE_ANSWER: dict[str, tuple[list[str], list[str], str, str]] = {
    "appearance": (["키 큰 사람이면 좋겠어요"], [], "키에 대한 내용이 매력 원문에 없다.", NO_INFO),
    "personality": (
        ["잘 웃고 배려심 있는 사람이면 좋겠어요"],
        ["배려심 깊다는 말 자주 들어요"],
        "배려심은 충족하지만 잘 웃는지는 적혀 있지 않다.",
        "부분 일치",
    ),
    "lifestyle": (["담배 안 피는 사람이 좋고"], ["비흡연자고"], "담배를 피우지 않는다는 같은 내용이다.", "동일"),
}
EXAMPLE_NOTE = (
    '나머지 기준은 요구 없음. "저는 낯을 좀 가려요"는 본인 소개라 요구가 아니고, '
    '"운동 좋아해요"는 어느 요구와도 관련이 없어 근거로 쓰지 않는다.'
)

_ANCHOR_LINES = "\n".join(
    f"### {key} ({CRITERION_KO[key]}): {CRITERION_HINT[key]}\n"
    + "\n".join(f"- 「{want}」 / 「{have}」 → {label}" for want, have, label in ANCHORS[key])
    for key in CRITERIA
)


def _render_example() -> str:
    lines = [f'이상형 원문: "{EXAMPLE_WANT}"', f'매력 원문: "{EXAMPLE_HAVE}"']
    for key in CRITERIA:
        if key not in EXAMPLE_ANSWER:
            continue
        want, have, reason, label = EXAMPLE_ANSWER[key]
        shown_want = ", ".join(f'"{piece}"' for piece in want)
        shown_have = ", ".join(f'"{piece}"' for piece in have)
        lines.append(f'- {key}: want_quote [{shown_want}] / have_quote [{shown_have}] / reason "{reason}" → {label}')
    lines.append(EXAMPLE_NOTE)
    return "\n".join(lines)


PAIR_SYSTEM = f"""
당신은 소개팅 매칭을 위한 평가기다. 한 사람이 상대에게 원하는 모습을 자유롭게 쓴 "이상형 원문"과, 다른 사람이 자기 모습을 자유롭게 쓴 "매력 원문"을 통째로 읽고, 8개 기준마다 이상형 원문의 요구를 매력 원문이 얼마나 충족하는지 라벨로 고른다. 점수는 매기지 않는다. 숫자는 코드가 정한다.

## 순서 (8개 기준마다)
1. want_quote: 이상형 원문에서 이 기준에 해당하는 요구를 찾아 원문 글자 그대로 복사한다. 요구가 없으면 빈 목록이고 label은 "요구 없음"이다.
2. have_quote: 요구가 있으면 매력 원문 전체에서 그 요구와 관련된 구절을 찾아 원문 글자 그대로 복사한다. 관련된 내용이 없으면 빈 목록이고 label은 "정보 없음"이다.
3. reason: 두 인용만 근거로 판단 이유를 한 문장으로 쓴다. 요구 없음이면 "요구 없음"처럼 짧게 쓴다.
4. label: 두 인용이 모두 있으면 아래 5단계 라벨 중 하나를 고른다.

## 인용 규칙
- 원문에 적힌 글자를 그대로 복사한다. 요약하거나 단어를 바꾸거나 문법을 고치지 않는다. 필요한 부분만 짧게 복사하고, 떨어져 있으면 여러 개로 나눈다.
- 이상형 원문의 한 구절은 가장 알맞은 기준 하나에만 넣는다. 같은 구절을 두 기준의 want_quote에 넣지 않는다.
- 매력 원문의 구절은 기준 이름에 얽매이지 않는다. 요구와 관련이 있으면 어느 구절이든 근거가 되고, 같은 구절을 여러 기준의 근거로 써도 된다. 예) 요구 「운동하는 사람」(lifestyle)의 근거로 매력 「헬스가 취미예요」를 쓴다.
- 매력 원문이 "{EMPTY_HAVE}"이면 요구가 있는 기준은 모두 정보 없음이다.

## 이상형 원문 읽기
- 글을 끝까지 구절 단위로 훑어서 요구를 하나도 빠뜨리지 않는다. 짧거나 낯선 구절("175 이상", "티키타카", "츤데레", "테토남")도 요구다.
- 상대에게 바라는 모습만 요구다. 글쓴이가 본인을 소개하는 말("저는 키가 작아요", "저는 낯을 좀 가려요")은 요구가 아니다. "저는 ~한 사람이 좋아요", "저는 ~한 사람은 어려워요"는 상대의 모습을 말하므로 요구다.
- 거절은 "~하지 않는" 요구로 읽는다. 예) "담배 피는 사람 싫어요" → 담배를 피우지 않는 사람
- 한도는 정도가 정해진 요구로 읽는다. 예) "진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고" → 적당히 진지한 사람
- 글쓴이 본인의 너그러움·허용("상대가 좋다면 어떤 취미라도 즐겨요", "뭐든 같이 해요")은 그 기준에 조건이 없다는 뜻이다. 함께 적은 제외 대상("독서빼고요")만 조건이다.
- 강조("진짜", "완전", "개", "ㄹㅇ", "엄청", "x100000", "진짜진짜", "다정다정", "!!!")는 그 특성이 특히 중요하다는 뜻이다.
- 확신 없는 말("~맞죠?", "~잘 모르겠어요", "~일까요?")은 요구로 보지 않는다. "~같아요"는 허용한다.
- 결혼관(결혼에 대한 생각·결혼 계획)은 어느 기준의 요구로도 보지 않는다.
- 기준 나누기: 가치관(인생관·가족관 등)과 대화·소통 방식(말이 잘 통함, 티키타카, 공감)은 personality다. relationship_values는 연락 빈도·애정표현·스킨십 같은 연애관만이다. 군필 여부·직업·학력·소속·거주지 같은 사실 정보는 background다. 어느 기준에도 가깝지 않은 말은 넣지 않는다.

## 매력 원문 읽기
- 글쓴이 본인이 가진 모습만 근거다. 매력 원문에서 상대에게 바라는 말은 근거가 아니다.
- 적혀 있지 않은 성격·외모·습관·사실을 추측하지 않는다.
- 확신 없는 말("키가 작은 것도 매력 맞죠..?", "장점이 뭔지는 잘 모르겠어요")은 근거로 쓰지 않는다. "~같아요"는 허용한다.

## 라벨 고르기
- 그 기준의 요구 전체를 얼마나 충족하는지로 고른다. 요구가 여럿인데 일부만 충족하면 낮춘다.
- 매력 원문이 요구를 충족하면, 더 구체적으로 쓰여 있어도 높은 라벨이다.
- 요구에 정도나 한도를 나타내는 말("적당히", "너무 ~하지 않게")이 있으면 그 정도까지 맞아야 높은 라벨이다. 한도를 넘거나 정반대이면 낮춘다.
- 강조된 요구는 매력 원문이 그 특성을 분명하고 강하게 갖췄을 때 높은 라벨이고, 약하게만 갖췄으면 낮춘다.
- 조건이 없는 요구(가리지 않음)는 매력 원문의 내용이 제외 대상에 해당하지 않으면 거의 일치, 해당하면 무관/충돌이다.
- 군필·직업·학력 같은 사실 정보(background)는 같은 사실이 분명할 때만 높은 라벨이다. 서로 다른 사실이면 낮은 라벨이다.
- 두 라벨 사이에서 망설여지면 낮은 쪽을 고른다.

## 라벨
- 요구 없음: 이상형 원문에 이 기준의 요구가 없다
- 정보 없음: 요구는 있는데 매력 원문에 관련된 내용이 없다
- 동일: 표현만 다르고 같은 의미
- 거의 일치: 핵심은 같지만 정도나 범위가 조금 다름
- 부분 일치: 같은 방향이지만 구체성이 다름
- 약한 관련: 관련은 있으나 일치라고 하기 어려움
- 무관/충돌: 관련이 없거나 반대

## 기준과 라벨 예시 (이상형 / 매력 → 라벨)
{_ANCHOR_LINES}

## 전체 예시
{_render_example()}
""".strip()


class CriterionJudgement(BaseModel):
    # 인용을 먼저 쓰게 해서, 라벨이 두 원문의 근거에서 벗어나지 않게 한다.
    want_quote: list[str] = Field(
        description="이상형 원문에서 이 기준의 요구를 원문 글자 그대로 복사한 구절들. 요구가 없으면 빈 목록."
    )
    have_quote: list[str] = Field(
        description="매력 원문에서 그 요구와 관련된 구절을 원문 글자 그대로 복사한 것들. 요구가 없거나 관련 내용이 없으면 빈 목록."
    )
    reason: str = Field(description="두 인용만 근거로 라벨을 고른 이유 한 문장. 라벨보다 먼저 쓴다. 요구 없음이면 짧게.")
    label: Label = Field(description="요구 없음 / 정보 없음 / 5단계 라벨 중 하나. 망설여지면 낮은 쪽.")


class PairEvaluation(BaseModel):
    impression: CriterionJudgement
    appearance: CriterionJudgement
    personality: CriterionJudgement
    vibe_style: CriterionJudgement
    interests: CriterionJudgement
    relationship_values: CriterionJudgement
    lifestyle: CriterionJudgement
    background: CriterionJudgement


assert tuple(PairEvaluation.model_fields) == CRITERIA, "PairEvaluation과 ex_traits.CRITERIA가 어긋났습니다."


def user_message(want: str, have: str, feedback: str = "") -> str:
    body = f"## 이상형 원문 (상대에게 원하는 모습)\n{want}\n\n## 매력 원문 (스스로 적은 자기 모습)\n{have or EMPTY_HAVE}"
    if feedback:
        body += (
            f"\n\n## 이전 답의 문제\n{feedback}\n"
            "인용은 원문에 적힌 글자 그대로 복사하고, 라벨은 인용과 맞게 다시 답한다."
        )
    return body


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


# 프롬프트·앵커·라벨·스키마 중 무엇이든 바뀌면 버전이 바뀌고, 그러면 캐시 키가 달라진다.
PAIR_PROMPT_VERSION = "pair-v1-" + _hash(
    PAIR_SYSTEM,
    user_message("{want}", "{have}", "{feedback}"),
    user_message("{want}", ""),
    json.dumps(PairEvaluation.model_json_schema(), sort_keys=True, ensure_ascii=False),
    json.dumps(SCORE_BY_LABEL, sort_keys=True, ensure_ascii=False),
)[:10]


def pair_cache_key(want: str, have: str, model: str, prompt_version: str = PAIR_PROMPT_VERSION) -> str:
    return _hash(want, have, prompt_version, model)


# --------------------------------------------------------------------------- 검증


def _text(value: Optional[str]) -> str:
    return (value or "").strip()


def _quotes(value) -> list[str]:
    return [piece.strip() for piece in (value or []) if piece and piece.strip()]


def _quote_violations(key: str, field_name: str, title: str, pieces: list[str], source_norm: str) -> list[str]:
    found = []
    for piece in pieces:
        norm = normalize_for_match(piece)
        if not norm:
            found.append(f"{key}: {field_name}에 의미 있는 글자가 없다 {piece!r}")
        elif norm not in source_norm:
            found.append(f"{key}: {field_name}가 {title} 원문에 없는 표현이다 {piece!r}")
    return found


def find_violations(evaluation: dict, want: str, have: str) -> list[str]:
    """기준별 응답의 문제를 찾는다. 위반이 없으면 빈 리스트.

    - 인용은 해당 원문의 부분 문자열이어야 한다. (공백·구두점·대소문자는 무시)
    - 이상형 원문의 같은 구절을 두 기준에 넣지 않는다. (요구가 두 번 세어지지 않게)
    - 라벨은 인용과 맞아야 한다.
        요구 없음 ↔ want_quote 없음
        정보 없음 → want_quote 있음, have_quote 없음
        5단계     → want_quote, have_quote 모두 있음
    """
    want_norm = normalize_for_match(want)
    have_norm = normalize_for_match(have)
    violations: list[str] = []
    used: dict[str, str] = {}
    for key in CRITERIA:
        item = evaluation[key]
        label = item["label"]
        wants = _quotes(item.get("want_quote"))
        haves = _quotes(item.get("have_quote"))
        violations += _quote_violations(key, "want_quote", "이상형", wants, want_norm)
        violations += _quote_violations(key, "have_quote", "매력", haves, have_norm)
        for piece in wants:
            norm = normalize_for_match(piece)
            if not norm:
                continue
            if norm in used:
                violations.append(
                    f"{key}: want_quote {piece!r}를 {used[norm]}에도 넣었다. 이상형 원문의 한 구절은 가장 알맞은 기준 하나에만 넣는다"
                )
            else:
                used[norm] = key
        if label == NOT_REQUESTED:
            if wants:
                violations.append(f"{key}: want_quote가 있는데 요구 없음을 골랐다. 요구가 아니면 want_quote를 비운다")
        elif not wants:
            violations.append(f"{key}: want_quote 없이 {label}을(를) 골랐다. 이상형 원문에 이 기준의 요구가 없으면 요구 없음이다")
        elif label == NO_INFO:
            if haves:
                violations.append(f"{key}: have_quote가 있는데 정보 없음을 골랐다. 관련 구절이 있으면 5단계 라벨 중 하나를 고른다")
        elif not haves:
            violations.append(f"{key}: have_quote 없이 {label}을(를) 골랐다. 매력 원문에 관련 내용이 없으면 정보 없음이다")
    return violations


def clean_evaluation(evaluation: dict) -> dict[str, dict]:
    """검증을 통과한 응답을 기준별 {label, want_quote, have_quote, reason}으로 정리한다."""
    cleaned = {}
    for key in CRITERIA:
        item = evaluation[key]
        label = item["label"]
        cleaned[key] = {
            "label": label,
            "want_quote": [] if label == NOT_REQUESTED else _quotes(item.get("want_quote")),
            "have_quote": _quotes(item.get("have_quote")) if label in MATCH_LABELS else [],
            "reason": (item.get("reason") or "").strip(),
        }
    return cleaned


# --------------------------------------------------------------------------- LLM 호출


def make_openai_judge(model: str, parse: Callable = parse_chat, max_retries: int = MAX_CALL_RETRIES) -> Callable:
    """두 원문을 한 번 평가하는 호출. 검증에 실패하면 문제를 알려 주고 max_retries번까지 다시 부른다."""

    def judge(want: str, have: str) -> dict[str, dict]:
        errors: list[str] = []
        feedback = ""
        for attempt in range(1, max_retries + 2):
            try:
                parsed, _raw = parse(
                    model,
                    PAIR_SYSTEM,
                    [{"role": "user", "content": user_message(want, have, feedback)}],
                    PairEvaluation,
                )
            except Exception as err:  # noqa: BLE001 — API 오류·응답 거부 등. 다음 시도로 넘어간다.
                errors.append(f"시도 {attempt}: 호출 실패 — {type(err).__name__}: {err}")
                feedback = ""
                continue
            evaluation = parsed.model_dump()
            violations = find_violations(evaluation, want, have)
            if not violations:
                return clean_evaluation(evaluation)
            errors.append(f"시도 {attempt}: " + "; ".join(violations))
            feedback = "\n".join(violations)
        raise ExScoringError("ex 평가 호출이 실패했습니다: " + " | ".join(errors))

    return judge


# --------------------------------------------------------------------------- 합의와 캐시


def _consensus(drawn: list[dict]) -> tuple[dict, bool]:
    """한 기준의 호출별 판단을 합친다. (합의 결과, 의견이 갈렸는가)

    요구 여부는 과반으로 정한다. 요구가 있으면 요구가 있다고 본 호출들의 점수 중앙값(짝수 개면 낮은 쪽)을 쓰고,
    라벨·인용·이유는 그 점수를 낸 호출 중 가장 많이 나온 라벨의 첫 호출에서 가져온다.
    (정보 없음과 무관/충돌은 둘 다 0점이라 라벨로 구분한다.)
    """
    total = len(drawn)
    requested = [sample for sample in drawn if sample["label"] != NOT_REQUESTED]
    votes = len(requested)
    if votes * 2 <= total:
        first = next(sample for sample in drawn if sample["label"] == NOT_REQUESTED)
        result = {
            "label": NOT_REQUESTED,
            "score": None,
            "want_quote": [],
            "have_quote": [],
            "reason": first["reason"],
            "votes": votes,
            "spread": 0.0,
        }
        return result, votes > 0
    scores = sorted(SCORE_BY_LABEL[sample["label"]] for sample in requested)
    score = scores[(len(scores) - 1) // 2]
    spread = scores[-1] - scores[0]
    at_median = [sample for sample in requested if SCORE_BY_LABEL[sample["label"]] == score]
    label = Counter(sample["label"] for sample in at_median).most_common(1)[0][0]
    chosen = next(sample for sample in at_median if sample["label"] == label)
    result = {
        "label": label,
        "score": score,
        "want_quote": chosen["want_quote"],
        "have_quote": chosen["have_quote"],
        "reason": chosen["reason"],
        "votes": votes,
        "spread": spread,
    }
    return result, votes < total or spread >= RESAMPLE_SPREAD


def consensus(samples: list[dict[str, dict]]) -> tuple[dict[str, dict], list[str]]:
    """호출별 평가들을 기준별 합의로 묶는다. (기준 -> 합의 결과, 의견이 갈린 기준들)"""
    criteria: dict[str, dict] = {}
    disputed: list[str] = []
    for key in CRITERIA:
        result, split = _consensus([sample[key] for sample in samples])
        criteria[key] = result
        if split:
            disputed.append(key)
    return criteria, disputed


@dataclass
class PairResult:
    cache_key: str
    criteria: dict[str, dict]  # 기준 -> {label, score, want_quote, have_quote, reason, votes, spread}
    calls: int
    resampled: bool
    model: str
    prompt_version: str
    want_text: str = ""
    have_text: str = ""
    samples: list = field(default_factory=list)  # 새로 평가한 결과에만 있다. 캐시에서 읽은 결과는 비어 있다.


class MemoryPairStore:
    """ex_pair_score 캐시의 메모리 구현. DB 구현의 부모이기도 하다."""

    def __init__(self) -> None:
        self._rows: dict[str, PairResult] = {}

    def get(self, key: str) -> Optional[PairResult]:
        return self._rows.get(key)

    def put(self, result: PairResult) -> PairResult:
        """이미 있으면 먼저 저장된 값을 유지한다. (한 번 매긴 평가는 바뀌지 않는다.)"""
        return self._rows.setdefault(result.cache_key, result)

    def __len__(self) -> int:
        return len(self._rows)


class AuroraPairStore(MemoryPairStore):
    """Aurora ex_pair_score. 현재 모델·프롬프트 버전의 행을 메모리에 올려 두고, 새 결과만 쓴다."""

    def __init__(self, conn, model: str, prompt_version: str = PAIR_PROMPT_VERSION) -> None:
        super().__init__()
        self.conn = conn
        self.model = model
        self.prompt_version = prompt_version

    def load(self) -> int:
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT cache_key, criteria, jsonb_array_length(samples), resampled
                    FROM ex_pair_score
                    WHERE model = %s AND prompt_version = %s
                    """,
                    (self.model, self.prompt_version),
                )
                fetched = cur.fetchall()
        except pg_errors.UndefinedTable as err:
            self.conn.rollback()
            raise ExScoringError(
                "ex_pair_score 테이블이 없습니다. QRious/migrations/aurora/05_ex_pair_score.sql을 적용해 주세요."
            ) from err
        self.conn.commit()
        for cache_key, criteria, calls, resampled in fetched:
            self._rows[cache_key] = PairResult(cache_key, criteria, calls, resampled, self.model, self.prompt_version)
        return len(fetched)

    def put(self, result: PairResult) -> PairResult:
        existing = self._rows.get(result.cache_key)
        if existing is not None:
            return existing
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ex_pair_score (
                  cache_key, want_text, have_text, criteria, samples, resampled, model, prompt_version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cache_key) DO NOTHING
                """,
                (
                    result.cache_key,
                    result.want_text,
                    result.have_text,
                    Jsonb(result.criteria),
                    Jsonb(result.samples),
                    result.resampled,
                    result.model,
                    result.prompt_version,
                ),
            )
            stored = result
            if cur.rowcount != 1:
                # 다른 실행이 먼저 저장했다. 그 값을 따른다.
                cur.execute(
                    "SELECT criteria, jsonb_array_length(samples), resampled FROM ex_pair_score WHERE cache_key = %s",
                    (result.cache_key,),
                )
                criteria, calls, resampled = cur.fetchone()
                stored = PairResult(result.cache_key, criteria, calls, resampled, result.model, result.prompt_version)
        self.conn.commit()
        self._rows[stored.cache_key] = stored
        return stored


class PairScorer:
    """원문 쌍 평가: 캐시 조회 → 없으면 3회 합의(+재샘플) → 저장."""

    def __init__(
        self,
        store: MemoryPairStore,
        model: str,
        judge: Optional[Callable] = None,
        prompt_version: str = PAIR_PROMPT_VERSION,
    ) -> None:
        self.store = store
        self.model = model
        self.prompt_version = prompt_version
        self.judge = judge or make_openai_judge(model)

    def key(self, want: str, have: str) -> str:
        return pair_cache_key(want, have, self.model, self.prompt_version)

    def compute(self, want: str, have: str) -> PairResult:
        """두 원문을 합의로 평가한다. 저장하지 않는다. (스레드에서 불러도 안전하다.)"""
        samples = [self.judge(want, have) for _ in range(SAMPLES)]
        criteria, disputed = consensus(samples)
        resampled = bool(disputed)
        if disputed:
            samples += [self.judge(want, have) for _ in range(RESAMPLE_EXTRA)]
            criteria, _ = consensus(samples)
            unstable = [key for key in CRITERIA if criteria[key]["spread"] >= RESAMPLE_SPREAD]
            if unstable:
                print(
                    f"경고: ex 평가 합의가 불안정합니다 (호출 {len(samples)}회, 기준 {', '.join(unstable)}): "
                    f"{want[:30]!r} / {have[:30]!r}",
                    file=sys.stderr,
                )
        return PairResult(
            cache_key=self.key(want, have),
            criteria=criteria,
            calls=len(samples),
            resampled=resampled,
            model=self.model,
            prompt_version=self.prompt_version,
            want_text=want,
            have_text=have,
            samples=samples,
        )

    def evaluate(self, want: str, have: str) -> PairResult:
        """캐시에 있으면 그대로, 없으면 합의로 평가해서 저장한다."""
        cached = self.store.get(self.key(want, have))
        if cached is not None:
            return cached
        return self.store.put(self.compute(want, have))

    def pending(self, directed: Iterable[tuple[Optional[str], Optional[str]]]) -> list[tuple[str, str]]:
        """(이상형 원문, 매력 원문) 방향 쌍 중 아직 평가가 없는 것. 같은 두 원문은 한 번만 센다.

        이상형 원문이 없으면 평가할 요구가 없으므로 뺀다. 매력 원문이 없으면 빈 문자열로 평가한다.
        """
        seen: set[str] = set()
        todo: list[tuple[str, str]] = []
        for want_text, have_text in directed:
            want, have = _text(want_text), _text(have_text)
            if not want:
                continue
            key = self.key(want, have)
            if key in seen or self.store.get(key) is not None:
                continue
            seen.add(key)
            todo.append((want, have))
        return todo

    def prewarm(
        self,
        directed: Iterable[tuple[Optional[str], Optional[str]]],
        workers: int = DEFAULT_WORKERS,
        progress: Callable[[str], None] = print,
    ) -> int:
        """평가해야 할 원문 쌍을 병렬로 미리 평가해 저장한다. 이후 매칭 실행은 저장된 값만 읽는다."""
        todo = self.pending(directed)
        if not todo:
            progress(f"ex 평가: 모두 캐시에 있음 ({len(self.store)}건)")
            return 0
        progress(f"ex 평가: 새로 평가할 원문 쌍 {len(todo)}개 (LLM 호출 최소 {len(todo) * SAMPLES}회, 동시 {workers}개)")

        failures: list[str] = []
        saved = 0
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(self.compute, want, have): (want, have) for want, have in todo}
            for number, future in enumerate(as_completed(futures), start=1):
                try:
                    result = future.result()
                except Exception as err:  # noqa: BLE001 — 나머지는 계속 평가하고 마지막에 한꺼번에 알린다.
                    failures.append(f"{type(err).__name__}: {err}")
                    continue
                self.store.put(result)
                saved += 1
                mark = ", 재샘플" if result.resampled else ""
                progress(f"[{number}/{len(todo)}] 저장 (호출 {result.calls}회{mark}, 누적 {saved})")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if failures:
            raise ExScoringError(
                f"ex 평가 {len(failures)}건이 실패했습니다. 저장된 평가는 유지되므로 다시 실행하면 이어서 합니다.\n"
                + "\n".join(failures[:5])
            )
        return saved


# --------------------------------------------------------------------------- 방향 점수와 ex 점수


def denominator_k() -> float:
    raw = os.getenv("EX_DENOM_K")
    return float(raw) if raw not in (None, "") else DEFAULT_K


@dataclass
class DirectionResult:
    defined: bool
    n: int
    k: float
    score: Optional[float]
    calls: int
    resampled: bool
    criteria: list[dict]

    def payload(self) -> dict:
        return {
            "defined": self.defined,
            "score": None if self.score is None else round(self.score, 6),
            "n": self.n,
            "k": self.k,
            "calls": self.calls,
            "resampled": self.resampled,
            "criteria": self.criteria,
        }


def direction_score(
    want_text: Optional[str], have_text: Optional[str], scorer: PairScorer, k: float
) -> DirectionResult:
    """dir(A→B) = Σ(A가 요구한 기준의 점수) / (n + k).

    정의되는 방향: 이상형 원문에 요구가 1개 이상. (B가 관련 내용을 안 적어 0점이어도 정의된 점수 0이다.)
    정의되지 않는 방향: A가 ex_want를 안 썼거나, 평가 결과 요구가 0개.
    """
    want = _text(want_text)
    if not want:
        return DirectionResult(defined=False, n=0, k=k, score=None, calls=0, resampled=False, criteria=[])
    result = scorer.evaluate(want, _text(have_text))
    criteria = [
        {"criterion": key, **result.criteria[key]}
        for key in CRITERIA
        if result.criteria[key]["label"] != NOT_REQUESTED
    ]
    n = len(criteria)
    if n == 0:
        return DirectionResult(
            defined=False, n=0, k=k, score=None, calls=result.calls, resampled=result.resampled, criteria=[]
        )
    total = sum(item["score"] for item in criteria)
    return DirectionResult(
        defined=True,
        n=n,
        k=k,
        score=total / (n + k),
        calls=result.calls,
        resampled=result.resampled,
        criteria=criteria,
    )


@dataclass
class ExResult:
    score: float  # weight_mode == "none"이면 0.0 (최종 점수 계산에서 쓰이지 않는다)
    weight_mode: str  # both | one_direction | none
    detail: dict


def ex_score(
    male_want: Optional[str],
    male_have: Optional[str],
    female_want: Optional[str],
    female_have: Optional[str],
    scorer: PairScorer,
    k: Optional[float] = None,
) -> ExResult:
    """각 인자는 ex_want / ex_have 원문. 적지 않았으면 None이나 빈 문자열."""
    k = denominator_k() if k is None else k
    male_to_female = direction_score(male_want, female_have, scorer, k)
    female_to_male = direction_score(female_want, male_have, scorer, k)

    if male_to_female.defined and female_to_male.defined:
        mode = "both"
        score = (male_to_female.score + female_to_male.score) / 2
    elif male_to_female.defined or female_to_male.defined:
        mode = "one_direction"
        defined = male_to_female if male_to_female.defined else female_to_male
        score = defined.score / 2
    else:
        mode = "none"
        score = 0.0

    detail = {
        "version": EX_DETAIL_VERSION,
        "weight_mode": mode,
        "k": k,
        "model": scorer.model,
        "prompt_version": scorer.prompt_version,
        "male_to_female": male_to_female.payload(),
        "female_to_male": female_to_male.payload(),
    }
    return ExResult(score=score, weight_mode=mode, detail=detail)


def final_score(
    mbti: float,
    tag: float,
    ex: float,
    weight_mode: str,
    w_mbti: float,
    w_tag: float,
    w_ex: float,
) -> float:
    """최종 점수. ex 점수가 정의되지 않으면(none) ex 가중치를 빼고 다시 계산한다."""
    if weight_mode == "none":
        return (w_mbti * mbti + w_tag * tag) / (w_mbti + w_tag)
    return (w_mbti * mbti + w_tag * tag + w_ex * ex) / (w_mbti + w_tag + w_ex)
