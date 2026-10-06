"""자유 텍스트(ex_want / ex_have)를 7개 기준으로 추출하고, 추출 결과(ex_trait)를 읽는다.

- 추출은 사람(registration)당 side(want/have)당 한 번만 한다. 쌍마다 하지 않는다.
- 값은 원문에서 그대로 발췌한 구절이다. 요약·재서술은 코드가 거른다(부분 문자열 검증).
- 부정·기피 표현은 추출하지 않는다.
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
)
CRITERION_KO = {
    "impression": "인상",
    "appearance": "외모",
    "personality": "성격",
    "vibe_style": "분위기/스타일",
    "interests": "취미/관심사",
    "relationship_values": "연애관/가치관",
    "lifestyle": "생활습관",
}
CRITERION_HINT = {
    "impression": "얼굴상·첫인상·이미지. 예) 강아지상, 순한 인상",
    "appearance": "키·체형·이목구비 등 눈에 보이는 외형. 예) 잘생긴, 키 큰, 이목구비 뚜렷한",
    "personality": "성격·성향·태도. 예) 배려심 많은, 내성적인, 활발한",
    "vibe_style": "전체적인 분위기·옷차림·꾸밈 스타일. 예) 깔끔한 스타일, 단정한 옷차림",
    "interests": "취미·관심사·좋아하는 활동. 예) 영화 감상, 전시회, 카페 투어",
    "relationship_values": "연애관·가치관·연애 방식. 예) 연락을 자주 하는 편, 결혼관",
    "lifestyle": "흡연·음주·운동·기상 시간·식습관 같은 생활 습관. 예) 담배 안 피는, 아침형, 운동하는",
}
SIDES = ("want", "have")
# 같은 기준에 대한 언급이 원문 여러 곳에 흩어져 있을 때 발췌문을 잇는 구분자.
EXCERPT_SEPARATOR = " | "
MAX_RETRIES = 2  # 첫 시도 뒤 최대 2번 더 → 최대 3회


class TraitExtraction(BaseModel):
    impression: Optional[str] = Field(description=CRITERION_HINT["impression"] + ". 없으면 null.")
    appearance: Optional[str] = Field(description=CRITERION_HINT["appearance"] + ". 없으면 null.")
    personality: Optional[str] = Field(description=CRITERION_HINT["personality"] + ". 없으면 null.")
    vibe_style: Optional[str] = Field(description=CRITERION_HINT["vibe_style"] + ". 없으면 null.")
    interests: Optional[str] = Field(description=CRITERION_HINT["interests"] + ". 없으면 null.")
    relationship_values: Optional[str] = Field(
        description=CRITERION_HINT["relationship_values"] + ". 없으면 null."
    )
    lifestyle: Optional[str] = Field(description=CRITERION_HINT["lifestyle"] + ". 없으면 null.")


_CRITERIA_LINES = "\n".join(f"- {key} ({CRITERION_KO[key]}): {CRITERION_HINT[key]}" for key in CRITERIA)
_SEPARATOR_MARK = EXCERPT_SEPARATOR.strip()

EXTRACT_SYSTEM = f"""
당신은 소개팅 매칭을 위한 텍스트 구조화 도구다. 사용자가 자유롭게 쓴 글에서 7개 기준에 해당하는 구절을 골라 낸다.

## 규칙
1. 값은 원문에 적힌 표현을 그대로 복사한 발췌문이다. 요약, 재서술, 단어 바꾸기, 문법 고치기를 하지 않는다.
   - 필요한 만큼만 짧게 발췌한다. 한 문장 전체일 필요는 없다.
   - 같은 기준에 대한 언급이 떨어져 있으면 각 발췌문을 그대로 두고 "{_SEPARATOR_MARK}"로 잇는다. 이어붙이면서 글자를 바꾸지 않는다.
2. 부정·기피 표현은 어떤 기준에도 넣지 않는다. 예) "담배 피는 사람 싫어요", "술 많이 마시는 사람은 별로", "~는 안 좋아해요".
   - 긍정형으로 쓰인 요구는 넣는다. 예) "담배 안 피는 사람이 좋아요", "술 안 마시는 사람" → lifestyle.
3. 글에 언급이 없는 기준은 null이다. 없는 내용을 추측해서 채우지 않는다.
4. 한 구절은 가장 알맞은 기준 하나에만 넣는다.
5. 방향이 want이면 "상대에게 원하는 모습", have이면 "본인이 가진 모습"을 같은 키에 담는다.

## 기준
{_CRITERIA_LINES}

## 예시 (want)
글: "잘 웃고 배려심 있는 사람이면 좋겠어요. 담배 안 피는 사람이 좋고 영화 같이 보러 다니고 싶어요. 술 많이 마시는 사람은 싫어요."
→ personality: "잘 웃고 배려심 있는 사람", lifestyle: "담배 안 피는 사람이 좋고", interests: "영화 같이 보러 다니고 싶어요", 나머지 null. (술 이야기는 부정 표현이라 넣지 않는다.)

## 예시 (have)
글: "강아지상이라는 말 자주 들어요. 운동 좋아하고 아침형 인간이에요."
→ impression: "강아지상", lifestyle: "운동 좋아하고 아침형 인간", 나머지 null.
""".strip()


def _user_message(side: str, text: str, feedback: str = "") -> str:
    label = "상대에게 원하는 모습(want)" if side == "want" else "본인이 가진 모습(have)"
    body = f"방향: {label}\n\n## 원문\n{text}"
    if feedback:
        body += f"\n\n## 이전 시도의 문제\n{feedback}\n원문에 실제로 적힌 글자만 그대로 복사해서 다시 답한다."
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


def split_excerpts(value: str, source_norm: str) -> list[str]:
    """값이 통째로 원문에 있으면 한 덩어리, 아니면 구분자로 나눈 발췌문들."""
    if normalize_for_match(value) in source_norm:
        return [value]
    if "|" in value:
        return [part.strip() for part in value.split("|") if part.strip()]
    return [value]


def find_violations(traits: dict[str, Optional[str]], source: str) -> list[str]:
    """원문의 부분 문자열이 아닌 값을 찾는다. 위반이 없으면 빈 리스트."""
    source_norm = normalize_for_match(source)
    violations = []
    for key in CRITERIA:
        value = traits.get(key)
        if value is None:
            continue
        for excerpt in split_excerpts(value, source_norm):
            norm = normalize_for_match(excerpt)
            if not norm:
                violations.append(f"{key}: 의미 있는 글자가 없는 값 {excerpt!r}")
            elif norm not in source_norm:
                violations.append(f"{key}: 원문에 없는 표현 {excerpt!r}")
    return violations


# --------------------------------------------------------------------------- 추출


@dataclass
class ExtractionOutcome:
    ok: bool
    traits: dict[str, Optional[str]]
    attempts: int
    error: Optional[str]
    raw: dict = field(default_factory=dict)


def extract_traits(
    side: str,
    text: str,
    model: str,
    parse: Callable = parse_chat,
    max_retries: int = MAX_RETRIES,
) -> ExtractionOutcome:
    """사람 한 명의 한 side 텍스트에서 7개 기준을 추출한다. 검증에 실패하면 최대 max_retries번 다시 시도한다."""
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
        traits = {key: clean_value(last_response.get(key)) for key in CRITERIA}
        violations = find_violations(traits, text)
        if not violations:
            return ExtractionOutcome(
                ok=True,
                traits=traits,
                attempts=attempt,
                error=None,
                raw={"response": last_response, "attempt_errors": attempt_errors},
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
