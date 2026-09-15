import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

load_dotenv()

Relation = Literal["opposed", "similar", "match"]
MatchLabel = Literal["incomparable", "opposed", "similar", "match"]

# 모델은 점수를 고르지 않는다. 숫자는 여기서 고정한다.
SCORE_BY_LABEL: dict[str, float] = {
    "incomparable": 0.0,
    "opposed": 0.0,
    "similar": 0.5,
    "match": 1.0,
}

_client: OpenAI | None = None


class ExScore(BaseModel):
    want_evidence: str = Field(
        description=(
            "이상형 텍스트에서 판단에 사용한 원문 구절. "
            "없으면 빈 문자열."
        )
    )
    have_evidence: str = Field(
        description=(
            "매력 텍스트에서 판단에 사용한 원문 구절. "
            "없으면 빈 문자열."
        )
    )
    comparable: bool = Field(
        description=(
            "두 글이 같은 주제를 말해 비교할 수 있으면 true. "
            "다른 이야기이거나 한쪽 근거가 없으면 false."
        )
    )
    relation: Relation = Field(
        description=(
            "comparable이 true일 때만 의미가 있다. "
            "opposed: 같은 주제를 말하지만 서로 반대. "
            "similar: 관련은 있으나 같은 속성은 아님. "
            "match: 같은 속성을 거의 같은 말로 말함. "
            "comparable이 false이면 opposed로 두어도 점수는 0이다."
        )
    )
    reason: str = Field(
        description=(
            "want_evidence와 have_evidence만 근거로 comparable과 relation을 고른 짧은 이유. "
            "텍스트에 없는 성격·외모·습관을 추측하지 않는다."
        )
    )

    @model_validator(mode="after")
    def require_both_sides_to_compare(self):
        if not self.want_evidence.strip() or not self.have_evidence.strip():
            self.comparable = False
        return self

    def label(self) -> MatchLabel:
        if not self.comparable:
            return "incomparable"
        return self.relation

    def score(self) -> float:
        return SCORE_BY_LABEL[self.label()]

    def evidence(self) -> str:
        parts = [part.strip() for part in (self.want_evidence, self.have_evidence) if part.strip()]
        return "\n".join(parts)

    def payload(self) -> dict:
        return {
            "score": self.score(),
            "comparable": self.comparable,
            "relation": self.relation,
            "match_kind": self.label(),
            "want_evidence": self.want_evidence,
            "have_evidence": self.have_evidence,
            "evidence": self.evidence(),
            "reason": self.reason,
        }


def make_prompt(ex_want: str | None, ex_have: str | None) -> str:
    want = (ex_want or "").strip() or "(없음)"
    have = (ex_have or "").strip() or "(없음)"
    return f"""
당신은 소개팅 매칭을 위한 분류기다. 점수를 매기지 말고, 두 글을 통째로 읽어 두 단계로만 분류한다.

순서: (1) want_evidence 인용 → (2) have_evidence 인용 → (3) comparable 선택 → (4) comparable이 true이면 relation 선택 → (5) reason 작성.
주어진 두 텍스트에 적힌 내용만 사용한다. 없는 성격·외모·말투·유머·에너지를 추측하지 않는다.
성격·대화·에너지·유머·외모처럼 칸을 나눠 채점하지 않는다.

## 1단계 comparable
- true: 두 글이 같은 주제를 말해 비교할 수 있다.
- false: 다른 이야기이거나, 한쪽 글에 비교할 구절이 없다.
  예) 이상형은 경청을 원하고 매력은 요리 취미만 말한다 → comparable=false.

comparable이 false이면 relation은 opposed로 두고, 점수는 코드가 0으로 처리한다.

## 2단계 relation (comparable이 true일 때만)
- opposed: 같은 주제를 말하지만 서로 반대다. 예) 이상형 "말수가 적은 사람" / 매력 "수다스러움".
- similar: 관련은 있으나 같은 속성이 아니다. 예) 이상형 "집중력이 좋은 사람" / 매력 "상대방 말을 잘 들어줌".
- match: 같은 속성을 거의 같은 말로 말한다. 예) 이상형 "잘 들어주는 사람" / 매력 "상대방 말을 잘 들어줌".

want_evidence와 have_evidence가 둘 다 있을 때만 comparable=true가 될 수 있다.
match는 엄격하다. 조금이라도 바꿔 말한 관련성이면 similar다.
comparable=false를 similar로, similar를 match로 올리지 않는다.

## 첫 번째 사람이 원하는 이상형
{want}

## 두 번째 사람이 가진 매력
{have}
""".strip()


def _openai_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        _client = OpenAI()
    return _client


def _empty_score(reason: str) -> ExScore:
    return ExScore(
        want_evidence="",
        have_evidence="",
        comparable=False,
        relation="opposed",
        reason=reason,
    )


def _one_way_score(ex_want: str | None, ex_have: str | None) -> ExScore:
    want = (ex_want or "").strip()
    have = (ex_have or "").strip()
    if not want or not have:
        return _empty_score("이상형 또는 매력 텍스트가 없어 비교할 수 없다.")

    completion = _openai_client().chat.completions.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
        messages=[
            {
                "role": "system",
                "content": (
                    "JSON 스키마에 맞춰 두 글을 두 단계로만 분류한다. "
                    "텍스트에 없는 내용을 추측하지 않는다."
                ),
            },
            {"role": "user", "content": make_prompt(want, have)},
        ],
        response_format=ExScore,
        temperature=0,
        reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "none"),
    )
    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"모델이 응답을 거부함: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("모델이 구조화 응답을 반환하지 않음")
    return message.parsed


def _arithmetic_mean(a: float, b: float) -> float:
    return (a + b) / 2


def ex_score(exw_a: str, exh_a: str, exw_b: str, exh_b: str) -> tuple[float, dict]:
    a_to_b = _one_way_score(exw_a, exh_b)
    b_to_a = _one_way_score(exw_b, exh_a)
    detail = {
        "male_to_female": a_to_b.payload(),
        "female_to_male": b_to_a.payload(),
    }
    return _arithmetic_mean(a_to_b.score(), b_to_a.score()), detail
