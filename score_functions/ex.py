import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

MatchKind = Literal["none", "conflict", "indirect", "direct"]

# 모델은 1~5가 아니라 범주만 고른다. 숫자는 여기서 고정한다.
POINTS_BY_KIND: dict[str, int] = {
    "none": 0,
    "conflict": 1,
    "indirect": 3,
    "direct": 5,
}
MAX_POINTS = 5

_client: OpenAI | None = None


class CriterionScore(BaseModel):
    evidence: str = Field(
        description=(
            "이 기준을 판단하는 데 사용한 원문 구절. "
            "이상형/매력 텍스트에서 그대로 인용한다. "
            "관련 구절이 없으면 빈 문자열."
        )
    )
    match_kind: MatchKind = Field(
        description=(
            "none: 이 기준을 판단할 단서가 없다. "
            "conflict: 같은 주제를 말하지만 서로 반대이거나 충돌한다. "
            "indirect: 같은 방향의 관련성은 있으나 같은 속성이 아니다. "
            "direct: 이상형이 원하는 속성을 매력이 같은 말로 직접 충족한다."
        )
    )
    reason: str = Field(
        description=(
            "evidence만 근거로 match_kind를 고른 짧은 이유. "
            "텍스트에 없는 성격·외모·습관을 추측하지 않는다."
        )
    )

    def points(self) -> int:
        return POINTS_BY_KIND[self.match_kind]


class ExScore(BaseModel):
    combination: CriterionScore = Field(
        description="평가 항목 1번, 성격의 조화 (combination)"
    )
    conversation: CriterionScore = Field(
        description="평가 항목 2번, 대화 방식의 어울림 (conversation)"
    )
    energy: CriterionScore = Field(
        description="평가 항목 3번, 에너지와 분위기 (energy)"
    )
    humor: CriterionScore = Field(
        description="평가 항목 4번, 유머 코드 (humor)"
    )
    visual: CriterionScore = Field(
        description="평가 항목 5번, 외적인 취향 (visual)"
    )

    def criteria(self) -> list[tuple[str, CriterionScore]]:
        return [
            ("성격의 조화 (combination)", self.combination),
            ("대화 방식의 어울림 (conversation)", self.conversation),
            ("에너지와 분위기 (energy)", self.energy),
            ("유머 코드 (humor)", self.humor),
            ("외적인 취향 (visual)", self.visual),
        ]

    def avg_score(self) -> float:
        scored = [
            item.points()
            for _, item in self.criteria()
            if item.match_kind != "none"
        ]
        if not scored:
            return 0.0
        return sum(scored) / (len(scored) * MAX_POINTS)


def make_prompt(ex_want: str | None, ex_have: str | None) -> str:
    want = (ex_want or "").strip() or "(없음)"
    have = (ex_have or "").strip() or "(없음)"
    return f"""
당신은 소개팅 매칭을 위한 분류기다. 점수를 매기지 말고, 각 기준을 네 범주 중 하나로만 분류한다.

각 기준마다 순서: (1) evidence 인용 → (2) match_kind 선택 → (3) reason 작성.
주어진 두 텍스트에 적힌 내용만 사용한다. 없는 성격·외모·말투·유머·에너지를 추측하지 않는다.
짧은 자유 텍스트는 보통 1~2개 기준만 해당한다. 해당하지 않으면 none이다.

## 평가 기준과 단서의 범위
1. 성격의 조화 (combination): 성격, 태도, 가치관, 관계에서의 성향.
2. 대화 방식의 어울림 (conversation): 경청, 말수, 대화 습관, 소통 스타일.
3. 에너지와 분위기 (energy): 활기, 차분함, 텐션, 함께 있을 때의 분위기.
4. 유머 코드 (humor): 유머, 장난, 드립, 웃기는 방식.
5. 외적인 취향 (visual): 외모, 체형, 스타일, 패션 등 눈에 보이는 취향.

## match_kind 정의 (이 네 가지만 허용)
- none: 이상형과 매력 어느 쪽에도 이 기준의 단서가 없다.
- conflict: 같은 기준의 단서가 양쪽에 있지만 서로 반대다. 예) 이상형 "말수가 적은 사람" / 매력 "수다스러움".
- indirect: 관련은 있으나 같은 속성이 아니다. 비슷한 인상, 원인-결과, 한 단계 추론이 필요하면 전부 여기. 예) 이상형 "집중력이 좋은 사람" / 매력 "상대방 말을 잘 들어줌" → conversation=indirect. 집중력과 경청은 다르므로 direct가 아니다.
- direct: 같은 속성을 거의 같은 말로 말한다. 예) 이상형 "잘 들어주는 사람" / 매력 "상대방 말을 잘 들어줌".

direct의 기준은 엄격하다. 조금이라도 바꿔 말한 관련성이면 indirect다.
conflict가 아니면 우호적으로 올리지 않는다. none을 indirect로, indirect를 direct로 올리지 않는다.

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


def _one_way_score(ex_want: str | None, ex_have: str | None) -> float:
    completion = _openai_client().chat.completions.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
        messages=[
            {
                "role": "system",
                "content": (
                    "JSON 스키마에 맞춰 분류만 한다. "
                    "텍스트에 없는 내용을 추측하지 않는다."
                ),
            },
            {"role": "user", "content": make_prompt(ex_want, ex_have)},
        ],
        response_format=ExScore,
        reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "none"),
    )
    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"모델이 응답을 거부함: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("모델이 구조화 응답을 반환하지 않음")
    return message.parsed.avg_score()


def _harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def ex_score(exw_a: str, exh_a: str, exw_b: str, exh_b: str) -> float:
    # 테스트용. openai 크레딧 아끼기.
    import time, random
    time.sleep(1)
    return float(random.randint(0, 10) / 10)
    # a_to_b = _one_way_score(exw_a, exh_b)
    # b_to_a = _one_way_score(exw_b, exh_a)
    # return _harmonic_mean(a_to_b, b_to_a)
