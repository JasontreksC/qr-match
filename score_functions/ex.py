"""ex 점수 (자유 텍스트 매력 비교) — 개선안 v2.

흐름
  1. 추출(ex_traits): 사람·side마다 8개 기준으로 발췌해 둔 값(ex_trait)을 읽는다.
  2. 항목 비교: A가 요구한 항목마다 B가 가진 같은 기준 값과 비교한다.
     - B가 그 항목을 안 적었으면 LLM 없이 0점(MISSING).
     - 양쪽에 있으면 캐시(ex_item_score)를 먼저 보고, 없으면 독립 호출 3회 → 중앙값.
       3회의 max-min이 0.5 이상이면 2회 더(총 5회) 뽑아 중앙값을 다시 구한다.
     - LLM은 숫자가 아니라 5단계 라벨을 고르고, 숫자는 코드가 정한다.
  3. 방향 점수:  dir(A→B) = Σ(항목 점수) / (n + k)   (n = A가 요구한 항목 수)
  4. ex 점수와 가중치 규칙(weight_mode):
       both          양방향 모두 정의됨   ex = (dir_AB + dir_BA) / 2
       one_direction 한쪽만 정의됨        ex = 그 점수 / 2   (정의되지 않은 방향은 0으로 보고 평균)
       none          양방향 모두 없음      ex 항을 빼고 최종 점수를 다시 계산
"""

import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional, get_args

from psycopg import errors as pg_errors
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from score_functions.ex_traits import CRITERIA, CRITERION_HINT, CRITERION_KO
from score_functions.llm import parse_chat

EX_DETAIL_VERSION = "ex-v2"

Label = Literal["동일", "거의 일치", "부분 일치", "약한 관련", "무관/충돌"]
Criterion = Literal[
    "impression",
    "appearance",
    "personality",
    "vibe_style",
    "interests",
    "relationship_values",
    "lifestyle",
    "background",
]
assert get_args(Criterion) == CRITERIA, "Criterion과 ex_traits.CRITERIA가 어긋났습니다."

# 모델은 점수를 고르지 않는다. 숫자는 여기서 고정한다.
SCORE_BY_LABEL: dict[str, float] = {
    "동일": 1.0,
    "거의 일치": 0.75,
    "부분 일치": 0.5,
    "약한 관련": 0.25,
    "무관/충돌": 0.0,
}
LABEL_BY_SCORE = {score: label for label, score in SCORE_BY_LABEL.items()}

SAMPLES = 3  # 항목마다 독립 호출 수
RESAMPLE_EXTRA = 2  # 불일치 시 추가로 뽑는 수 (총 5회)
RESAMPLE_SPREAD = 0.5  # max - min 이 이 값 이상이면 재샘플
MAX_CALL_RETRIES = 2  # 응답 검증에 실패한 호출을 다시 부르는 최대 횟수
DEFAULT_K = 1.0  # 분모 보정 합계/(n+k). 검증 세트로 확정하기 전의 시작값
DEFAULT_WORKERS = 6

MISSING_REASON = "상대가 이 항목을 적지 않음"


class ExScoringError(RuntimeError):
    pass


# --------------------------------------------------------------------------- 프롬프트

# 앵커 예시: (원하는 모습, 가진 모습, 라벨). 실제 텍스트를 본 뒤 보강한다.
# 이 목록은 프롬프트에 들어가므로, 바꾸면 COMPARE_PROMPT_VERSION이 바뀌어 캐시 키가 달라진다.
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
        ("진지한 사람 좋아하고 그렇다고 너무 진지한 사람은 말고", "진중하지만 유머도 있는", "거의 일치"),
        ("진지한 사람 좋아하고 그렇다고 너무 진지한 사람은 말고", "항상 엄청 진지하고 무거운", "부분 일치"),
        ("진지한 사람 좋아하고 그렇다고 너무 진지한 사람은 말고", "장난기 많고 가벼운", "무관/충돌"),
    ],
    "vibe_style": [
        ("깔끔한 스타일", "단정한 옷차림", "동일"),
    ],
    "interests": [
        ("영화", "영화 감상, 넷플릭스", "동일"),
        ("영화", "전시회", "약한 관련"),
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

_ANCHOR_LINES = "\n".join(
    f"### {key} ({CRITERION_KO[key]}): {CRITERION_HINT[key]}\n"
    + "\n".join(f"- 원하는 모습 「{want}」 / 가진 모습 「{have}」 → {label}" for want, have, label in ANCHORS[key])
    for key in CRITERIA
)

COMPARE_SYSTEM = f"""
당신은 소개팅 매칭을 위한 비교 분류기다. 한 사람이 상대에게 "원하는 모습"과 상대가 "가진 모습"을 같은 기준끼리 비교해서 일치 정도를 라벨로 고른다. 점수는 매기지 않는다. 숫자는 코드가 정한다.

## 규칙
- 항목마다 독립적으로 판단한다. 다른 항목이나 기준 밖의 내용을 끌어오지 않는다.
- 주어진 두 구절에 적힌 내용만 쓴다. 적혀 있지 않은 성격·외모·습관을 추측하지 않는다.
- 가진 모습이 원하는 모습을 충족하면, 가진 모습이 더 구체적으로 쓰여 있어도 높은 라벨이다. 원하는 모습의 일부만 충족하면 낮춘다.
- 원하는 모습에 정도를 한정하는 말("너무 ~한 사람은 말고", "지나치게 ~하면 부담")이 함께 있으면, 그 한정까지 지켜야 높은 라벨이다. 한정을 어기거나 한정과 정반대이면 낮춘다.
- 군필·직업·학력 같은 사실 정보(background)는 같은 사실이 분명할 때만 높은 라벨을 준다. 서로 다른 사실이면 낮은 라벨이고, 적히지 않은 사실은 추측하지 않는다.
- 두 라벨 사이에서 망설여지면 낮은 쪽을 고른다.
- 각 항목은 reason(한 문장)을 먼저 쓰고, 그 다음에 label을 고른다.
- criterion은 입력에 적힌 키를 그대로 쓰고, 입력의 모든 항목에 한 번씩 답한다.

## 라벨
- 동일: 표현만 다르고 같은 의미
- 거의 일치: 핵심은 같지만 정도나 범위가 조금 다름
- 부분 일치: 같은 방향이지만 구체성이 다름
- 약한 관련: 관련은 있으나 일치라고 하기 어려움
- 무관/충돌: 관련이 없거나 반대

## 앵커 예시
{_ANCHOR_LINES}
""".strip()


class ItemJudgement(BaseModel):
    criterion: Criterion = Field(description="비교하는 기준 키. 입력에 적힌 키를 그대로 쓴다.")
    reason: str = Field(
        description="원하는 모습과 가진 모습만 근거로 왜 그 라벨인지 설명하는 한 문장. 라벨을 고르기 전에 쓴다."
    )
    label: Label = Field(description="라벨 정의와 앵커에 따라 고른 일치 정도. 망설여지면 낮은 쪽.")


class PairJudgement(BaseModel):
    items: list[ItemJudgement]


@dataclass(frozen=True)
class Item:
    criterion: str
    requested: str
    possessed: str


def compare_user_message(items: list[Item]) -> str:
    blocks = []
    for number, item in enumerate(items, start=1):
        blocks.append(
            f"### 항목 {number}: {item.criterion} ({CRITERION_KO[item.criterion]})\n"
            f"원하는 모습: {item.requested}\n"
            f"가진 모습: {item.possessed}"
        )
    return "## 비교할 항목\n\n" + "\n\n".join(blocks)


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


# 프롬프트·앵커·라벨·스키마 중 무엇이든 바뀌면 버전이 바뀌고, 그러면 캐시 키가 달라진다.
COMPARE_PROMPT_VERSION = "cmp-v2-" + _hash(
    COMPARE_SYSTEM,
    compare_user_message([Item("impression", "{requested}", "{possessed}")]),
    json.dumps(PairJudgement.model_json_schema(), sort_keys=True, ensure_ascii=False),
    json.dumps(SCORE_BY_LABEL, sort_keys=True, ensure_ascii=False),
)[:10]


def cache_key(item: Item, model: str, prompt_version: str = COMPARE_PROMPT_VERSION) -> str:
    return _hash(item.criterion, item.requested, item.possessed, prompt_version, model)


# --------------------------------------------------------------------------- LLM 호출

Judgement = dict  # criterion -> (label, reason)


def make_openai_judge(model: str, parse: Callable = parse_chat, max_retries: int = MAX_CALL_RETRIES) -> Callable:
    """items를 한 번에 비교하는 호출. 응답이 항목과 맞지 않으면 max_retries번까지 다시 부른다."""

    def judge(items: list[Item]) -> Judgement:
        expected = {item.criterion for item in items}
        last_error: Optional[Exception] = None
        for _ in range(max_retries + 1):
            try:
                parsed, _raw = parse(
                    model,
                    COMPARE_SYSTEM,
                    [{"role": "user", "content": compare_user_message(items)}],
                    PairJudgement,
                )
                got: Judgement = {}
                for judgement in parsed.items:
                    if judgement.criterion in got:
                        raise ValueError(f"항목이 중복되었습니다: {judgement.criterion}")
                    got[judgement.criterion] = (judgement.label, judgement.reason.strip())
                if set(got) != expected:
                    raise ValueError(f"응답 항목이 입력과 다릅니다: 기대 {sorted(expected)}, 응답 {sorted(got)}")
                return got
            except Exception as err:  # noqa: BLE001 — 다음 시도로 넘어간다
                last_error = err
        raise ExScoringError(f"항목 비교 호출이 실패했습니다: {type(last_error).__name__}: {last_error}") from last_error

    return judge


# --------------------------------------------------------------------------- 합의와 캐시


@dataclass
class ItemResult:
    cache_key: str
    criterion: str
    requested: str
    possessed: str
    label: str
    score: float
    samples: list[dict]
    spread: float
    resampled: bool
    model: str
    prompt_version: str

    @property
    def reason(self) -> str:
        """합의된 점수와 같은 점수를 낸 첫 호출의 이유."""
        for sample in self.samples:
            if sample["score"] == self.score:
                return sample["reason"]
        return ""


def _median(scores: list[float]) -> float:
    ordered = sorted(scores)
    return ordered[len(ordered) // 2]  # 호출 수가 3 또는 5로 홀수이므로 실제 표본 값이다.


def _spread(samples: list[dict]) -> float:
    scores = [sample["score"] for sample in samples]
    return max(scores) - min(scores)


class MemoryItemStore:
    """ex_item_score 캐시의 메모리 구현. DB 구현의 부모이기도 하다."""

    def __init__(self) -> None:
        self._rows: dict[str, ItemResult] = {}

    def get(self, key: str) -> Optional[ItemResult]:
        return self._rows.get(key)

    def put(self, result: ItemResult) -> ItemResult:
        """이미 있으면 먼저 저장된 값을 유지한다. (한 번 매긴 점수는 바뀌지 않는다.)"""
        return self._rows.setdefault(result.cache_key, result)

    def __len__(self) -> int:
        return len(self._rows)


class AuroraItemStore(MemoryItemStore):
    """Aurora ex_item_score. 현재 모델·프롬프트 버전의 행을 메모리에 올려 두고, 새 결과만 쓴다."""

    def __init__(self, conn, model: str, prompt_version: str = COMPARE_PROMPT_VERSION) -> None:
        super().__init__()
        self.conn = conn
        self.model = model
        self.prompt_version = prompt_version

    def load(self) -> int:
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT cache_key, criterion, requested_text, possessed_text, label, score,
                           samples, spread, resampled, model, prompt_version
                    FROM ex_item_score
                    WHERE model = %s AND prompt_version = %s
                    """,
                    (self.model, self.prompt_version),
                )
                fetched = cur.fetchall()
        except pg_errors.UndefinedTable as err:
            self.conn.rollback()
            raise ExScoringError(
                "ex_item_score 테이블이 없습니다. QRious/migrations/aurora/03_ex_item_score.sql을 적용해 주세요."
            ) from err
        self.conn.commit()
        for row in fetched:
            result = ItemResult(
                cache_key=row[0],
                criterion=row[1],
                requested=row[2],
                possessed=row[3],
                label=row[4],
                score=float(row[5]),
                samples=row[6],
                spread=float(row[7]),
                resampled=row[8],
                model=row[9],
                prompt_version=row[10],
            )
            self._rows[result.cache_key] = result
        return len(fetched)

    def put(self, result: ItemResult) -> ItemResult:
        existing = self._rows.get(result.cache_key)
        if existing is not None:
            return existing
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ex_item_score (
                  cache_key, criterion, requested_text, possessed_text, label, score,
                  samples, spread, resampled, model, prompt_version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cache_key) DO NOTHING
                """,
                (
                    result.cache_key,
                    result.criterion,
                    result.requested,
                    result.possessed,
                    result.label,
                    result.score,
                    Jsonb(result.samples),
                    result.spread,
                    result.resampled,
                    result.model,
                    result.prompt_version,
                ),
            )
            inserted = cur.rowcount == 1
            stored = result
            if not inserted:
                # 다른 실행이 먼저 저장했다. 그 값을 따른다.
                cur.execute(
                    """
                    SELECT label, score, samples, spread, resampled
                    FROM ex_item_score WHERE cache_key = %s
                    """,
                    (result.cache_key,),
                )
                label, score, samples, spread, resampled = cur.fetchone()
                stored = ItemResult(
                    result.cache_key,
                    result.criterion,
                    result.requested,
                    result.possessed,
                    label,
                    float(score),
                    samples,
                    float(spread),
                    resampled,
                    result.model,
                    result.prompt_version,
                )
        self.conn.commit()
        self._rows[stored.cache_key] = stored
        return stored


class ItemScorer:
    """항목 비교: 캐시 조회 → 없으면 3회 합의(+재샘플) → 저장."""

    def __init__(
        self,
        store: MemoryItemStore,
        model: str,
        judge: Optional[Callable] = None,
        prompt_version: str = COMPARE_PROMPT_VERSION,
    ) -> None:
        self.store = store
        self.model = model
        self.prompt_version = prompt_version
        self.judge = judge or make_openai_judge(model)

    def key(self, item: Item) -> str:
        return cache_key(item, self.model, self.prompt_version)

    def lookup(self, item: Item) -> Optional[ItemResult]:
        return self.store.get(self.key(item))

    def compute(self, items: list[Item]) -> list[ItemResult]:
        """한 쌍의 항목들을 합의로 채점한다. 저장하지 않는다. (스레드에서 불러도 안전하다.)"""
        by_criterion = {item.criterion: item for item in items}
        samples: dict[str, list[dict]] = {item.criterion: [] for item in items}

        def draw(subset: list[Item], times: int) -> None:
            for _ in range(times):
                judgement = self.judge(subset)
                for item in subset:
                    label, reason = judgement[item.criterion]
                    samples[item.criterion].append({"label": label, "score": SCORE_BY_LABEL[label], "reason": reason})

        draw(items, SAMPLES)
        disputed = [item for item in items if _spread(samples[item.criterion]) >= RESAMPLE_SPREAD]
        if disputed:
            draw(disputed, RESAMPLE_EXTRA)
        resampled = {item.criterion for item in disputed}

        results = []
        for criterion, item in by_criterion.items():
            drawn = samples[criterion]
            score = _median([sample["score"] for sample in drawn])
            spread = _spread(drawn)
            if spread >= RESAMPLE_SPREAD:
                print(
                    f"경고: 항목 합의가 불안정합니다 ({criterion}, 호출 {len(drawn)}회, spread {spread:.2f}): "
                    f"{item.requested[:30]!r} / {item.possessed[:30]!r}",
                    file=sys.stderr,
                )
            results.append(
                ItemResult(
                    cache_key=self.key(item),
                    criterion=criterion,
                    requested=item.requested,
                    possessed=item.possessed,
                    label=LABEL_BY_SCORE[score],
                    score=score,
                    samples=drawn,
                    spread=spread,
                    resampled=criterion in resampled,
                    model=self.model,
                    prompt_version=self.prompt_version,
                )
            )
        return results

    def score_items(self, items: list[Item]) -> dict[Item, ItemResult]:
        """캐시에 있으면 그대로, 없는 항목만 합의로 채점해서 저장한다."""
        found: dict[Item, ItemResult] = {}
        missing: list[Item] = []
        for item in items:
            cached = self.lookup(item)
            if cached is not None:
                found[item] = cached
            else:
                missing.append(item)
        if missing:
            for result in self.compute(missing):
                stored = self.store.put(result)
                found[Item(result.criterion, result.requested, result.possessed)] = stored
        return found

    def pending_units(self, directed: Iterable[tuple[Optional[dict], Optional[dict]]]) -> list[list[Item]]:
        """아직 점수가 없는 항목을 방향 쌍(요구자의 want, 상대의 have)별 작업 단위로 묶는다.

        같은 항목이 여러 쌍에 나오면 처음 나온 쌍에서만 채점하므로 같은 항목을 두 번 부르지 않는다.
        """
        seen: set[str] = set()
        units: list[list[Item]] = []
        for requester, possessor in directed:
            unit = []
            for item in llm_items(requester, possessor):
                key = self.key(item)
                if key in seen or self.store.get(key) is not None:
                    continue
                seen.add(key)
                unit.append(item)
            if unit:
                units.append(unit)
        return units

    def prewarm(
        self,
        directed: Iterable[tuple[Optional[dict], Optional[dict]]],
        workers: int = DEFAULT_WORKERS,
        progress: Callable[[str], None] = print,
    ) -> int:
        """채점해야 할 항목을 병렬로 미리 채점해 저장한다. 이후 매칭 실행은 저장된 값만 읽는다."""
        units = self.pending_units(directed)
        if not units:
            progress(f"항목 점수: 모두 캐시에 있음 ({len(self.store)}건)")
            return 0
        total_items = sum(len(unit) for unit in units)
        progress(f"항목 점수: 새로 채점할 {total_items}개 ({len(units)}개 호출 묶음, 동시 {workers}개)")

        failures: list[str] = []
        saved = 0
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(self.compute, unit): unit for unit in units}
            for number, future in enumerate(as_completed(futures), start=1):
                try:
                    results = future.result()
                except Exception as err:  # noqa: BLE001 — 나머지는 계속 채점하고 마지막에 한꺼번에 알린다.
                    failures.append(f"{type(err).__name__}: {err}")
                    continue
                for result in results:
                    self.store.put(result)
                saved += len(results)
                progress(f"[{number}/{len(units)}] 항목 {len(results)}개 저장 (누적 {saved}/{total_items})")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if failures:
            raise ExScoringError(
                f"항목 비교 {len(failures)}건이 실패했습니다. 저장된 항목은 유지되므로 다시 실행하면 이어서 합니다.\n"
                + "\n".join(failures[:5])
            )
        return saved


def llm_items(requester: Optional[dict], possessor: Optional[dict]) -> list[Item]:
    """요구자가 요구한 기준 중 상대도 적은 항목. (이 항목만 LLM이 비교한다.)"""
    if not requester:
        return []
    items = []
    for criterion in CRITERIA:
        requested = requester.get(criterion)
        possessed = possessor.get(criterion) if possessor else None
        if requested and possessed:
            items.append(Item(criterion, requested, possessed))
    return items


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
    criteria: list[dict]

    def payload(self) -> dict:
        return {
            "defined": self.defined,
            "score": None if self.score is None else round(self.score, 6),
            "n": self.n,
            "k": self.k,
            "criteria": self.criteria,
        }


def direction_score(
    requester: Optional[dict], possessor: Optional[dict], scorer: ItemScorer, k: float
) -> DirectionResult:
    """dir(A→B) = Σ(A가 요구한 항목의 점수) / (n + k).

    정의되는 방향: A가 요구한 항목이 1개 이상. (B가 모두 안 적어 0점이어도 정의된 점수 0이다.)
    정의되지 않는 방향: A가 ex_want를 안 썼거나 추출 결과 요구 항목이 0개.
    """
    requested = [criterion for criterion in CRITERIA if requester and requester.get(criterion)]
    if not requested:
        return DirectionResult(defined=False, n=0, k=k, score=None, criteria=[])

    llm = {item.criterion: item for item in llm_items(requester, possessor)}
    results = scorer.score_items(list(llm.values())) if llm else {}

    criteria = []
    total = 0.0
    for criterion in requested:
        want = requester[criterion]
        item = llm.get(criterion)
        if item is None:
            criteria.append(
                {
                    "criterion": criterion,
                    "requested": want,
                    "possessed": None,
                    "label": None,
                    "score": 0.0,
                    "source": "MISSING",
                    "reason": MISSING_REASON,
                }
            )
            continue
        result = results[item]
        total += result.score
        criteria.append(
            {
                "criterion": criterion,
                "requested": want,
                "possessed": item.possessed,
                "label": result.label,
                "score": result.score,
                "source": "LLM",
                "reason": result.reason,
                "spread": result.spread,
                "resampled": result.resampled,
            }
        )
    n = len(requested)
    return DirectionResult(defined=True, n=n, k=k, score=total / (n + k), criteria=criteria)


@dataclass
class ExResult:
    score: float  # weight_mode == "none"이면 0.0 (최종 점수 계산에서 쓰이지 않는다)
    weight_mode: str  # both | one_direction | none
    detail: dict


def ex_score(
    male_want: Optional[dict],
    male_have: Optional[dict],
    female_want: Optional[dict],
    female_have: Optional[dict],
    scorer: ItemScorer,
    k: Optional[float] = None,
) -> ExResult:
    """각 인자는 추출된 8개 기준 dict. 원문이 없거나 추출이 유효하지 않으면 None."""
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
