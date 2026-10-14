"""ex 점수 단위 테스트. OpenAI·Aurora를 부르지 않는다.

  python -m unittest discover -s tests -v
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from typing import get_args

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as match_main  # noqa: E402
from score_functions import ex as ex_mod  # noqa: E402
from score_functions import ex_traits as traits_mod  # noqa: E402
from score_functions.ex import (  # noqa: E402
    EMPTY_HAVE,
    EXAMPLE_ANSWER,
    EXAMPLE_HAVE,
    EXAMPLE_WANT,
    MATCH_LABELS,
    NO_INFO,
    NOT_REQUESTED,
    PAIR_PROMPT_VERSION,
    PAIR_SYSTEM,
    CriterionJudgement,
    ExScoringError,
    MemoryPairStore,
    PairEvaluation,
    PairScorer,
    consensus,
    direction_score,
    ex_score,
    final_score,
    find_violations,
    make_openai_judge,
    pair_cache_key,
    user_message,
)
from score_functions.tag import tag_inclusion_min  # noqa: E402

MODEL = "test-model-2026-01-01"
CRITERIA = traits_mod.CRITERIA

# 긍정 요구와 부정 표현이 한 문장에 섞인 실제 원문.
MIXED_WANT = (
    "진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고 ㅎ\n"
    "외적인 이상형은 송하영 느낌 ㅎ\n"
    "상대가 좋다면 어떤 취미라도 즐깁니다.\n"
    "독서빼고요. 근데 저는 너무 압도적으로 예쁜 사람은 상대하기 어려워요.. 부담감 생겨버림 ㅠ"
)


def traits(**values):
    return {key: values.get(key) for key in CRITERIA}


def val(statement, *evidence):
    """추출 LLM이 돌려주는 기준 하나: 재작성한 구절(statement)과 원문 근거(evidence)."""
    return {"statement": statement, "evidence": list(evidence)}


def crit(label=NOT_REQUESTED, want=(), have=(), reason="이유"):
    """평가 LLM이 돌려주는 기준 하나."""
    return {"want_quote": list(want), "have_quote": list(have), "reason": reason, "label": label}


def evaluation(**criteria):
    """평가 응답 전체. 적지 않은 기준은 요구 없음."""
    return {key: criteria.get(key, crit()) for key in CRITERIA}


def judged(label: str, key: str = "lifestyle", reason: str = "") -> dict:
    """검증을 통과해 정리된 기준 하나 (judge가 돌려주는 형태)."""
    return {
        "label": label,
        "want_quote": [] if label == NOT_REQUESTED else [f"{key} 요구"],
        "have_quote": [f"{key} 매력"] if label in MATCH_LABELS else [],
        "reason": reason or f"{key} {label}",
    }


def sample(**labels):
    """호출 한 번의 정리된 평가. 적지 않은 기준은 요구 없음."""
    return {key: judged(labels.get(key, NOT_REQUESTED), key) for key in CRITERIA}


class ScriptedJudge:
    """기준별로 호출 순서대로 라벨을 돌려주는 가짜 LLM. 적지 않은 기준은 요구 없음. 호출 기록을 남긴다."""

    def __init__(self, script: dict[str, list[str]]):
        self.script = {key: list(labels) for key, labels in script.items()}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, want, have):
        self.calls.append((want, have))
        return sample(**{key: labels.pop(0) for key, labels in self.script.items()})


def scorer_with(script: dict[str, list[str]], store=None):
    judge = ScriptedJudge(script)
    return PairScorer(MemoryPairStore() if store is None else store, MODEL, judge=judge), judge


def quiet(_message: str) -> None:
    pass


# --------------------------------------------------------------------------- 원문 평가 (매칭에 쓰임)


class PairValidationTests(unittest.TestCase):
    """인용은 각 원문의 글자 그대로여야 하고, 라벨은 인용과 맞아야 한다."""

    WANT = "잘 웃고, 배려심 있는 사람이면 좋겠어요! 담배 안 피는 사람이 좋아요. 키 큰 사람이면 좋겠어요."
    HAVE = "배려심 깊다는 말 자주 들어요. 비흡연자고 운동 좋아해요."

    def check(self, **criteria):
        return find_violations(evaluation(**criteria), self.WANT, self.HAVE)

    def test_quotes_ignore_spacing_and_punctuation(self):
        found = self.check(
            personality=crit("부분 일치", ["잘 웃고 배려심 있는 사람이면 좋겠어요"], ["배려심 깊다는 말 자주 들어요"]),
            lifestyle=crit("동일", ["담배 안 피는 사람이 좋아요"], ["비흡연자고"]),
            appearance=crit(NO_INFO, ["키 큰 사람이면 좋겠어요"]),
        )
        self.assertEqual(found, [])

    def test_want_quote_must_be_copied_from_the_want_text(self):
        found = self.check(personality=crit("거의 일치", ["다정한 사람"], ["배려심 깊다는 말 자주 들어요"]))
        self.assertEqual(len(found), 1)
        self.assertIn("personality: want_quote가 이상형 원문에 없는 표현", found[0])

    def test_have_quote_must_be_copied_from_the_have_text(self):
        # 이상형 원문에 있는 구절이라도 매력 원문에 없으면 근거가 아니다.
        found = self.check(lifestyle=crit("동일", ["담배 안 피는 사람이 좋아요"], ["담배 안 피는 사람"]))
        self.assertEqual(len(found), 1)
        self.assertIn("lifestyle: have_quote가 매력 원문에 없는 표현", found[0])

    def test_each_quote_piece_is_checked_one_by_one(self):
        ok = self.check(lifestyle=crit("거의 일치", ["담배 안 피는 사람이 좋아요"], ["비흡연자고", "운동 좋아해요"]))
        self.assertEqual(ok, [])
        bad = self.check(lifestyle=crit("거의 일치", ["담배 안 피는 사람이 좋아요"], ["비흡연자고", "술도 안 마셔요"]))
        self.assertEqual(len(bad), 1)

    def test_quote_without_letters_is_a_violation(self):
        found = self.check(lifestyle=crit(NO_INFO, ["🚭"]))
        self.assertEqual(len(found), 1)
        self.assertIn("의미 있는 글자가 없다", found[0])

    def test_label_must_agree_with_the_quotes(self):
        cases = {
            "요구 없음인데 요구 인용": crit(NOT_REQUESTED, ["담배 안 피는 사람이 좋아요"]),
            "요구 인용 없이 라벨": crit("동일", [], ["비흡연자고"]),
            "정보 없음인데 매력 인용": crit(NO_INFO, ["담배 안 피는 사람이 좋아요"], ["비흡연자고"]),
            "매력 인용 없이 5단계 라벨": crit("동일", ["담배 안 피는 사람이 좋아요"]),
        }
        for title, item in cases.items():
            with self.subTest(title):
                found = self.check(lifestyle=item)
                self.assertEqual(len(found), 1, found)
                self.assertTrue(found[0].startswith("lifestyle:"), found[0])

    def test_one_want_phrase_belongs_to_one_criterion(self):
        found = self.check(
            impression=crit(NO_INFO, ["잘 웃고 배려심 있는 사람이면 좋겠어요"]),
            personality=crit("부분 일치", ["잘 웃고, 배려심 있는 사람이면 좋겠어요"], ["배려심 깊다는 말 자주 들어요"]),
        )
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].startswith("personality:"), found[0])
        self.assertIn("impression에도 넣었다", found[0])

    def test_one_have_phrase_may_support_several_criteria(self):
        found = find_violations(
            evaluation(
                appearance=crit("약한 관련", ["몸이 탄탄했으면 좋겠어요"], ["헬스가 취미예요"]),
                lifestyle=crit("거의 일치", ["운동하는 사람이 좋고"], ["헬스가 취미예요"]),
            ),
            "운동하는 사람이 좋고 몸이 탄탄했으면 좋겠어요",
            "헬스가 취미예요",
        )
        self.assertEqual(found, [])

    def test_empty_have_text_only_allows_no_info(self):
        ok = find_violations(evaluation(lifestyle=crit(NO_INFO, ["담배 안 피는 사람이 좋아요"])), self.WANT, "")
        self.assertEqual(ok, [])
        bad = find_violations(evaluation(lifestyle=crit("동일", ["담배 안 피는 사람이 좋아요"], ["비흡연자"])), self.WANT, "")
        self.assertEqual(len(bad), 1)
        self.assertIn("매력 원문에 없는 표현", bad[0])

    def test_nothing_requested_is_valid(self):
        self.assertEqual(self.check(), [])

    def test_real_mixed_text_can_be_quoted_without_dropping_anything(self):
        """추출할 때 일부가 빠지던 실제 원문을, 기준별로 원문 그대로 인용한 답이 검증을 통과한다."""
        have = "진중한 편인데 유머도 있어요. 영화 감상이랑 전시회 다니는 거 좋아해요."
        response = evaluation(
            appearance=crit(NO_INFO, ["외적인 이상형은 송하영 느낌", "너무 압도적으로 예쁜 사람은 상대하기 어려워요"]),
            personality=crit(
                "거의 일치", ["진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고"], ["진중한 편인데 유머도 있어요"]
            ),
            interests=crit(
                "거의 일치",
                ["상대가 좋다면 어떤 취미라도 즐깁니다", "독서빼고요"],
                ["영화 감상이랑 전시회 다니는 거 좋아해요"],
            ),
        )
        self.assertEqual(find_violations(response, MIXED_WANT, have), [])

    def test_clean_evaluation_trims_quotes_and_drops_blank_pieces(self):
        cleaned = ex_mod.clean_evaluation(
            evaluation(lifestyle=crit("동일", [" 담배 안 피는 사람이 좋아요 ", "  "], ["비흡연자고"], " 같은 뜻 "))
        )
        self.assertEqual(
            cleaned["lifestyle"],
            {"label": "동일", "want_quote": ["담배 안 피는 사람이 좋아요"], "have_quote": ["비흡연자고"], "reason": "같은 뜻"},
        )
        self.assertEqual((cleaned["personality"]["label"], cleaned["personality"]["want_quote"]), (NOT_REQUESTED, []))


class JudgeTests(unittest.TestCase):
    """두 원문을 한 번 평가하는 호출: 검증에 실패하면 문제를 알려 주고 다시 묻는다."""

    GOOD = evaluation(lifestyle=crit("동일", ["담배 안 피는 사람이 좋고"], ["비흡연자고"]))
    BAD = evaluation(lifestyle=crit("동일", ["담배 싫어하는 사람"], ["비흡연자고"]))  # 이상형 원문에 없는 구절

    def fake_parse(self, responses):
        queue = list(responses)
        seen = []

        def parse(model, system, messages, response_format):
            seen.append((system, messages[0]["content"]))
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return response_format.model_validate(item), "{}"

        parse.seen = seen
        return parse

    def test_valid_response_passes_on_first_attempt(self):
        parse = self.fake_parse([self.GOOD])
        result = make_openai_judge(MODEL, parse=parse)(EXAMPLE_WANT, EXAMPLE_HAVE)
        self.assertEqual(result["lifestyle"]["label"], "동일")
        self.assertEqual(result["lifestyle"]["have_quote"], ["비흡연자고"])
        self.assertEqual(result["personality"]["label"], NOT_REQUESTED)
        self.assertEqual(len(parse.seen), 1)
        system, message = parse.seen[0]
        self.assertEqual(system, PAIR_SYSTEM)
        self.assertIn(EXAMPLE_WANT, message)
        self.assertIn(EXAMPLE_HAVE, message)
        self.assertNotIn("이전 답의 문제", message)

    def test_retry_feeds_back_the_violation_and_then_succeeds(self):
        parse = self.fake_parse([self.BAD, self.GOOD])
        result = make_openai_judge(MODEL, parse=parse)(EXAMPLE_WANT, EXAMPLE_HAVE)
        self.assertEqual(result["lifestyle"]["want_quote"], ["담배 안 피는 사람이 좋고"])
        self.assertEqual(len(parse.seen), 2)
        self.assertIn("이전 답의 문제", parse.seen[1][1])
        self.assertIn("이상형 원문에 없는 표현", parse.seen[1][1])

    def test_api_error_counts_as_an_attempt_and_can_recover(self):
        parse = self.fake_parse([RuntimeError("boom"), self.GOOD])
        result = make_openai_judge(MODEL, parse=parse)(EXAMPLE_WANT, EXAMPLE_HAVE)
        self.assertEqual(result["lifestyle"]["label"], "동일")
        self.assertNotIn("이전 답의 문제", parse.seen[1][1])

    def test_gives_up_after_the_retries(self):
        parse = self.fake_parse([self.BAD] * 3)
        with self.assertRaises(ExScoringError) as caught:
            make_openai_judge(MODEL, parse=parse)(EXAMPLE_WANT, EXAMPLE_HAVE)
        self.assertEqual(len(parse.seen), 3)
        self.assertIn("시도 3", str(caught.exception))

    def test_empty_have_text_is_shown_as_not_written(self):
        parse = self.fake_parse([evaluation(lifestyle=crit(NO_INFO, ["담배 안 피는 사람이 좋고"]))])
        result = make_openai_judge(MODEL, parse=parse)(EXAMPLE_WANT, "")
        self.assertEqual(result["lifestyle"]["label"], NO_INFO)
        self.assertIn(EMPTY_HAVE, parse.seen[0][1])


class ConsensusTests(unittest.TestCase):
    WANT, HAVE = "담배 안 피는 사람", "비흡연자"

    def evaluate(self, script):
        scorer, judge = scorer_with(script)
        return scorer.evaluate(self.WANT, self.HAVE), judge

    def test_three_agreeing_calls_do_not_resample(self):
        result, judge = self.evaluate({"lifestyle": ["동일"] * 3})
        item = result.criteria["lifestyle"]
        self.assertEqual((item["label"], item["score"], item["votes"], item["spread"]), ("동일", 1.0, 3, 0.0))
        self.assertFalse(result.resampled)
        self.assertEqual((result.calls, len(judge.calls)), (3, 3))

    def test_small_disagreement_uses_the_median_without_resampling(self):
        result, judge = self.evaluate({"lifestyle": ["동일", "거의 일치", "거의 일치"]})
        item = result.criteria["lifestyle"]
        self.assertEqual((item["label"], item["score"]), ("거의 일치", 0.75))
        self.assertFalse(result.resampled)
        self.assertEqual(len(judge.calls), 3)

    def test_wide_disagreement_resamples_to_five_and_takes_the_median(self):
        result, judge = self.evaluate({"lifestyle": ["동일", "무관/충돌", "동일", "동일", "거의 일치"]})
        item = result.criteria["lifestyle"]
        self.assertTrue(result.resampled)
        self.assertEqual((result.calls, len(judge.calls), len(result.samples)), (5, 5, 5))
        self.assertEqual((item["label"], item["score"], item["spread"]), ("동일", 1.0, 1.0))

    def test_split_on_whether_it_was_requested_resamples(self):
        result, _ = self.evaluate({"lifestyle": ["동일", NOT_REQUESTED, "동일", "동일", "동일"]})
        item = result.criteria["lifestyle"]
        self.assertTrue(result.resampled)
        self.assertEqual((item["label"], item["score"], item["votes"]), ("동일", 1.0, 4))

    def test_requested_by_a_minority_is_not_requested(self):
        result, _ = self.evaluate({"lifestyle": [NOT_REQUESTED, "동일", NOT_REQUESTED, NOT_REQUESTED, NOT_REQUESTED]})
        item = result.criteria["lifestyle"]
        self.assertTrue(result.resampled)
        self.assertEqual((item["label"], item["score"], item["votes"], item["want_quote"]), (NOT_REQUESTED, None, 1, []))

    def test_resampling_redraws_every_criterion(self):
        result, _ = self.evaluate(
            {"lifestyle": ["동일"] * 5, "interests": ["약한 관련", "동일", "약한 관련", "약한 관련", "약한 관련"]}
        )
        self.assertTrue(result.resampled)
        self.assertEqual(result.criteria["lifestyle"]["votes"], 5)
        self.assertEqual(result.criteria["interests"]["label"], "약한 관련")

    def test_zero_score_keeps_the_most_common_zero_label(self):
        # 정보 없음과 무관/충돌은 둘 다 0점이라, 라벨은 더 많이 나온 쪽을 쓴다.
        result, _ = self.evaluate({"lifestyle": [NO_INFO, "무관/충돌", NO_INFO]})
        item = result.criteria["lifestyle"]
        self.assertEqual((item["label"], item["score"]), (NO_INFO, 0.0))
        self.assertFalse(result.resampled)

    def test_even_number_of_requested_calls_takes_the_lower_middle(self):
        samples = [
            sample(lifestyle="동일"),
            sample(lifestyle="약한 관련"),
            sample(lifestyle="거의 일치"),
            sample(lifestyle="부분 일치"),
            sample(),
        ]
        criteria, _ = consensus(samples)
        item = criteria["lifestyle"]
        self.assertEqual((item["label"], item["score"], item["votes"]), ("부분 일치", 0.5, 4))

    def test_quotes_and_reason_come_from_a_call_with_the_chosen_label(self):
        samples = [
            {**sample(), "lifestyle": judged("동일", reason="첫째")},
            {**sample(), "lifestyle": judged("거의 일치", reason="둘째")},
            {**sample(), "lifestyle": judged("거의 일치", reason="셋째")},
        ]
        criteria, disputed = consensus(samples)
        self.assertEqual(criteria["lifestyle"]["reason"], "둘째")
        self.assertEqual(disputed, [])


class PairCacheTests(unittest.TestCase):
    def test_cached_pair_is_not_asked_again_and_keeps_its_first_result(self):
        store = MemoryPairStore()
        first_scorer, _ = scorer_with({"lifestyle": ["동일"] * 3}, store)
        first = first_scorer.evaluate("담배 안 피는 사람", "비흡연자")
        again, judge = scorer_with({"lifestyle": ["무관/충돌"] * 3}, store)
        second = again.evaluate("담배 안 피는 사람", "비흡연자")
        self.assertEqual(judge.calls, [])
        self.assertEqual(second.criteria, first.criteria)

    def test_cache_key_depends_on_every_input(self):
        base = pair_cache_key("담배 안 피는 사람", "비흡연자", MODEL)
        self.assertEqual(base, pair_cache_key("담배 안 피는 사람", "비흡연자", MODEL, PAIR_PROMPT_VERSION))
        self.assertNotEqual(base, pair_cache_key("담배 안 피는 사람!", "비흡연자", MODEL))
        self.assertNotEqual(base, pair_cache_key("담배 안 피는 사람", "금연 중", MODEL))
        self.assertNotEqual(base, pair_cache_key("비흡연자", "담배 안 피는 사람", MODEL))
        self.assertNotEqual(base, pair_cache_key("담배 안 피는 사람", "비흡연자", "other-model-2026-02-02"))
        self.assertNotEqual(base, pair_cache_key("담배 안 피는 사람", "비흡연자", MODEL, prompt_version="pair-v1-other"))

    def test_pending_skips_empty_wants_and_asks_each_text_pair_once(self):
        scorer, _ = scorer_with({})
        pending = scorer.pending(
            [
                ("영화 좋아하는 사람", "영화 좋아해요"),
                (" 영화 좋아하는 사람 ", "영화 좋아해요\n"),
                ("", "운동해요"),
                (None, "운동해요"),
                ("   ", "운동해요"),
                ("운동하는 사람", None),
                ("운동하는 사람", ""),
            ]
        )
        self.assertEqual(pending, [("영화 좋아하는 사람", "영화 좋아해요"), ("운동하는 사람", "")])

    def test_prewarm_saves_everything_so_pair_scoring_never_calls_the_llm(self):
        scorer, judge = scorer_with({"lifestyle": ["동일"] * 6})
        directed = [("아침형인 사람", "일찍 일어나요"), ("담배 안 피는 사람", None)]
        self.assertEqual(scorer.prewarm(directed, workers=2, progress=quiet), 2)
        self.assertEqual(len(judge.calls), 6)
        result = ex_score("아침형인 사람", None, "담배 안 피는 사람", "일찍 일어나요", scorer, k=1.0)
        self.assertEqual(len(judge.calls), 6)
        self.assertEqual(result.weight_mode, "both")
        self.assertEqual(scorer.prewarm(directed, progress=quiet), 0)

    def test_failed_pairs_raise_but_finished_pairs_stay_saved(self):
        def judge(want, have):
            if want == "실패할 요구":
                raise ExScoringError("호출 실패")
            return sample(lifestyle="동일")

        scorer = PairScorer(MemoryPairStore(), MODEL, judge=judge)
        directed = [("실패할 요구", "매력"), ("성공할 요구", "매력")]
        with self.assertRaises(ExScoringError):
            scorer.prewarm(directed, workers=2, progress=quiet)
        self.assertEqual(scorer.pending(directed), [("실패할 요구", "매력")])


class DirectionTests(unittest.TestCase):
    def test_undefined_without_a_want_text_and_no_llm_call(self):
        scorer, judge = scorer_with({})
        for want in (None, "", "   "):
            with self.subTest(want=want):
                result = direction_score(want, "아침형이에요", scorer, k=1.0)
                self.assertFalse(result.defined)
                self.assertIsNone(result.score)
                self.assertIsNone(result.payload()["score"])
        self.assertEqual(judge.calls, [])

    def test_undefined_when_the_text_requests_nothing(self):
        scorer, judge = scorer_with({})
        result = direction_score("저는 낯을 좀 가려요", "아침형이에요", scorer, k=1.0)
        self.assertFalse(result.defined)
        self.assertEqual((result.n, result.criteria, result.calls), (0, [], 3))
        self.assertEqual(len(judge.calls), 3)

    def test_score_is_sum_over_n_plus_k_and_no_info_counts_as_zero(self):
        scorer, _ = scorer_with({"lifestyle": ["동일"] * 3, "interests": ["부분 일치"] * 3, "personality": [NO_INFO] * 3})
        result = direction_score("아침형, 영화, 다정한 사람", "일찍 일어나고 영화 좋아해요", scorer, k=1.0)
        self.assertTrue(result.defined)
        self.assertEqual(result.n, 3)
        self.assertAlmostEqual(result.score, (1.0 + 0.5 + 0.0) / (3 + 1.0))
        self.assertEqual([row["criterion"] for row in result.criteria], ["personality", "interests", "lifestyle"])
        by_criterion = {row["criterion"]: row for row in result.criteria}
        self.assertEqual((by_criterion["personality"]["label"], by_criterion["personality"]["score"]), (NO_INFO, 0.0))
        self.assertEqual(by_criterion["lifestyle"]["want_quote"], ["lifestyle 요구"])

    def test_partner_who_wrote_nothing_gives_a_defined_zero(self):
        scorer, judge = scorer_with({"lifestyle": [NO_INFO] * 3})
        result = direction_score("담배 안 피는 사람", None, scorer, k=1.0)
        self.assertTrue(result.defined)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(judge.calls[0], ("담배 안 피는 사람", ""))
        # 빈 매력 원문은 None이든 공백이든 같은 캐시를 쓴다.
        direction_score("담배 안 피는 사람", "  ", scorer, k=1.0)
        self.assertEqual(len(judge.calls), 3)

    def test_k_changes_the_denominator(self):
        scorer, _ = scorer_with({"lifestyle": ["동일"] * 3})
        self.assertEqual(direction_score("아침형인 사람", "일찍 일어나요", scorer, k=0.0).score, 1.0)
        self.assertEqual(direction_score("아침형인 사람", "일찍 일어나요", scorer, k=1.0).score, 0.5)

    def test_payload_keeps_quotes_votes_and_reason(self):
        scorer, _ = scorer_with({"lifestyle": ["동일"] * 3})
        payload = direction_score("아침형인 사람", "일찍 일어나요", scorer, k=1.0).payload()
        self.assertEqual((payload["calls"], payload["resampled"]), (3, False))
        self.assertEqual(
            set(payload["criteria"][0]),
            {"criterion", "label", "score", "want_quote", "have_quote", "reason", "votes", "spread"},
        )
        json.dumps(payload, ensure_ascii=False)  # ex_detail로 저장할 수 있어야 한다.


class ExScoreModeTests(unittest.TestCase):
    def setUp(self):
        self.scorer, _ = scorer_with({"lifestyle": ["동일"] * 12})

    def test_both_directions_average(self):
        result = ex_score("아침형인 사람", "일찍 일어나요", "담배 안 피는 사람", "비흡연자예요", self.scorer, k=1.0)
        self.assertEqual(result.weight_mode, "both")
        self.assertAlmostEqual(result.score, (0.5 + 0.5) / 2)
        self.assertEqual(result.detail["weight_mode"], "both")

    def test_one_direction_is_halved(self):
        result = ex_score("아침형인 사람", None, None, "일찍 일어나요", self.scorer, k=1.0)
        self.assertEqual(result.weight_mode, "one_direction")
        self.assertAlmostEqual(result.score, 0.5 / 2)
        self.assertTrue(result.detail["male_to_female"]["defined"])
        self.assertFalse(result.detail["female_to_male"]["defined"])

    def test_no_direction_drops_the_ex_term(self):
        result = ex_score(None, "일찍 일어나요", "", None, self.scorer, k=1.0)
        self.assertEqual(result.weight_mode, "none")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.detail["weight_mode"], "none")

    def test_requester_who_wrote_but_partner_did_not_is_zero_not_dropped(self):
        scorer, _ = scorer_with({"lifestyle": [NO_INFO] * 3})
        result = ex_score("아침형인 사람", None, None, None, scorer, k=1.0)
        self.assertEqual(result.weight_mode, "one_direction")
        self.assertEqual(result.score, 0.0)

    def test_detail_records_how_it_was_scored(self):
        detail = ex_score("아침형인 사람", None, None, "일찍 일어나요", self.scorer, k=1.0).detail
        self.assertEqual(detail["version"], ex_mod.EX_DETAIL_VERSION)
        self.assertEqual((detail["model"], detail["prompt_version"], detail["k"]), (MODEL, PAIR_PROMPT_VERSION, 1.0))


class PairPromptTests(unittest.TestCase):
    """평가 프롬프트가 원문 전체를 읽는 규칙과 기준 정의를 담고 있는지 지킨다. (LLM은 부르지 않는다.)"""

    def test_prompt_example_is_valid_by_our_own_rules(self):
        """프롬프트에 실은 예시가 규칙을 어기면 모델이 그걸 따라 한다. 예시 데이터를 같은 검증으로 확인한다."""
        response = evaluation(
            **{key: crit(label, want, have, reason) for key, (want, have, reason, label) in EXAMPLE_ANSWER.items()}
        )
        self.assertEqual(find_violations(response, EXAMPLE_WANT, EXAMPLE_HAVE), [])
        PairEvaluation.model_validate(response)
        self.assertIn(EXAMPLE_WANT, PAIR_SYSTEM)
        self.assertIn(EXAMPLE_HAVE, PAIR_SYSTEM)
        for want, have, reason, label in EXAMPLE_ANSWER.values():
            for piece in want + have:
                self.assertIn(f'"{piece}"', PAIR_SYSTEM)
            self.assertIn(f'reason "{reason}" → {label}', PAIR_SYSTEM)
        self.assertIn(NO_INFO, {label for _w, _h, _r, label in EXAMPLE_ANSWER.values()})
        self.assertIn('"저는 낯을 좀 가려요"는 본인 소개라 요구가 아니고', PAIR_SYSTEM)

    def test_every_criterion_has_its_definition_and_anchors_in_the_prompt(self):
        self.assertEqual(set(ex_mod.ANCHORS), set(CRITERIA))
        for key in CRITERIA:
            with self.subTest(criterion=key):
                header = f"### {key} ({traits_mod.CRITERION_KO[key]}): {traits_mod.CRITERION_HINT[key]}"
                self.assertIn(header, PAIR_SYSTEM)
                for want, have, label in ex_mod.ANCHORS[key]:
                    self.assertIn(label, MATCH_LABELS)
                    self.assertIn(f"- 「{want}」 / 「{have}」 → {label}", PAIR_SYSTEM)

    def test_values_go_to_personality_and_marriage_is_never_a_request(self):
        self.assertIn("### personality (성격/가치관)", PAIR_SYSTEM)
        self.assertIn("### relationship_values (연애관)", PAIR_SYSTEM)
        self.assertIn("가치관(인생관·가족관 등)과 대화·소통 방식(말이 잘 통함, 티키타카, 공감)은 personality다", PAIR_SYSTEM)
        self.assertIn("relationship_values는 연락 빈도·애정표현·스킨십 같은 연애관만이다", PAIR_SYSTEM)
        mentions = [line for line in PAIR_SYSTEM.splitlines() if "결혼" in line]
        self.assertEqual(mentions, ["- 결혼관(결혼에 대한 생각·결혼 계획)은 어느 기준의 요구로도 보지 않는다."])

    def test_whole_text_reading_rules(self):
        system = PAIR_SYSTEM
        self.assertIn("글을 끝까지 구절 단위로 훑어서 요구를 하나도 빠뜨리지 않는다", system)
        self.assertIn("글쓴이가 본인을 소개하는 말", system)
        self.assertIn('"저는 ~한 사람이 좋아요", "저는 ~한 사람은 어려워요"는 상대의 모습을 말하므로 요구다', system)
        self.assertIn('거절은 "~하지 않는" 요구로 읽는다', system)
        self.assertIn('"진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고" → 적당히 진지한 사람', system)
        self.assertIn("그 기준에 조건이 없다는 뜻이다", system)
        self.assertIn('함께 적은 제외 대상("독서빼고요")만 조건이다', system)
        for slang in ('"진짜"', '"완전"', '"개"', '"ㄹㅇ"', '"x100000"', '"다정다정"'):
            self.assertIn(slang, system)
        for phrase in ('"~맞죠?"', '"~잘 모르겠어요"', '"~일까요?"'):
            self.assertIn(phrase, system)
        self.assertIn('"~같아요"는 허용한다', system)
        self.assertIn("매력 원문에서 상대에게 바라는 말은 근거가 아니다", system)

    def test_quote_rules(self):
        system = PAIR_SYSTEM
        self.assertIn("원문에 적힌 글자를 그대로 복사한다", system)
        self.assertIn("같은 구절을 두 기준의 want_quote에 넣지 않는다", system)
        self.assertIn("같은 구절을 여러 기준의 근거로 써도 된다", system)
        self.assertIn(f'매력 원문이 "{EMPTY_HAVE}"이면 요구가 있는 기준은 모두 정보 없음이다', system)

    def test_label_rules_cover_degree_emphasis_no_condition_and_facts(self):
        system = PAIR_SYSTEM
        self.assertIn("요구가 여럿인데 일부만 충족하면 낮춘다", system)
        self.assertIn("정도나 한도를 나타내는 말", system)
        self.assertIn("강조된 요구는 매력 원문이 그 특성을 분명하고 강하게 갖췄을 때 높은 라벨", system)
        self.assertIn("조건이 없는 요구(가리지 않음)는", system)
        self.assertIn("사실 정보(background)는 같은 사실이 분명할 때만 높은 라벨", system)
        self.assertIn("두 라벨 사이에서 망설여지면 낮은 쪽을 고른다", system)
        for line in (
            "「성격이 진짜x100 다정한 사람」 / 「다정한 편이에요」 → 부분 일치",
            "「진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고」 / 「장난기 많고 가벼운 편」 → 무관/충돌",
            "「상대가 좋다면 어떤 취미라도 즐겨요. 독서빼고요」 / 「독서, 글쓰기」 → 무관/충돌",
            "「군필」 / 「미필」 → 무관/충돌",
        ):
            self.assertIn(line, system)

    def test_schema_asks_for_quotes_and_reason_before_the_label(self):
        self.assertEqual(list(CriterionJudgement.model_fields), ["want_quote", "have_quote", "reason", "label"])
        self.assertEqual(tuple(PairEvaluation.model_fields), CRITERIA)
        self.assertEqual(set(get_args(ex_mod.Label)), {NOT_REQUESTED, NO_INFO, *MATCH_LABELS})
        self.assertEqual(set(ex_mod.SCORE_BY_LABEL), {NO_INFO, *MATCH_LABELS})  # 요구 없음은 점수가 없다

    def test_user_message_carries_both_texts(self):
        message = user_message("이상형 글", "매력 글")
        self.assertIn("## 이상형 원문 (상대에게 원하는 모습)\n이상형 글", message)
        self.assertIn("## 매력 원문 (스스로 적은 자기 모습)\n매력 글", message)
        self.assertIn(EMPTY_HAVE, user_message("이상형 글", ""))
        self.assertIn("## 이전 답의 문제\n문제", user_message("이상형 글", "매력 글", "문제"))

    def test_prompt_version_names_the_pair_prompt(self):
        self.assertTrue(PAIR_PROMPT_VERSION.startswith("pair-v1-"))


# --------------------------------------------------------------------------- ex_trait 추출 (extract_ex_traits.py)


class VerbatimValidationTests(unittest.TestCase):
    """evidence는 원문 그대로여야 하고, statement는 길이와 거절 표현만 본다."""

    SOURCE = "잘 웃고, 배려심 있는 사람이면 좋겠어요! 담배 안 피는 사람이 좋아요. 술 마시는 사람은 싫어요."

    def test_evidence_ignores_spacing_and_punctuation(self):
        found = traits_mod.find_violations(
            {
                "personality": val("잘 웃고 배려심 있는 사람", "잘 웃고 배려심 있는 사람이면 좋겠어요"),
                "lifestyle": val("담배를 피우지 않는 사람", "담배 안 피는 사람이 좋아요"),
            },
            self.SOURCE,
        )
        self.assertEqual(found, [])

    def test_statement_may_be_rewritten_freely(self):
        # statement는 재작성이므로 원문에 없는 글자여도 된다. 근거(evidence)만 원문이어야 한다.
        found = traits_mod.find_violations({"personality": val("다정하고 친절한 사람", "배려심 있는 사람")}, self.SOURCE)
        self.assertEqual(found, [])

    def test_evidence_not_in_source_is_a_violation(self):
        found = traits_mod.find_violations({"personality": val("친절한 사람", "다정하고 친절한 사람")}, self.SOURCE)
        self.assertEqual(len(found), 1)
        self.assertIn("personality", found[0])
        self.assertIn("evidence", found[0])

    def test_each_evidence_piece_is_checked_one_by_one(self):
        ok = traits_mod.find_violations(
            {"lifestyle": val("담배를 피우지 않는 사람", "담배 안 피는 사람이 좋아요", "배려심 있는 사람")}, self.SOURCE
        )
        self.assertEqual(ok, [])
        bad = traits_mod.find_violations(
            {"lifestyle": val("담배를 피우지 않는 사람", "담배 안 피는 사람이 좋아요", "운동하는 사람")}, self.SOURCE
        )
        self.assertEqual(len(bad), 1)

    def test_empty_evidence_is_a_violation(self):
        found = traits_mod.find_violations({"interests": val("영화")}, self.SOURCE)
        self.assertEqual(len(found), 1)
        self.assertIn("evidence가 비어", found[0])

    def test_evidence_without_letters_is_a_violation(self):
        self.assertEqual(len(traits_mod.find_violations({"interests": val("맥주", "🍺")}, self.SOURCE)), 1)

    def test_blank_statement_is_a_violation(self):
        found = traits_mod.find_violations({"interests": val("  ", "담배 안 피는 사람이 좋아요")}, self.SOURCE)
        self.assertEqual(len(found), 1)
        self.assertIn("statement가 비어", found[0])

    def test_statement_that_still_contains_a_rejection_is_a_violation(self):
        for statement in ("술 마시는 사람은 싫어요", "너무 진지한 사람 말고", "과하게 예쁜 사람은 부담스러운"):
            with self.subTest(statement=statement):
                found = traits_mod.find_violations({"lifestyle": val(statement, "술 마시는 사람은 싫어요")}, self.SOURCE)
                self.assertEqual(len(found), 1)
                self.assertIn("거절 표현", found[0])

    def test_rewritten_rejection_passes(self):
        for statement in ("술을 마시지 않는 사람", "적당히 진지한 사람", "과하게 예쁘지 않은 외모"):
            with self.subTest(statement=statement):
                found = traits_mod.find_violations({"lifestyle": val(statement, "술 마시는 사람은 싫어요")}, self.SOURCE)
                self.assertEqual(found, [])

    def test_overlong_statement_is_a_violation(self):
        found = traits_mod.find_violations(
            {"lifestyle": val("가" * (traits_mod.MAX_STATEMENT_CHARS + 1), "담배 안 피는 사람이 좋아요")}, self.SOURCE
        )
        self.assertEqual(len(found), 1)

    def test_null_criteria_are_not_checked(self):
        self.assertEqual(traits_mod.find_violations({key: None for key in traits_mod.CRITERIA}, self.SOURCE), [])

    def test_blank_string_becomes_none(self):
        self.assertIsNone(traits_mod.clean_value("   "))
        self.assertEqual(traits_mod.clean_value(" 영화 "), "영화")


class ExtractionTests(unittest.TestCase):
    TEXT = "강아지상이라는 말 자주 들어요. 운동 좋아하고 아침형 인간이에요."
    GOOD = {
        "impression": val("강아지상", "강아지상이라는 말 자주 들어요"),
        "lifestyle": val("운동을 좋아하는 아침형 인간", "운동 좋아하고 아침형 인간이에요"),
    }
    BAD = {"impression": val("순한 강아지 같은 인상", "순한 강아지 같은 인상")}  # evidence가 원문에 없다

    def fake_parse(self, responses):
        queue = list(responses)
        seen = []

        def parse(model, system, messages, response_format):
            seen.append(messages[0]["content"])
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return response_format(**{key: item.get(key) for key in traits_mod.CRITERIA}), "{}"

        parse.seen = seen
        return parse

    def test_valid_response_passes_on_first_attempt(self):
        outcome = traits_mod.extract_traits("have", self.TEXT, MODEL, parse=self.fake_parse([self.GOOD]))
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.attempts, 1)
        self.assertIsNone(outcome.traits["personality"])

    def test_only_the_rewritten_statement_becomes_the_trait(self):
        """채점에 쓰이는 값(traits)은 statement이고, 원문 근거는 따로 담긴다."""
        outcome = traits_mod.extract_traits("have", self.TEXT, MODEL, parse=self.fake_parse([self.GOOD]))
        self.assertEqual(outcome.traits["lifestyle"], "운동을 좋아하는 아침형 인간")
        self.assertEqual(outcome.evidence["lifestyle"], ["운동 좋아하고 아침형 인간이에요"])
        self.assertNotIn("운동 좋아하고 아침형 인간이에요", outcome.traits.values())
        self.assertNotIn("personality", outcome.evidence)
        # 근거는 사람이 보도록 raw_response에도 남는다.
        self.assertEqual(outcome.raw["response"]["lifestyle"]["evidence"], ["운동 좋아하고 아침형 인간이에요"])

    def test_retry_feeds_back_the_violation_and_then_succeeds(self):
        parse = self.fake_parse([self.BAD, self.GOOD])
        outcome = traits_mod.extract_traits("have", self.TEXT, MODEL, parse=parse)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.attempts, 2)
        self.assertIn("evidence가 원문에 없는 표현", parse.seen[1])
        self.assertNotIn("이전 시도의 문제", parse.seen[0])

    def test_leftover_rejection_in_statement_is_retried(self):
        text = "술 마시는 사람은 싫어요."
        leftover = {"lifestyle": val("술 마시는 사람은 싫어요", "술 마시는 사람은 싫어요")}
        fixed = {"lifestyle": val("술을 마시지 않는 사람", "술 마시는 사람은 싫어요")}
        parse = self.fake_parse([leftover, fixed])
        outcome = traits_mod.extract_traits("want", text, MODEL, parse=parse)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.traits["lifestyle"], "술을 마시지 않는 사람")
        self.assertIn("거절 표현", parse.seen[1])

    def test_three_failures_mark_extraction_failed_with_all_traits_null(self):
        outcome = traits_mod.extract_traits("have", self.TEXT, MODEL, parse=self.fake_parse([self.BAD, self.BAD, self.BAD]))
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.attempts, 3)
        self.assertTrue(outcome.error)
        self.assertTrue(all(value is None for value in outcome.traits.values()))
        self.assertEqual(outcome.evidence, {})

    def test_api_error_counts_as_an_attempt_and_can_recover(self):
        parse = self.fake_parse([RuntimeError("boom"), self.GOOD])
        outcome = traits_mod.extract_traits("have", self.TEXT, MODEL, parse=parse)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.attempts, 2)
        self.assertIn("boom", outcome.raw["attempt_errors"][0])

    def test_nothing_to_extract_is_a_valid_empty_result(self):
        outcome = traits_mod.extract_traits("want", self.TEXT, MODEL, parse=self.fake_parse([{}]))
        self.assertTrue(outcome.ok)
        self.assertTrue(all(value is None for value in outcome.traits.values()))


class ExtractionPromptTests(unittest.TestCase):
    """실제 원문에서 놓쳤던 구절이 추출 프롬프트·기준으로 다뤄지는지 지킨다. (LLM은 부르지 않는다.)"""

    def test_background_is_an_extracted_and_compared_criterion(self):
        self.assertEqual(traits_mod.CRITERIA[-1], "background")
        self.assertIn("background", traits_mod.TraitExtraction.model_fields)
        self.assertIn("background", ex_mod.ANCHORS)
        self.assertIn("background", PAIR_SYSTEM)
        self.assertIn("background", traits_mod.EXTRACT_SYSTEM)

    def test_missed_phrases_have_a_home_in_the_hints(self):
        self.assertIn("맞춰주는", traits_mod.CRITERION_HINT["personality"])
        self.assertIn("안아주는", traits_mod.CRITERION_HINT["relationship_values"])
        self.assertIn("군필", traits_mod.CRITERION_HINT["background"])
        # 규칙 6이 이 구절들을 어느 기준에 넣을지 직접 알려 준다.
        self.assertIn('"잘 맞춰줘요"', traits_mod.EXTRACT_SYSTEM)
        self.assertIn('"잘 안아줄 수 있는 사람"', traits_mod.EXTRACT_SYSTEM)

    def test_values_go_to_personality_and_relationship_values_is_only_about_dating(self):
        hint = traits_mod.CRITERION_HINT
        self.assertEqual(traits_mod.CRITERION_KO["personality"], "성격/가치관")
        self.assertEqual(traits_mod.CRITERION_KO["relationship_values"], "연애관")
        self.assertIn("가치관", hint["personality"])
        self.assertNotIn("가치관", hint["relationship_values"])
        self.assertIn("연애에 관한 것만", hint["relationship_values"])
        prompt = traits_mod.EXTRACT_SYSTEM
        self.assertIn("가치관(인생관·가족관 등)은 personality에 넣는다", prompt)
        self.assertIn("relationship_values에는 연락 빈도·애정표현·스킨십 같은 연애관만 넣는다", prompt)
        # 결혼관은 어느 기준의 설명·예시에도 없고, 넣지 말라고만 한 번 적혀 있다.
        for key, text in hint.items():
            self.assertNotIn("결혼", text, key)
        self.assertIn("결혼관(결혼에 대한 생각·결혼 계획)은 어느 기준에도 넣지 않는다", prompt)
        mentions = [line for line in prompt.splitlines() if "결혼" in line]
        self.assertEqual(len(mentions), 1, mentions)  # 제외 규칙 한 줄 외에는 어디에도 없다

    def test_rejections_are_rewritten_not_dropped_or_copied(self):
        prompt = traits_mod.EXTRACT_SYSTEM
        # 완전한 거절은 "~하지 않는", 한도는 정도 표현, 범위에서 빼는 말은 "~를 제외한"으로 바꿔 쓰라고 알려 준다.
        self.assertIn('"OOO한 사람은 싫어요/별로예요" → "OOO하지 않는 사람"', prompt)
        self.assertIn('"진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고" → "적당히 진지한 사람"', prompt)
        self.assertIn('"독서는 빼고요" → 앞 구절의 의미 뒤에 "(독서 제외)"를 붙여 쓴다', prompt)
        self.assertIn("거절 표현을 남기지 않는다", prompt)
        # 요구를 말하는 구절은 '안'이 들어 있어도 그대로 쓴다.
        self.assertIn('"담배 안 피는 사람이 좋아요" → "담배를 피우지 않는 사람"', prompt)

    def test_rewritten_form_of_the_real_mixed_text_passes_validation(self):
        """실제 원문(MIXED_WANT)에서 재작성 + 근거 인용을 한 결과가 검증을 통과한다."""
        rewritten = {
            "personality": val("적당히 진지한 사람", "진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고"),
            "appearance": val(
                "송하영 느낌이되 과하게 예쁘지 않은 사람",
                "외적인 이상형은 송하영 느낌",
                "너무 압도적으로 예쁜 사람은 상대하기 어려워요",
            ),
            "interests": val("상대의 취미는 가리지 않음(독서 제외)", "상대가 좋다면 어떤 취미라도 즐깁니다", "독서빼고요"),
        }
        self.assertEqual(traits_mod.find_violations(rewritten, MIXED_WANT), [])
        # 같은 결과에서 거절 표현을 그대로 남기면 걸린다.
        copied = {"personality": val("너무 진지한 사람 말고", "그렇다고 너무 진지한 사람 말고")}
        self.assertEqual(len(traits_mod.find_violations(copied, MIXED_WANT)), 1)

    def test_writers_own_attitude_is_not_turned_into_a_condition_on_the_partner(self):
        """실제로 왜곡됐던 사례: 본인이 어떤 취미든 즐긴다는 말이 '어떤 취미든 즐기는 사람'이라는 상대 조건으로 바뀌었다."""
        prompt = traits_mod.EXTRACT_SYSTEM
        self.assertIn("구절의 주체를 바꾸지 않는다", prompt)
        self.assertIn("글쓴이 본인의 너그러움·허용", prompt)
        self.assertIn('"상대의 ~는 가리지 않음"이라고 쓴다', prompt)
        # 잘못된 재작성을 틀린 예로 보여 준다.
        self.assertIn("× \"독서를 제외한 어떤 취미라도 함께 즐기는 사람\"", prompt)
        # 올바른 재작성 예.
        self.assertIn('interests: "상대의 취미는 가리지 않음(독서 제외)"', prompt)
        # 상대에 대한 요구를 말하는 "저는 ~ 사람이 좋아요/어려워요"는 계속 조건으로 쓴다.
        self.assertIn('"저는 ~한 사람이 좋아요", "저는 ~한 사람은 어려워요"도 상대의 모습을 말하므로 조건이다', prompt)
        # 본인 소개는 want에, 상대에 대한 바람은 have에 넣지 않는다.
        self.assertIn("본인을 소개하는 말", prompt)
        self.assertIn("have 글에서 상대에게 바라는 말은 본인이 가진 모습이 아니므로 넣지 않는다", prompt)

    def test_nonstandard_emphasis_is_carried_into_the_rewrite(self):
        """실제로 사라졌던 사례: '성격이 진짜x100000 다정한 사람'이 강조 없이 '다정하며'로 재작성됐다."""
        prompt = traits_mod.EXTRACT_SYSTEM
        self.assertIn("강조는 지우지 않고 정도를 나타내는 말로 옮긴다", prompt)
        for slang in ('"진짜"', '"완전"', '"개"', '"ㄹㅇ"', '"x100000"', '"다정다정"'):
            self.assertIn(slang, prompt)
        self.assertIn('"성격이 진짜x100000 다정한 사람" → "무엇보다 다정한 사람"', prompt)
        self.assertIn('× "다정한 사람" — 강조가 사라진 오류다', prompt)
        self.assertIn("같은 기준에 속한 특성은 빠뜨리지 않고 모두 담는다", prompt)
        # 예시: 비표준 강조("진짜x9999", "개웃기면")를 정도 표현으로 옮기고, 같은 기준의 다른 특성도 남긴다.
        example = next(item for item in traits_mod.EXAMPLES if "비표준 강조" in item[1])
        statement, evidence = example[3]["personality"]
        self.assertIn("진짜x9999", " ".join(evidence))
        self.assertIn("무엇보다 다정", statement)
        self.assertIn("매우 웃기", statement)
        self.assertIn("낭만을 아는", statement)  # 강조 때문에 다른 특성이 빠지지 않는다
        self.assertNotIn("x9999", statement)  # 이상한 표기는 정리되어 statement에 남지 않는다

    def test_unsure_statements_are_not_extracted(self):
        prompt = traits_mod.EXTRACT_SYSTEM
        self.assertIn("확신 없는 말은 넣지 않는다", prompt)
        self.assertIn("evidence에도 statement에도 쓰지 않는다", prompt)
        self.assertIn("굳이 적자면 직진남이에요", prompt)  # 망설이는 말이 붙어도 단정하면 넣는다
        self.assertIn('"키가 작은 것도 매력 맞죠..?"는 확신 없는 질문이라 넣지 않는다', prompt)
        # 확신 없는 말의 목록과, 허용하는 추측형 어미.
        for phrase in ('"~맞죠?"', '"~잘 모르겠어요"', '"~일까요?"'):
            self.assertIn(phrase, prompt)
        self.assertIn('"~같아요", "~인 것 같아요"는 허용한다', prompt)
        self.assertIn('"내향적인 것 같아요" → "내향적인 편"', prompt)
        self.assertIn('"장점이 뭔지는 잘 모르겠어요"는 확신 없는 말이라 넣지 않는다', prompt)
        # 한도는 "적당히 ~한"처럼 정도를 나타내는 말로 바꿔 쓰라고 한다. 스키마 설명도 같다.
        self.assertIn('"적당히 ~한"처럼 정도를 나타내는 말로 바꿔 쓴다', prompt)
        description = traits_mod.TraitValue.model_fields["statement"].description
        self.assertIn("적당히 ~한", description)
        self.assertIn("잘 모르겠어요", description)

    def test_prompt_examples_are_valid_by_our_own_rules(self):
        """프롬프트에 실은 예시가 규칙을 어기면 모델이 그걸 따라 한다. 예시 데이터를 같은 검증으로 확인한다."""
        for side, title, text, items, _note in traits_mod.EXAMPLES:
            with self.subTest(example=title or side):
                response = {key: val(statement, *evidence) for key, (statement, evidence) in items.items()}
                self.assertEqual(traits_mod.find_violations(response, text), [])
                self.assertIn(text, traits_mod.EXTRACT_SYSTEM)
                for statement, evidence in items.values():
                    self.assertIn(f'statement "{statement}"', traits_mod.EXTRACT_SYSTEM)
                    for piece in evidence:
                        self.assertIn(f'"{piece}"', traits_mod.EXTRACT_SYSTEM)
                    self.assertNotIn("맞죠", statement)

    def test_examples_cover_the_phrases_that_were_missed(self):
        examples = " ".join(f"{text} {items}" for _s, _t, text, items, _n in traits_mod.EXAMPLES)
        for phrase in ("담배를 피우지 않", "과하게 느끼하지 않은", "과하게 예쁘지 않은", "군필", "잘 안아주는", "잘 맞춰주는"):
            self.assertIn(phrase, examples)


class TraitReadinessTests(unittest.TestCase):
    def row(self, text, ok=True, version=None):
        return traits_mod.TraitRow("r1", "want", ok, text, version or traits_mod.EXTRACT_PROMPT_VERSION, traits(interests="영화"))

    def test_usable_only_when_text_version_and_ok_match(self):
        self.assertIsNotNone(traits_mod.usable_traits("영화 좋아요", self.row("영화 좋아요")))
        self.assertIsNone(traits_mod.usable_traits("영화 좋아요", self.row("영화 좋아요!")))
        self.assertIsNone(traits_mod.usable_traits("영화 좋아요", self.row("영화 좋아요", ok=False)))
        self.assertIsNone(traits_mod.usable_traits("영화 좋아요", self.row("영화 좋아요", version="ext-v1-old")))
        self.assertIsNone(traits_mod.usable_traits("영화 좋아요", None))
        self.assertIsNone(traits_mod.usable_traits("  ", self.row("  ")))

    def test_problems_list_every_text_without_a_valid_row(self):
        students = {"r1": {"ex_want": "영화 좋아요", "ex_have": None}, "r2": {"ex_want": "", "ex_have": "운동해요"}}
        book = traits_mod.TraitBook({("r1", "want"): self.row("영화 좋아요")})
        problems = traits_mod.trait_problems(students, book)
        self.assertEqual(problems, ["r2 have: 추출 안 됨"])


# --------------------------------------------------------------------------- 최종 점수와 매칭


class FinalScoreTests(unittest.TestCase):
    W = (match_main.W_MBTI, match_main.W_TAG, match_main.W_EX)

    def test_weights_are_one_to_six_to_three_like_the_previous_code(self):
        self.assertEqual(self.W, (1.0, 6.0, 3.0))
        old = 0.5 * 0.1 + 0.4 * 0.6 + 0.8 * 0.3
        self.assertAlmostEqual(final_score(0.5, 0.4, 0.8, "both", *self.W), old)

    def test_none_mode_recomputes_without_ex(self):
        self.assertAlmostEqual(final_score(0.5, 0.4, 0.0, "none", *self.W), (0.5 * 1 + 0.4 * 6) / 7)

    def test_one_direction_keeps_the_ex_term(self):
        self.assertAlmostEqual(final_score(0.5, 0.4, 0.25, "one_direction", *self.W), (0.5 + 2.4 + 0.75) / 10)


class MatchingOrderTests(unittest.TestCase):
    def scored(self, male, female, final, tag_min=0.0, ex=0.0, mbti=0.0):
        return {
            "male_id": male, "female_id": female, "final_score": final,
            "tag_min_incl": tag_min, "ex_score": ex, "mbti_score": mbti,
        }

    def test_ties_break_by_tag_inclusion_then_ex_then_mbti(self):
        rows = [
            self.scored("m1", "f1", 0.5, tag_min=0.2),
            self.scored("m2", "f2", 0.5, tag_min=0.4),
            self.scored("m3", "f3", 0.5, tag_min=0.4, ex=0.3),
            self.scored("m4", "f4", 0.5, tag_min=0.4, ex=0.3, mbti=0.5),
        ]
        order = [item["male_id"] for item in match_main.pick_unique_matches(rows)]
        self.assertEqual(order, ["m4", "m3", "m2", "m1"])

    def test_remaining_ties_are_stable_across_input_order(self):
        rows = [self.scored(f"m{i}", f"f{i}", 0.5) for i in range(6)]
        first = [item["male_id"] for item in match_main.pick_unique_matches(rows)]
        second = [item["male_id"] for item in match_main.pick_unique_matches(list(reversed(rows)))]
        self.assertEqual(first, second)

    def test_greedy_pairing_skips_people_already_taken(self):
        rows = [self.scored("m1", "f1", 0.9), self.scored("m1", "f2", 0.8), self.scored("m2", "f1", 0.7), self.scored("m2", "f2", 0.6)]
        picked = [(item["male_id"], item["female_id"]) for item in match_main.pick_unique_matches(rows)]
        self.assertEqual(picked, [("m1", "f1"), ("m2", "f2")])

    def test_tag_inclusion_min_is_the_smaller_direction(self):
        # A→B: {a,b}∩{a,b} = 2/2 = 1.0, B→A: {x,y}∩{x} = 1/2 = 0.5 → 작은 값 0.5
        self.assertEqual(tag_inclusion_min(["a", "b"], ["x"], ["x", "y"], ["a", "b"]), 0.5)
        self.assertEqual(tag_inclusion_min(["a", "b"], [], ["x"], ["a", "b"]), 0.0)
        self.assertEqual(tag_inclusion_min([], [], ["x"], ["x"]), 0.0)

    def test_old_format_scores_csv_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scores.csv")
            old_header = [name for name in match_main.SCORE_FIELDS if name != "tag_min_incl"]
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(old_header)
            with self.assertRaises(RuntimeError):
                match_main.ensure_scores_header_current(path)
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(match_main.SCORE_FIELDS)
            match_main.ensure_scores_header_current(path)  # 새 형식이면 통과
            match_main.ensure_scores_header_current(os.path.join(tmp, "missing.csv"))  # 파일이 없으면 통과

    def test_scores_csv_from_an_older_ex_version_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scores.csv")

            def write(version):
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=match_main.SCORE_FIELDS)
                    writer.writeheader()
                    writer.writerow({"row_num": 1, "male_id": "m1", "female_id": "f1", "status": "skipped_age"})
                    writer.writerow(
                        {
                            "row_num": 2, "male_id": "m1", "female_id": "f2", "status": "scored",
                            "ex_detail": json.dumps({"version": version}),
                        }
                    )

            write("ex-v2")
            with self.assertRaises(RuntimeError):
                match_main.ensure_scores_header_current(path)
            write(ex_mod.EX_DETAIL_VERSION)
            match_main.ensure_scores_header_current(path)


class ModelResolutionTests(unittest.TestCase):
    def test_model_must_be_set_and_never_latest(self):
        from score_functions import llm

        saved = os.environ.get("OPENAI_MODEL")
        try:
            os.environ.pop("OPENAI_MODEL", None)
            with self.assertRaises(RuntimeError):
                llm.resolve_model()
            os.environ["OPENAI_MODEL"] = "some-model-latest"
            with self.assertRaises(RuntimeError):
                llm.resolve_model()
            os.environ["OPENAI_MODEL"] = "some-model-2026-01-01"
            self.assertEqual(llm.resolve_model(), "some-model-2026-01-01")
        finally:
            if saved is None:
                os.environ.pop("OPENAI_MODEL", None)
            else:
                os.environ["OPENAI_MODEL"] = saved


if __name__ == "__main__":
    unittest.main()
