"""ex 점수 개선안 v2 단위 테스트. OpenAI·Aurora를 부르지 않는다.

  python -m unittest discover -s tests -v
"""

import csv
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as match_main  # noqa: E402
from score_functions import ex as ex_mod  # noqa: E402
from score_functions import ex_traits as traits_mod  # noqa: E402
from score_functions.ex import (  # noqa: E402
    COMPARE_PROMPT_VERSION,
    COMPARE_SYSTEM,
    Item,
    ItemScorer,
    MemoryItemStore,
    cache_key,
    direction_score,
    ex_score,
    final_score,
    llm_items,
)
from score_functions.tag import tag_inclusion_min  # noqa: E402

MODEL = "test-model-2026-01-01"


def traits(**values):
    return {key: values.get(key) for key in traits_mod.CRITERIA}


class ScriptedJudge:
    """criterion별로 호출 순서대로 라벨을 돌려주는 가짜 LLM. 호출 기록을 남긴다."""

    def __init__(self, script: dict[str, list[str]]):
        self.script = {key: list(labels) for key, labels in script.items()}
        self.calls: list[list[str]] = []

    def __call__(self, items):
        self.calls.append([item.criterion for item in items])
        return {item.criterion: (self.script[item.criterion].pop(0), f"{item.criterion} 이유") for item in items}


def scorer_with(script: dict[str, list[str]], store=None):
    judge = ScriptedJudge(script)
    return ItemScorer(MemoryItemStore() if store is None else store, MODEL, judge=judge), judge


class VerbatimValidationTests(unittest.TestCase):
    SOURCE = "잘 웃고, 배려심 있는 사람이면 좋겠어요! 담배 안 피는 사람이 좋아요. 술 마시는 사람은 싫어요."

    def test_excerpt_ignores_spacing_and_punctuation(self):
        found = traits_mod.find_violations(traits(personality="잘 웃고 배려심 있는 사람", lifestyle="담배 안 피는 사람이 좋아요"), self.SOURCE)
        self.assertEqual(found, [])

    def test_paraphrase_is_a_violation(self):
        found = traits_mod.find_violations(traits(personality="다정하고 친절한 사람"), self.SOURCE)
        self.assertEqual(len(found), 1)
        self.assertIn("personality", found[0])

    def test_joined_excerpts_are_checked_one_by_one(self):
        ok = traits_mod.find_violations(traits(lifestyle="담배 안 피는 사람이 좋아요 | 배려심 있는 사람"), self.SOURCE)
        self.assertEqual(ok, [])
        bad = traits_mod.find_violations(traits(lifestyle="담배 안 피는 사람이 좋아요 | 운동하는 사람"), self.SOURCE)
        self.assertEqual(len(bad), 1)

    def test_value_without_letters_is_a_violation(self):
        self.assertEqual(len(traits_mod.find_violations(traits(interests="🍺"), self.SOURCE)), 1)

    def test_blank_string_becomes_none(self):
        self.assertIsNone(traits_mod.clean_value("   "))
        self.assertEqual(traits_mod.clean_value(" 영화 "), "영화")


class ExtractionTests(unittest.TestCase):
    TEXT = "강아지상이라는 말 자주 들어요. 운동 좋아하고 아침형 인간이에요."

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
        parse = self.fake_parse([{"impression": "강아지상", "lifestyle": "운동 좋아하고 아침형 인간"}])
        outcome = traits_mod.extract_traits("have", self.TEXT, MODEL, parse=parse)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.attempts, 1)
        self.assertEqual(outcome.traits["impression"], "강아지상")
        self.assertIsNone(outcome.traits["personality"])

    def test_retry_feeds_back_the_violation_and_then_succeeds(self):
        parse = self.fake_parse([{"impression": "순한 강아지 같은 인상"}, {"impression": "강아지상"}])
        outcome = traits_mod.extract_traits("have", self.TEXT, MODEL, parse=parse)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.attempts, 2)
        self.assertIn("원문에 없는 표현", parse.seen[1])
        self.assertNotIn("이전 시도의 문제", parse.seen[0])

    def test_three_failures_mark_extraction_failed_with_all_traits_null(self):
        bad = {"impression": "순한 강아지 같은 인상"}
        outcome = traits_mod.extract_traits("have", self.TEXT, MODEL, parse=self.fake_parse([bad, bad, bad]))
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.attempts, 3)
        self.assertTrue(outcome.error)
        self.assertTrue(all(value is None for value in outcome.traits.values()))

    def test_api_error_counts_as_an_attempt_and_can_recover(self):
        parse = self.fake_parse([RuntimeError("boom"), {"impression": "강아지상"}])
        outcome = traits_mod.extract_traits("have", self.TEXT, MODEL, parse=parse)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.attempts, 2)
        self.assertIn("boom", outcome.raw["attempt_errors"][0])

    def test_nothing_to_extract_is_a_valid_empty_result(self):
        outcome = traits_mod.extract_traits("want", self.TEXT, MODEL, parse=self.fake_parse([{}]))
        self.assertTrue(outcome.ok)
        self.assertTrue(all(value is None for value in outcome.traits.values()))


class PromptCoverageTests(unittest.TestCase):
    """실제 원문에서 놓쳤던 구절이 프롬프트·기준으로 다뤄지는지 지킨다. (LLM은 부르지 않는다.)"""

    # 한정하는 부정 표현과 떨어져 있는 같은 기준 구절이 섞인 원문.
    MIXED = (
        "진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고 ㅎ\n"
        "외적인 이상형은 송하영 느낌 ㅎ\n"
        "상대가 좋다면 어떤 취미라도 즐깁니다.\n"
        "독서빼고요. 근데 저는 너무 압도적으로 예쁜 사람은 상대하기 어려워요.. 부담감 생겨버림 ㅠ"
    )

    def test_background_is_an_extracted_and_compared_criterion(self):
        self.assertEqual(traits_mod.CRITERIA[-1], "background")
        self.assertIn("background", traits_mod.TraitExtraction.model_fields)
        self.assertIn("background", ex_mod.ANCHORS)
        self.assertIn("background", ex_mod.COMPARE_SYSTEM)
        self.assertIn("background", traits_mod.EXTRACT_SYSTEM)

    def test_missed_phrases_have_a_home_in_the_hints(self):
        self.assertIn("맞춰주는", traits_mod.CRITERION_HINT["personality"])
        self.assertIn("안아주는", traits_mod.CRITERION_HINT["relationship_values"])
        self.assertIn("군필", traits_mod.CRITERION_HINT["background"])
        # 규칙 4가 이 구절들을 어느 기준에 넣을지 직접 알려 준다.
        self.assertIn('"잘 맞춰줘요"', traits_mod.EXTRACT_SYSTEM)
        self.assertIn('"잘 안아줄 수 있는 사람"', traits_mod.EXTRACT_SYSTEM)

    def test_qualifying_negation_is_extracted_together_with_the_positive(self):
        found = traits_mod.find_violations(
            traits(
                personality="진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고",
                appearance="송하영 느낌 | 너무 압도적으로 예쁜 사람은 상대하기 어려워요.. 부담감 생겨버림",
                interests="상대가 좋다면 어떤 취미라도 즐깁니다.",
            ),
            self.MIXED,
        )
        self.assertEqual(found, [])
        self.assertIn("정도·범위를 한정하는 말", traits_mod.EXTRACT_SYSTEM)

    def test_independent_dislike_stays_excluded(self):
        # "독서빼고요"는 다른 대상에 대한 독립된 기피 표현이라 넣지 않는다고 프롬프트가 알려 준다.
        self.assertIn("독서는 빼고요", traits_mod.EXTRACT_SYSTEM)
        self.assertIn("다른 대상에 대한 기피", traits_mod.EXTRACT_SYSTEM)

    def test_prompt_examples_are_verbatim_excerpts_of_their_own_text(self):
        """프롬프트에 실은 예시가 원문에 없는 글자를 가르치면 안 된다."""
        want_text = (
            "다정한 사람 좋아하고 그렇다고 너무 느끼한 사람은 말고요. 외모는 아이유 느낌이 좋은데 너무 압도적으로 예쁜 사람은 부담스러워요. "
            "군필이면 좋겠고 잘 안아주는 사람이었으면 해요. 책 읽는 건 별로예요."
        )
        want = traits(
            personality="다정한 사람 좋아하고 그렇다고 너무 느끼한 사람은 말고요",
            appearance="외모는 아이유 느낌이 좋은데 너무 압도적으로 예쁜 사람은 부담스러워요",
            background="군필이면 좋겠고",
            relationship_values="잘 안아주는 사람이었으면 해요",
        )
        have_text = "군필이고 IT 회사 다녀요. 상대에게 잘 맞춰주는 편이에요."
        have = traits(background="군필이고 IT 회사 다녀요", personality="상대에게 잘 맞춰주는 편이에요")
        self.assertEqual(traits_mod.find_violations(want, want_text), [])
        self.assertEqual(traits_mod.find_violations(have, have_text), [])
        for text in (want_text, have_text, *[v for v in (*want.values(), *have.values()) if v]):
            self.assertIn(text, traits_mod.EXTRACT_SYSTEM)

    def test_compare_prompt_reads_qualifiers_and_facts(self):
        self.assertIn("정도를 한정하는 말", ex_mod.COMPARE_SYSTEM)
        self.assertIn("사실 정보(background)", ex_mod.COMPARE_SYSTEM)
        self.assertIn("「군필」 / 가진 모습 「미필」 → 무관/충돌", ex_mod.COMPARE_SYSTEM)


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


class ConsensusTests(unittest.TestCase):
    ITEM = Item("lifestyle", "담배 안 피는", "비흡연자")

    def test_three_agreeing_calls_do_not_resample(self):
        scorer, judge = scorer_with({"lifestyle": ["동일", "동일", "동일"]})
        result = scorer.score_items([self.ITEM])[self.ITEM]
        self.assertEqual((result.label, result.score, result.spread, result.resampled), ("동일", 1.0, 0.0, False))
        self.assertEqual(len(judge.calls), 3)

    def test_small_disagreement_uses_the_median_without_resampling(self):
        scorer, judge = scorer_with({"lifestyle": ["동일", "거의 일치", "거의 일치"]})
        result = scorer.score_items([self.ITEM])[self.ITEM]
        self.assertEqual((result.label, result.score, result.resampled), ("거의 일치", 0.75, False))
        self.assertEqual(len(judge.calls), 3)

    def test_wide_disagreement_resamples_to_five_and_takes_the_median(self):
        scorer, judge = scorer_with({"lifestyle": ["동일", "무관/충돌", "동일", "동일", "거의 일치"]})
        result = scorer.score_items([self.ITEM])[self.ITEM]
        self.assertTrue(result.resampled)
        self.assertEqual(len(result.samples), 5)
        self.assertEqual((result.label, result.score), ("동일", 1.0))
        self.assertEqual(result.spread, 1.0)
        self.assertEqual(len(judge.calls), 5)

    def test_resample_only_re_asks_the_disputed_items(self):
        a = Item("lifestyle", "아침형", "일찍 일어나는")
        b = Item("interests", "영화", "전시회")
        scorer, judge = scorer_with(
            {
                "lifestyle": ["동일", "동일", "동일"],
                "interests": ["약한 관련", "동일", "약한 관련", "약한 관련", "약한 관련"],
            }
        )
        out = scorer.score_items([a, b])
        self.assertEqual(judge.calls[:3], [["lifestyle", "interests"]] * 3)  # 한 호출에 그 쌍의 항목만, 3회
        self.assertEqual(judge.calls[3:], [["interests"]] * 2)  # 재샘플은 불일치 항목만
        self.assertFalse(out[a].resampled)
        self.assertTrue(out[b].resampled)
        self.assertEqual(out[b].label, "약한 관련")

    def test_cached_item_is_not_asked_again_and_keeps_its_first_score(self):
        store = MemoryItemStore()
        scorer, judge = scorer_with({"lifestyle": ["동일"] * 3}, store)
        first = scorer.score_items([self.ITEM])[self.ITEM]
        again, judge2 = scorer_with({"lifestyle": ["무관/충돌"] * 3}, store)
        second = again.score_items([self.ITEM])[self.ITEM]
        self.assertEqual(judge2.calls, [])
        self.assertEqual(second.score, first.score)

    def test_cache_key_depends_on_every_input(self):
        base = cache_key(self.ITEM, MODEL)
        self.assertEqual(base, cache_key(Item("lifestyle", "담배 안 피는", "비흡연자"), MODEL))
        self.assertNotEqual(base, cache_key(Item("interests", "담배 안 피는", "비흡연자"), MODEL))
        self.assertNotEqual(base, cache_key(Item("lifestyle", "담배 안 피는 사람", "비흡연자"), MODEL))
        self.assertNotEqual(base, cache_key(Item("lifestyle", "담배 안 피는", "금연 중"), MODEL))
        self.assertNotEqual(base, cache_key(self.ITEM, "other-model-2026-02-02"))
        self.assertNotEqual(base, cache_key(self.ITEM, MODEL, prompt_version="cmp-v2-other"))

    def test_prompt_version_tracks_the_anchors(self):
        self.assertTrue(COMPARE_PROMPT_VERSION.startswith("cmp-v2-"))
        self.assertIn("담배 안 피는", COMPARE_SYSTEM)  # 앵커가 프롬프트에 들어가 있다 → 버전 해시에도 들어간다
        self.assertIn("생활습관", COMPARE_SYSTEM)

    def test_pending_units_do_not_ask_the_same_item_twice(self):
        scorer, _ = scorer_with({})
        shared = traits(lifestyle="아침형")
        have = traits(lifestyle="일찍 일어나는")
        units = scorer.pending_units([(shared, have), (shared, have)])
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0], [Item("lifestyle", "아침형", "일찍 일어나는")])

    def test_prewarm_saves_everything_so_pair_scoring_never_calls_the_llm(self):
        scorer, judge = scorer_with({"lifestyle": ["동일"] * 3, "interests": ["부분 일치"] * 3})
        want = traits(lifestyle="아침형", interests="영화")
        have = traits(lifestyle="일찍 일어나는", interests="영화 감상")
        scorer.prewarm([(want, have)], workers=2, progress=lambda _msg: None)
        calls_after_prewarm = len(judge.calls)
        result = direction_score(want, have, scorer, k=1.0)
        self.assertEqual(len(judge.calls), calls_after_prewarm)
        self.assertEqual(result.n, 2)


class DirectionTests(unittest.TestCase):
    def test_missing_items_score_zero_without_calling_the_llm(self):
        scorer, judge = scorer_with({})
        result = direction_score(traits(lifestyle="아침형", interests="영화"), traits(personality="활발한"), scorer, k=1.0)
        self.assertTrue(result.defined)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(judge.calls, [])
        self.assertEqual({row["source"] for row in result.criteria}, {"MISSING"})

    def test_everything_missing_is_a_defined_zero_not_undefined(self):
        scorer, _ = scorer_with({})
        self.assertTrue(direction_score(traits(lifestyle="아침형"), None, scorer, k=1.0).defined)

    def test_score_is_sum_over_n_plus_k(self):
        scorer, _ = scorer_with({"lifestyle": ["동일"] * 3, "interests": ["부분 일치"] * 3})
        result = direction_score(
            traits(lifestyle="아침형", interests="영화", personality="다정한"),
            traits(lifestyle="일찍 일어나는", interests="영화 감상"),  # personality는 안 적음 → MISSING
            scorer,
            k=1.0,
        )
        self.assertEqual(result.n, 3)
        self.assertAlmostEqual(result.score, (1.0 + 0.5 + 0.0) / (3 + 1.0))
        by_criterion = {row["criterion"]: row for row in result.criteria}
        self.assertEqual(by_criterion["personality"]["source"], "MISSING")
        self.assertEqual(by_criterion["lifestyle"]["label"], "동일")

    def test_k_changes_the_denominator(self):
        scorer, _ = scorer_with({"lifestyle": ["동일"] * 3})
        result = direction_score(traits(lifestyle="아침형"), traits(lifestyle="일찍 일어나는"), scorer, k=0.0)
        self.assertEqual(result.score, 1.0)

    def test_undefined_without_request(self):
        scorer, _ = scorer_with({})
        for requester in (None, traits()):
            result = direction_score(requester, traits(lifestyle="아침형"), scorer, k=1.0)
            self.assertFalse(result.defined)
            self.assertIsNone(result.score)
            self.assertIsNone(result.payload()["score"])

    def test_llm_items_only_where_both_sides_wrote_something(self):
        items = llm_items(traits(lifestyle="아침형", interests="영화"), traits(lifestyle="일찍 일어나는"))
        self.assertEqual(items, [Item("lifestyle", "아침형", "일찍 일어나는")])


class ExScoreModeTests(unittest.TestCase):
    def setUp(self):
        self.scorer, _ = scorer_with({"lifestyle": ["동일"] * 6})

    def test_both_directions_average(self):
        result = ex_score(
            traits(lifestyle="아침형"), traits(lifestyle="일찍 일어나는"),
            traits(lifestyle="아침형"), traits(lifestyle="일찍 일어나는"),
            self.scorer, k=1.0,
        )
        self.assertEqual(result.weight_mode, "both")
        self.assertAlmostEqual(result.score, (0.5 + 0.5) / 2)
        self.assertEqual(result.detail["weight_mode"], "both")

    def test_one_direction_is_halved(self):
        result = ex_score(
            traits(lifestyle="아침형"), None,
            None, traits(lifestyle="일찍 일어나는"),
            self.scorer, k=1.0,
        )
        self.assertEqual(result.weight_mode, "one_direction")
        self.assertAlmostEqual(result.score, 0.5 / 2)
        self.assertTrue(result.detail["male_to_female"]["defined"])
        self.assertFalse(result.detail["female_to_male"]["defined"])

    def test_no_direction_drops_the_ex_term(self):
        result = ex_score(None, traits(lifestyle="x"), None, None, self.scorer, k=1.0)
        self.assertEqual(result.weight_mode, "none")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.detail["weight_mode"], "none")

    def test_requester_who_wrote_but_partner_did_not_is_zero_not_dropped(self):
        result = ex_score(traits(lifestyle="아침형"), None, None, None, self.scorer, k=1.0)
        self.assertEqual(result.weight_mode, "one_direction")
        self.assertEqual(result.score, 0.0)


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
