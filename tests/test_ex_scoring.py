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


def val(statement, *evidence):
    """LLM이 돌려주는 기준 하나: 재작성한 구절(statement)과 원문 근거(evidence)."""
    return {"statement": statement, "evidence": list(evidence)}


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


class PromptCoverageTests(unittest.TestCase):
    """실제 원문에서 놓쳤던 구절이 프롬프트·기준으로 다뤄지는지 지킨다. (LLM은 부르지 않는다.)"""

    # 긍정 요구와 부정 표현이 한 문장에 섞인 원문.
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
        self.assertEqual(ex_mod.COMPARE_SYSTEM.count("결혼"), 0)
        # 비교 프롬프트의 기준 설명도 같은 정의를 쓴다.
        self.assertIn("### personality (성격/가치관)", ex_mod.COMPARE_SYSTEM)
        self.assertIn("### relationship_values (연애관)", ex_mod.COMPARE_SYSTEM)

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
        """실제 원문(MIXED)에서 재작성 + 근거 인용을 한 결과가 검증을 통과한다."""
        rewritten = {
            "personality": val("적당히 진지한 사람", "진지한 사람 좋아하고 그렇다고 너무 진지한 사람 말고"),
            "appearance": val(
                "송하영 느낌이되 과하게 예쁘지 않은 사람",
                "외적인 이상형은 송하영 느낌",
                "너무 압도적으로 예쁜 사람은 상대하기 어려워요",
            ),
            "interests": val("상대의 취미는 가리지 않음(독서 제외)", "상대가 좋다면 어떤 취미라도 즐깁니다", "독서빼고요"),
        }
        self.assertEqual(traits_mod.find_violations(rewritten, self.MIXED), [])
        # 같은 결과에서 거절 표현을 그대로 남기면 걸린다.
        copied = {"personality": val("너무 진지한 사람 말고", "그렇다고 너무 진지한 사람 말고")}
        self.assertEqual(len(traits_mod.find_violations(copied, self.MIXED)), 1)

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

    def test_compare_prompt_reads_emphasis_words(self):
        system = ex_mod.COMPARE_SYSTEM
        self.assertIn('강조어("무엇보다", "특히", "매우", "아주")', system)
        self.assertIn("「무엇보다 다정한 사람」 / 가진 모습 「다정하고 배려심이 깊은 사람」 → 거의 일치", system)
        self.assertIn("「무엇보다 다정한 사람」 / 가진 모습 「다정한 편」 → 부분 일치", system)

    def test_compare_prompt_reads_no_condition_statements(self):
        system = ex_mod.COMPARE_SYSTEM
        self.assertIn("상대의 ~는 가리지 않음", system)
        self.assertIn("「상대의 취미는 가리지 않음(독서 제외)」 / 가진 모습 「영화 감상, 전시회」 → 거의 일치", system)
        self.assertIn("「상대의 취미는 가리지 않음(독서 제외)」 / 가진 모습 「독서, 글쓰기」 → 무관/충돌", system)

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

    def test_compare_prompt_reads_degree_words_and_facts(self):
        self.assertIn("정도나 한도를 나타내는 말", ex_mod.COMPARE_SYSTEM)
        self.assertIn("「적당히 진지한 사람」 / 가진 모습 「장난기 많고 가벼운」 → 무관/충돌", ex_mod.COMPARE_SYSTEM)
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
