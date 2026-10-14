"""ex 원문 평가를 소규모(남 N명 × 여 N명)로 시험 실행해 점수가 합리적인지 사람이 검토하기 위한 스크립트.

흐름
  1. 접수 참가자 중 남자 N명, 여자 N명을 registration_id 오름차순으로 앞에서부터 고른다. (기본 N=20)
  2. N×N 쌍을 채점한다. 이상형 원문과 매력 원문을 그대로 평가한다.   → 평가 결과는 메모리에만 둔다
  3. 점수표 CSV를 만들고, 1위부터 탐욕적으로 쌍을 골라 매칭표 CSV를 만든다.
  4. test_match_output/ 에 시작 시각을 붙여 저장한다.

DB에는 아무것도 쓰지 않는다. (읽기 전용 연결)
  - match_result는 읽지도 쓰지도 않는다. (start_fresh_job / commit_selected_matches를 부르지 않는다.)
  - ex_pair_score 캐시도 쓰지 않는다. 같은 입력이면 캐시가 같은 점수를 돌려주므로, 캐시를 쓰면 여러 번 돌려도
    점수가 변하지 않아 회차별 변동을 볼 수 없다. 매 회차마다 LLM이 새로 평가한다.
  - matching_state/ 도 건드리지 않는다.

사용법
  python test_match.py                 # 점검만: 대상자와 평가할 원문 쌍 수를 보여 준다. LLM 호출·쓰기 없음
  python test_match.py --execute       # 실제 실행 (OpenAI 호출)
  python test_match.py --compare       # test_match_output/ 의 점수표들을 겹쳐 회차별 변동을 요약한다. (DB·LLM 없음)
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime

import db
import main as match_main
from score_functions.ex import (
    PAIR_PROMPT_VERSION,
    RESAMPLE_EXTRA,
    SAMPLES,
    ExScoringError,
    MemoryPairStore,
    PairScorer,
    ex_score,
    final_score as combine_final_score,
)
from score_functions.ex_traits import CRITERION_KO
from score_functions.llm import get_client, resolve_model
from score_functions.mbti import mbti_score
from score_functions.tag import tag_inclusion_min, tag_score

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "test_match_output")
SCORE_PREFIX = "점수표_"
MATCH_PREFIX = "매칭표_"
PREWARM_ATTEMPTS = 5
RETRY_PAUSE_SECONDS = 30
DEFAULT_WORKERS = 4  # 분당 토큰 제한(TPM)에 걸리지 않게 조금 낮게 시작한다.

PERSON_COLS = [
    ("name", "이름"),
    ("mbti", "MBTI"),
    ("birth", "생년월일"),
    ("have", "가진매력"),
    ("want", "원하는매력"),
    ("age_pref", "나이선호"),
    ("ex_have", "기타매력(원문)"),
    ("ex_want", "기타이상형(원문)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ex 원문 평가 시험 실행 (DB에는 쓰지 않습니다).")
    parser.add_argument("--execute", action="store_true", help="실제로 실행합니다. 없으면 점검만 합니다.")
    parser.add_argument("--compare", action="store_true", help="저장된 회차별 점수표를 비교 요약합니다.")
    parser.add_argument("--round", type=int, help="대상 차수. 없으면 .env의 MATCH_ROUND.")
    parser.add_argument("--people", type=int, default=20, help="성별마다 고를 인원 (기본 20).")
    args = parser.parse_args()
    if args.people < 1:
        parser.error("--people은 1 이상이어야 합니다.")
    return args


# --------------------------------------------------------------------------- 출력 형식


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None)
    return str(value)


def person_cells(person: dict) -> list[str]:
    return [_text(person.get(key)) for key, _label in PERSON_COLS]


def _quoted(pieces: list[str]) -> str:
    return " | ".join(f"「{piece}」" for piece in pieces)


def fmt_direction(payload: dict) -> str:
    """한 방향의 기준별 평가 내용을 사람이 읽기 좋은 여러 줄 문장으로."""
    if not payload["defined"]:
        return "(요구한 기준 없음)"
    lines = []
    for item in payload["criteria"]:
        name = CRITERION_KO[item["criterion"]]
        have = _quoted(item["have_quote"]) or "(관련 내용 없음)"
        lines.append(
            f"[{name}] 요구 {_quoted(item['want_quote'])} vs 매력 {have} "
            f"→ {item['label']}({item['score']:g}) — {item['reason']}"
        )
    mark = " ⚠재샘플" if payload.get("resampled") else ""
    lines.append(
        f"= {payload['score']:.3f}  (합계 / (n={payload['n']} + k={payload['k']:g}), 호출 {payload['calls']}회{mark})"
    )
    return "\n".join(lines)


def write_csv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def unique_stamp(started: datetime) -> str:
    """연월일_시분. 같은 분에 또 돌렸으면 초를 붙여 덮어쓰지 않는다."""
    stamp = started.strftime("%Y%m%d_%H%M")
    if os.path.exists(os.path.join(OUT_DIR, f"{SCORE_PREFIX}{stamp}.csv")):
        stamp = started.strftime("%Y%m%d_%H%M%S")
    return stamp


# --------------------------------------------------------------------------- 단계별 작업


def select_people(students, male_all, female_all, count):
    male_ids = male_all[:count]
    female_ids = female_all[:count]
    if len(male_ids) < count or len(female_ids) < count:
        print(f"주의: 남 {len(male_ids)}명, 여 {len(female_ids)}명뿐입니다. (요청 {count}명)")
    return male_ids, female_ids


def print_people(students, male_ids, female_ids) -> None:
    for title, ids in (("남", male_ids), ("여", female_ids)):
        print(f"\n[{title} {len(ids)}명]")
        for number, rid in enumerate(ids, start=1):
            person = students[rid]
            have = "O" if (person.get("ex_have") or "").strip() else "-"
            want = "O" if (person.get("ex_want") or "").strip() else "-"
            print(f"  {number:>2}. {rid}  {person.get('name') or ''}  (기타매력 {have} / 기타이상형 {want})")


def build_directed(students, male_ids, female_ids):
    directed = []
    for mid in male_ids:
        for wid in female_ids:
            m, w = students[mid], students[wid]
            if match_main.pair_age_ok(m, w):
                directed.append((m.get("ex_want"), w.get("ex_have")))
                directed.append((w.get("ex_want"), m.get("ex_have")))
    return directed


def _no_llm(_want: str, _have: str):
    raise RuntimeError("점검 모드에서는 LLM을 부르지 않습니다.")


def prewarm_with_retry(scorer: PairScorer, directed, workers: int) -> None:
    """일부 호출이 실패해도 메모리에 쌓인 평가는 남는다. 남은 원문 쌍만 다시 시도한다.

    실패의 대부분은 분당 토큰 제한(429)이다. 재시도할 때마다 동시 호출 수를 절반으로 줄이고 잠시 쉰다.
    """
    for attempt in range(1, PREWARM_ATTEMPTS + 1):
        try:
            scorer.prewarm(directed, workers=workers)
            return
        except ExScoringError as err:
            print(f"\n채점 일부 실패 ({attempt}/{PREWARM_ATTEMPTS}회차): {str(err).splitlines()[0]}")
            if attempt == PREWARM_ATTEMPTS:
                raise
            workers = max(1, workers // 2)
            print(f"{RETRY_PAUSE_SECONDS}초 쉬고 동시 {workers}개로 남은 원문 쌍만 다시 평가합니다.")
            time.sleep(RETRY_PAUSE_SECONDS)


def score_all_pairs(students, male_ids, female_ids, scorer) -> list[dict]:
    rows = []
    for mid in male_ids:
        for wid in female_ids:
            m, w = students[mid], students[wid]
            row = {"male_id": mid, "female_id": wid}
            if not match_main.pair_age_ok(m, w):
                rows.append({**row, "status": "skipped_age"})
                continue
            tag_args = (m["want"] or [], m["have"] or [], w["want"] or [], w["have"] or [])
            ms = mbti_score(m["mbti"], w["mbti"])
            ts = tag_score(*tag_args)
            ex = ex_score(m.get("ex_want"), m.get("ex_have"), w.get("ex_want"), w.get("ex_have"), scorer)
            final = combine_final_score(ms, ts, ex.score, ex.weight_mode, match_main.W_MBTI, match_main.W_TAG, match_main.W_EX)
            rows.append(
                {
                    **row,
                    "status": "scored",
                    "mbti_score": ms,
                    "tag_score": ts,
                    "ex_score": ex.score,
                    "ex_detail": ex.detail,
                    "final_score": final,
                    "tag_min_incl": tag_inclusion_min(*tag_args),
                }
            )
    return rows


def write_score_table(path, stamp, rows, students, selected_pairs) -> None:
    scored = [r for r in rows if r["status"] == "scored"]
    rank_of = {(r["male_id"], r["female_id"]): n for n, r in enumerate(sorted(scored, key=match_main._tie_break_key), start=1)}
    header = [
        "회차", "전체순위", "매칭선정",
        "남_ID", "남_이름", "남_MBTI", "여_ID", "여_이름", "여_MBTI",
        "상태", "최종점수", "MBTI점수", "태그점수", "기타매력점수", "기타매력_방식", "남→여", "여→남",
        "남→여 항목 상세", "여→남 항목 상세",
    ]
    out = []
    for r in rows:
        m, w = students[r["male_id"]], students[r["female_id"]]
        base = [stamp]
        ids = [r["male_id"], m.get("name") or "", m.get("mbti") or "", r["female_id"], w.get("name") or "", w.get("mbti") or ""]
        if r["status"] != "scored":
            out.append(base + ["", ""] + ids + ["연령 조건 불일치"] + [""] * 9)
            continue
        detail = r["ex_detail"]
        m2f, f2m = detail["male_to_female"], detail["female_to_male"]
        pair = (r["male_id"], r["female_id"])
        out.append(
            base
            + [rank_of[pair], "O" if pair in selected_pairs else ""]
            + ids
            + [
                "채점",
                f"{r['final_score']:.4f}",
                f"{r['mbti_score']:.4f}",
                f"{r['tag_score']:.4f}",
                f"{r['ex_score']:.4f}",
                detail["weight_mode"],
                "" if m2f["score"] is None else f"{m2f['score']:.4f}",
                "" if f2m["score"] is None else f"{f2m['score']:.4f}",
                fmt_direction(m2f),
                fmt_direction(f2m),
            ]
        )
    out.sort(key=lambda cells: (cells[1] == "", cells[1] if cells[1] != "" else 0))
    write_csv(path, header, out)


def write_match_table(path, stamp, matches, students, male_ids, female_ids) -> None:
    header = ["회차", "순위", "최종점수", "MBTI점수", "태그점수", "기타매력점수", "기타매력_방식"]
    header += [f"남_{label}" for _k, label in PERSON_COLS] + [f"여_{label}" for _k, label in PERSON_COLS]
    header += ["남→여 항목 상세", "여→남 항목 상세"]
    out = []
    for rank, item in enumerate(matches, start=1):
        m, w = students[item["male_id"]], students[item["female_id"]]
        detail = item["ex_detail"]
        out.append(
            [
                stamp, rank, f"{item['final_score']:.4f}", f"{item['mbti_score']:.4f}",
                f"{item['tag_score']:.4f}", f"{item['ex_score']:.4f}", detail["weight_mode"],
                *person_cells(m), *person_cells(w),
                fmt_direction(detail["male_to_female"]), fmt_direction(detail["female_to_male"]),
            ]
        )
    matched_m = {item["male_id"] for item in matches}
    matched_w = {item["female_id"] for item in matches}
    blank_person = [""] * len(PERSON_COLS)
    for mid in male_ids:
        if mid not in matched_m:
            out.append([stamp, "미매칭", "", "", "", "", "", *person_cells(students[mid]), *blank_person, "", ""])
    for wid in female_ids:
        if wid not in matched_w:
            out.append([stamp, "미매칭", "", "", "", "", "", *blank_person, *person_cells(students[wid]), "", ""])
    write_csv(path, header, out)


# --------------------------------------------------------------------------- 회차 비교


def _read_scores(path: str) -> dict[tuple[str, str], dict]:
    result = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["상태"] != "채점":
                continue
            result[(row["남_ID"], row["여_ID"])] = {
                "final": float(row["최종점수"]),
                "ex": float(row["기타매력점수"]),
                "name": f"{row['남_이름']}×{row['여_이름']}",
            }
    return result


def _read_match_pairs(path: str) -> set[tuple[str, str]]:
    pairs = set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["순위"].isdigit():
                pairs.add((row["남_이름"], row["여_이름"]))
    return pairs


def compare_runs() -> int:
    score_files = sorted(glob.glob(os.path.join(OUT_DIR, f"{SCORE_PREFIX}*.csv")))
    if len(score_files) < 2:
        print(f"비교하려면 {OUT_DIR} 에 점수표가 2개 이상 필요합니다. (지금 {len(score_files)}개)")
        return 1
    runs = [_read_scores(path) for path in score_files]
    print(f"== 회차 비교: {len(runs)}회 ==")
    for path in score_files:
        print("  " + os.path.basename(path))

    common = set(runs[0])
    for run in runs[1:]:
        common &= set(run)
    spreads = []
    for key in common:
        finals = [run[key]["final"] for run in runs]
        exs = [run[key]["ex"] for run in runs]
        spreads.append((max(finals) - min(finals), max(exs) - min(exs), key))
    final_ranges = [s[0] for s in spreads]
    ex_ranges = [s[1] for s in spreads]
    print(f"\n모든 회차에서 채점된 쌍: {len(common)}개")
    print(f"최종점수 변동폭(max-min): 평균 {statistics.mean(final_ranges):.4f}, 최대 {max(final_ranges):.4f}")
    print(f"기타매력점수 변동폭(max-min): 평균 {statistics.mean(ex_ranges):.4f}, 최대 {max(ex_ranges):.4f}")
    print(f"기타매력점수가 회차마다 똑같은 쌍: {sum(1 for r in ex_ranges if r == 0)}개 / {len(common)}개")

    print("\n변동이 큰 쌍 상위 10 (최종점수 변동폭 / 기타매력 변동폭)")
    for final_range, ex_range, key in sorted(spreads, reverse=True)[:10]:
        print(f"  {runs[0][key]['name']}: {final_range:.4f} / {ex_range:.4f}")

    match_files = sorted(glob.glob(os.path.join(OUT_DIR, f"{MATCH_PREFIX}*.csv")))
    if len(match_files) >= 2:
        sets = [_read_match_pairs(path) for path in match_files]
        counts = Counter(pair for pairs in sets for pair in pairs)
        in_all = [pair for pair, n in counts.items() if n == len(sets)]
        print(f"\n매칭표 {len(sets)}개: 전 회차 공통 쌍 {len(in_all)}개 / 회차별 쌍 수 {[len(s) for s in sets]}")
        for pair, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            if n < len(sets):
                print(f"  {pair[0]} × {pair[1]}: {n}/{len(sets)}회에서 선정")
    return 0


# --------------------------------------------------------------------------- 실행


def main() -> int:
    args = parse_args()
    if args.compare:
        return compare_runs()

    started = datetime.now()
    if args.round is not None:
        match_main.MATCH_ROUND = args.round  # fetch_students가 이 값으로 접수를 읽는다.
    mode = "실행" if args.execute else "점검만 (LLM 호출·쓰기 없음)"
    print(f"== ex 원문 평가 시험 [{mode}] ==")
    print(f"대상 DB: Aurora {db.aurora_host()}  /  차수 {match_main.MATCH_ROUND}")
    print(f"평가 프롬프트 {PAIR_PROMPT_VERSION}")

    model = None
    if args.execute:
        model = resolve_model()
        get_client()  # OPENAI_API_KEY 확인
        print(f"모델: {model}")

    read_conn = db.connect(read_only=True)  # 서버가 쓰기를 거부한다. match_result 등에 실수로도 쓸 수 없다.
    try:
        try:
            students, male_all, female_all = match_main.fetch_students(read_conn)
        except Exception:  # noqa: BLE001 — fetch_students는 접수가 없으면 일반 Exception을 던진다.
            read_conn.rollback()
            with read_conn.cursor() as cur:
                cur.execute("SELECT round, count(*) FROM registration GROUP BY round ORDER BY round")
                available = cur.fetchall()
            read_conn.commit()
            print(
                f"\n오류: {match_main.MATCH_ROUND}차 접수가 없습니다. Aurora의 차수별 접수: "
                + ", ".join(f"{r}차 {n}건" for r, n in available)
                + "\n--round 로 차수를 지정해 주세요."
            )
            return 1
        read_conn.commit()
    finally:
        read_conn.close()

    male_ids, female_ids = select_people(students, male_all, female_all, args.people)
    people = {rid: students[rid] for rid in male_ids + female_ids}
    print(f"접수 {len(students)}명 중 남 {len(male_all)}명 / 여 {len(female_all)}명 → 남 {len(male_ids)}명, 여 {len(female_ids)}명 선택")
    print_people(students, male_ids, female_ids)
    directed = build_directed(people, male_ids, female_ids)

    if not args.execute:
        pending = PairScorer(MemoryPairStore(), "점검", judge=_no_llm).pending(directed)
        print(
            f"\n평가할 원문 쌍 {len(pending)}개 → LLM 호출 최소 {len(pending) * SAMPLES}회 "
            f"(의견이 갈리면 쌍마다 +{RESAMPLE_EXTRA}회)"
        )
        print("\n점검만 했습니다. 아무것도 쓰지 않았습니다. 실제로 실행하려면 --execute를 붙이세요.")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = unique_stamp(started)

    # 채점 (평가 결과는 메모리에만 둔다)
    print("\n-- 쌍 채점 --")
    scorer = PairScorer(MemoryPairStore(), model)
    prewarm_with_retry(scorer, directed, workers=int(os.getenv("EX_WORKERS", str(DEFAULT_WORKERS))))
    rows = score_all_pairs(people, male_ids, female_ids, scorer)

    # 탐욕적 선정 (CSV로만 낸다)
    scored = [r for r in rows if r["status"] == "scored"]
    matches = match_main.pick_unique_matches(scored)
    selected_pairs = {(m["male_id"], m["female_id"]) for m in matches}

    score_path = os.path.join(OUT_DIR, f"{SCORE_PREFIX}{stamp}.csv")
    match_path = os.path.join(OUT_DIR, f"{MATCH_PREFIX}{stamp}.csv")
    write_score_table(score_path, stamp, rows, people, selected_pairs)
    write_match_table(match_path, stamp, matches, people, male_ids, female_ids)

    modes = Counter(r["ex_detail"]["weight_mode"] for r in scored)
    print(f"\n== 완료 ({stamp}) ==")
    print(f"채점 {len(scored)}쌍, 연령 조건 불일치 {len(rows) - len(scored)}쌍, 기타매력 방식 {dict(modes)}")
    print(f"매칭 {len(matches)}쌍 (DB에는 올리지 않았습니다)")
    for rank, item in enumerate(matches[:10], start=1):
        m, w = people[item["male_id"]], people[item["female_id"]]
        print(
            f"  {rank:>2}. {m.get('name')} × {w.get('name')}  최종 {item['final_score']:.3f} "
            f"(MBTI {item['mbti_score']:.2f}, 태그 {item['tag_score']:.2f}, 기타 {item['ex_score']:.2f})"
        )
    print(f"\n저장: {score_path}\n      {match_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
