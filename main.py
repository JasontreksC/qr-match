import csv
import hashlib
import json
import os
from typing import Iterator

from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from db import connect

load_dotenv()

MATCH_ROUND = int(os.getenv('MATCH_ROUND', 1))
if MATCH_ROUND not in (1, 2):
    raise ValueError("MATCH_ROUND는 1 또는 2여야 합니다.")

# 최종 점수 = (W_MBTI·MBTI + W_TAG·태그 + W_EX·ex) / (W_MBTI + W_TAG + W_EX)
# 이전 코드의 0.1 : 0.6 : 0.3과 같은 비율이다. 안내 페이지(matching/page.tsx)의 "1 : 6 : 3"과도 같다.
# ex 점수가 정의되지 않는 쌍(weight_mode == "none")은 W_EX를 빼고 다시 계산한다.
W_MBTI = 1.0
W_TAG = 6.0
W_EX = 3.0

# 최종 점수가 같을 때의 마지막 순서를 정하는 고정 시드. (실행마다 같은 결과를 내기 위함)
TIE_BREAK_SEED = "qrious-ex-v2"
# 추출이 안 된 접수가 있어도 그 접수의 ex 텍스트를 "없음"으로 보고 진행하려면 1.
ALLOW_UNEXTRACTED = os.getenv("EX_ALLOW_UNEXTRACTED") == "1"

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(ROOT, "matching_state")
PAIRS_CSV = os.path.join(STATE_DIR, "pairs.csv")
SCORES_CSV = os.path.join(STATE_DIR, "scores.csv")
SELECTED_CSV = os.path.join(STATE_DIR, "selected.csv")
PROGRESS_FILE = os.path.join(STATE_DIR, "current_row.txt")
PHASE_FILE = os.path.join(STATE_DIR, "phase.txt")
TOTAL_FILE = os.path.join(STATE_DIR, "total_rows.txt")
OUTPUT_CSV = os.path.join(ROOT, "output.csv")

SCORE_FIELDS = [
    "row_num",
    "male_id",
    "female_id",
    "status",
    "mbti_score",
    "tag_score",
    "ex_score",
    "ex_detail",
    "final_score",
    "tag_min_incl",
]
SELECTED_FIELDS = [
    "순위",
    "male_id",
    "female_id",
    "mbti_score",
    "tag_score",
    "ex_score",
    "ex_detail",
    "final_score",
]
PERSON_FIELDS = [
    ("registration_id", "ID"),
    ("name", "이름"),
    ("gender", "성별"),
    ("birth", "생년월일"),
    ("mbti", "MBTI"),
    ("phone", "전화번호"),
    ("have", "가진매력"),
    ("want", "원하는매력"),
    ("age_pref", "나이선호"),
    ("ex_have", "기타매력"),
    ("ex_want", "기타이상형"),
]


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None)
    return str(value)


def _gender_label(gender) -> str:
    if gender is True:
        return "여"
    if gender is False:
        return "남"
    return _as_text(gender)


def _person_cells(person: dict) -> list[str]:
    cells = []
    for key, _label in PERSON_FIELDS:
        value = person.get(key)
        if key == "gender":
            cells.append(_gender_label(value))
        else:
            cells.append(_as_text(value))
    return cells


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_int_file(path: str, default: int = 0) -> int:
    if not os.path.exists(path):
        return default
    raw = open(path, encoding="utf-8").read().strip()
    return int(raw) if raw else default


def _read_text_file(path: str, default: str = "") -> str:
    if not os.path.exists(path):
        return default
    return open(path, encoding="utf-8").read().strip() or default


def _fsync_written(f) -> None:
    f.flush()
    os.fsync(f.fileno())


def count_csv_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def last_score_row_num(path: str = SCORES_CSV) -> int:
    if not os.path.exists(path):
        return 0
    last = 0
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("row_num") or "").strip()
            if not raw:
                continue
            last = int(raw)
    return last


def resume_row(progress_path: str = PROGRESS_FILE, scores_path: str = SCORES_CSV) -> int:
    current = _read_int_file(progress_path, 1)
    last_scored = last_score_row_num(scores_path)
    return max(current, last_scored + 1)


def write_pairs_csv(male_ids: list[str], female_ids: list[str], path: str = PAIRS_CSV) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    total = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_num", "male_id", "female_id"])
        for mid in male_ids:
            for wid in female_ids:
                total += 1
                writer.writerow([total, mid, wid])
        _fsync_written(f)
    _atomic_write(os.path.join(os.path.dirname(os.path.abspath(path)), "total_rows.txt"), str(total))
    return total


def iter_pairs_from(start_row: int, path: str = PAIRS_CSV) -> Iterator[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["row_num"])
            if n < start_row:
                continue
            yield row


def _parse_ex_detail(raw: str | None):
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _dump_ex_detail(detail) -> str:
    if not detail:
        return ""
    return json.dumps(detail, ensure_ascii=False, separators=(",", ":"))


def append_score_row(row: dict, path: str = SCORES_CSV) -> None:
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        _fsync_written(f)


def load_scored_rows(path: str = SCORES_CSV) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "scored":
                continue
            rows.append(
                {
                    "male_id": row["male_id"],
                    "female_id": row["female_id"],
                    "mbti_score": float(row["mbti_score"]),
                    "tag_score": float(row["tag_score"]),
                    "ex_score": float(row["ex_score"]),
                    "ex_detail": _parse_ex_detail(row.get("ex_detail")),
                    "final_score": float(row["final_score"]),
                    "tag_min_incl": float(row.get("tag_min_incl") or 0.0),
                }
            )
    return rows


def ensure_scores_header_current(path: str = SCORES_CSV) -> None:
    """이전 방식으로 쌓은 점수 CSV에 새 방식 점수를 이어 붙이지 않게 막는다."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f), [])
    if header != SCORE_FIELDS:
        raise RuntimeError(
            f"{path}가 이전 채점 방식으로 만든 파일입니다. 이어서 계산하면 점수 방식이 섞입니다. "
            "matching_state 폴더를 비운 뒤 처음부터 다시 실행해 주세요."
        )


def _tie_break_key(item: dict) -> tuple:
    """최종 점수가 같을 때의 순서: 양방향 포함 계수의 최솟값 → ex 점수 → MBTI → 고정 시드 해시.

    값은 CSV에 저장된 자릿수(6자리)로 맞춰 비교한다.
    """
    digest = hashlib.sha256(
        f"{TIE_BREAK_SEED}|{item['male_id']}|{item['female_id']}".encode("utf-8")
    ).hexdigest()
    return (
        -round(item["final_score"], 6),
        -round(item.get("tag_min_incl", 0.0), 6),
        -round(item["ex_score"], 6),
        -round(item["mbti_score"], 6),
        digest,
    )


def pick_unique_matches(scored_rows: list[dict]) -> list[dict]:
    ranked = sorted(scored_rows, key=_tie_break_key)
    taken_m: set[str] = set()
    taken_w: set[str] = set()
    selected = []
    for item in ranked:
        if item["male_id"] in taken_m or item["female_id"] in taken_w:
            continue
        taken_m.add(item["male_id"])
        taken_w.add(item["female_id"])
        selected.append(item)
    return selected


def write_selected_csv(matches: list[dict], path: str = SELECTED_CSV) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SELECTED_FIELDS)
        writer.writeheader()
        for rank, item in enumerate(matches, start=1):
            writer.writerow(
                {
                    "순위": rank,
                    "male_id": item["male_id"],
                    "female_id": item["female_id"],
                    "mbti_score": f"{item['mbti_score']:.6f}",
                    "tag_score": f"{item['tag_score']:.6f}",
                    "ex_score": f"{item['ex_score']:.6f}",
                    "ex_detail": _dump_ex_detail(item.get("ex_detail")),
                    "final_score": f"{item['final_score']:.6f}",
                }
            )
        _fsync_written(f)


def load_selected_csv(path: str = SELECTED_CSV) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "순위": int(row["순위"]),
                    "male_id": row["male_id"],
                    "female_id": row["female_id"],
                    "mbti_score": float(row["mbti_score"]),
                    "tag_score": float(row["tag_score"]),
                    "ex_score": float(row["ex_score"]),
                    "ex_detail": _parse_ex_detail(row.get("ex_detail")),
                    "final_score": float(row["final_score"]),
                }
            )
    return rows


def write_output_csv(matches: list[dict], students: dict, path: str = OUTPUT_CSV) -> str:
    headers = [
        "순위",
        "최종점수",
        "MBTI점수",
        "태그점수",
        "기타매력점수",
    ]
    headers += [f"남_{label}" for _key, label in PERSON_FIELDS]
    headers += [f"여_{label}" for _key, label in PERSON_FIELDS]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rank, item in enumerate(matches, start=1):
            m = students[item["male_id"]]
            w = students[item["female_id"]]
            writer.writerow(
                [
                    rank,
                    f"{item['final_score']:.4f}",
                    f"{item['mbti_score']:.4f}",
                    f"{item['tag_score']:.4f}",
                    f"{item['ex_score']:.4f}",
                    *_person_cells(m),
                    *_person_cells(w),
                ]
            )
        _fsync_written(f)
    return path


def fetch_students(conn) -> tuple[dict, list[str], list[str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
    SELECT jsonb_build_object(
    'registration_id', r.registration_id,
    'student_id', s.student_id,
    'name', s.name,
    'gender', s.gender,
    'birth', s.birth,
    'mbti', r.mbti,
    'phone', s.phone,
    'want', COALESCE(
        (
        SELECT jsonb_agg(c.name ORDER BY c.name)
        FROM want w
        JOIN charm c ON c.charm_id = w.charm_id
        WHERE w.registration_id = r.registration_id
        ),
        '[]'::jsonb
    ),
    'have', COALESCE(
        (
        SELECT jsonb_agg(c.name ORDER BY c.name)
        FROM have h
        JOIN charm c ON c.charm_id = h.charm_id
        WHERE h.registration_id = r.registration_id
        ),
        '[]'::jsonb
    ),
    'age_pref', COALESCE(
        (
        SELECT jsonb_agg(ap.name ORDER BY ap.sort_order)
        FROM prefer_age pa
        JOIN age_pref ap ON ap.age_pref_id = pa.age_pref_id
        WHERE pa.registration_id = r.registration_id
        ),
        '[]'::jsonb
    ),
    'ex_want', ew.charm,
    'ex_have', eh.charm
    ) AS student
    FROM registration r
    JOIN student s ON s.student_id = r.student_id
    LEFT JOIN ex_want ew ON ew.registration_id = r.registration_id
    LEFT JOIN ex_have eh ON eh.registration_id = r.registration_id
    WHERE r.round = %s
    ORDER BY r.registration_id;
            """,
            (MATCH_ROUND,),
        )
        rows = [row[0] for row in cur.fetchall()]
    if not rows:
        raise Exception("쿼리 결과가 없음!")
    students = {r["registration_id"]: r for r in rows}
    male_ids = [r["registration_id"] for r in rows if r["gender"] is False]
    female_ids = [r["registration_id"] for r in rows if r["gender"] is True]
    return students, male_ids, female_ids


def truncate_match_result(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM match_result WHERE round = %s", (MATCH_ROUND,))
    conn.commit()


def already_committed_ranks(conn) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT rank FROM match_result WHERE round = %s", (MATCH_ROUND,))
        return {row[0] for row in cur.fetchall()}


def insert_match_result(conn, item: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO match_result (
                rank, male_id, female_id,
                mbti_score, tag_score, ex_score, ex_detail, final_score, round
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                item["순위"],
                item["male_id"],
                item["female_id"],
                item["mbti_score"],
                item["tag_score"],
                item["ex_score"],
                None if item.get("ex_detail") is None else Jsonb(item["ex_detail"]),
                item["final_score"],
                MATCH_ROUND,
            ),
        )
    conn.commit()


def start_fresh_job(male_ids: list[str], female_ids: list[str], conn) -> int:
    os.makedirs(STATE_DIR, exist_ok=True)
    truncate_match_result(conn)
    for path in (SCORES_CSV, SELECTED_CSV, PROGRESS_FILE, PHASE_FILE, TOTAL_FILE, OUTPUT_CSV, PAIRS_CSV):
        if os.path.exists(path):
            os.remove(path)
    total = write_pairs_csv(male_ids, female_ids)
    _atomic_write(PROGRESS_FILE, "1")
    _atomic_write(PHASE_FILE, "scoring")
    print(f"새 매칭 시작: {MATCH_ROUND}차, 곱집합 {total}쌍 저장 → {PAIRS_CSV}")
    print(f"match_result에서 {MATCH_ROUND}차 행을 삭제했습니다.")
    return total


def pair_age_ok(m: dict, w: dict) -> bool:
    from score_functions.age import birth_accepts

    return birth_accepts(m.get("birth"), w.get("birth"), m["age_pref"] or []) and birth_accepts(
        w.get("birth"), m.get("birth"), w["age_pref"] or []
    )


class ExContext:
    """ex 점수 계산에 필요한 것: 접수별 추출 결과와 항목 채점기."""

    def __init__(self, traits: dict, scorer) -> None:
        self.traits = traits  # registration_id -> {"want": dict | None, "have": dict | None}
        self.scorer = scorer


def prepare_ex_scoring(conn, students: dict, male_ids: list[str], female_ids: list[str]) -> ExContext:
    """매칭을 시작하기 전에 ex 점수에 필요한 것을 모두 준비한다.

    1. ex_trait(추출 결과)가 모든 ex 텍스트에 대해 유효한지 확인한다.
    2. 연령 조건을 통과하는 모든 쌍의 항목을 미리 채점해 ex_item_score에 저장한다.
       이후 쌍별 채점은 저장된 값만 읽는다. (같은 입력이면 다시 돌려도 점수가 바뀌지 않는다.)
    """
    from score_functions.ex import COMPARE_PROMPT_VERSION, AuroraItemStore, ItemScorer
    from score_functions.ex_traits import load_trait_book, trait_problems, usable_traits
    from score_functions.llm import resolve_model

    book = load_trait_book(conn, list(students))
    problems = trait_problems(students, book)
    if problems:
        shown = "\n".join(f"  - {line}" for line in problems[:20])
        more = f"\n  ... 외 {len(problems) - 20}건" if len(problems) > 20 else ""
        message = (
            f"유효한 추출 결과(ex_trait)가 없는 ex 텍스트가 {len(problems)}건 있습니다.\n{shown}{more}\n"
            "python extract_ex_traits.py --execute 로 추출한 뒤 다시 실행해 주세요."
        )
        if not ALLOW_UNEXTRACTED:
            raise RuntimeError(message + "\n(무시하고 진행하려면 EX_ALLOW_UNEXTRACTED=1. 그 접수의 ex 텍스트는 없음으로 처리됩니다.)")
        print("경고: " + message + "\nEX_ALLOW_UNEXTRACTED=1이므로 그 접수의 ex 텍스트는 없음으로 처리합니다.")

    traits = {
        registration_id: {
            side: usable_traits(student.get(f"ex_{side}"), book.get(registration_id, side))
            for side in ("want", "have")
        }
        for registration_id, student in students.items()
    }

    model = resolve_model()
    store = AuroraItemStore(conn, model, COMPARE_PROMPT_VERSION)
    cached = store.load()
    print(f"항목 점수 캐시: {cached}건 (모델 {model}, 프롬프트 {COMPARE_PROMPT_VERSION})")
    scorer = ItemScorer(store, model)

    directed = []
    for mid in male_ids:
        for wid in female_ids:
            if pair_age_ok(students[mid], students[wid]):
                directed.append((traits[mid]["want"], traits[wid]["have"]))
                directed.append((traits[wid]["want"], traits[mid]["have"]))
    scorer.prewarm(directed, workers=int(os.getenv("EX_WORKERS", "6")))
    return ExContext(traits, scorer)


def score_remaining_pairs(students: dict, start_row: int, total: int, ex_ctx: ExContext) -> None:
    from score_functions.mbti import mbti_score
    from score_functions.tag import tag_inclusion_min, tag_score
    from score_functions.ex import ex_score, final_score as combine_final_score

    for pair in iter_pairs_from(start_row):
        n = int(pair["row_num"])
        mid = pair["male_id"]
        wid = pair["female_id"]
        _atomic_write(PROGRESS_FILE, str(n))
        print(f"[{n}/{total}] 매칭 중: {mid} × {wid}")

        m = students[mid]
        w = students[wid]
        if not pair_age_ok(m, w):
            append_score_row(
                {
                    "row_num": n,
                    "male_id": mid,
                    "female_id": wid,
                    "status": "skipped_age",
                    "mbti_score": "",
                    "tag_score": "",
                    "ex_score": "",
                    "ex_detail": "",
                    "final_score": "",
                    "tag_min_incl": "",
                }
            )
            print(f"[{n}/{total}] 연령 조건 불일치 → 건너뜀")
            continue

        ms = mbti_score(m["mbti"], w["mbti"])
        tag_args = (m["want"] or [], m["have"] or [], w["want"] or [], w["have"] or [])
        ts = tag_score(*tag_args)
        m_traits = ex_ctx.traits[mid]
        w_traits = ex_ctx.traits[wid]
        ex = ex_score(m_traits["want"], m_traits["have"], w_traits["want"], w_traits["have"], ex_ctx.scorer)
        final_score = combine_final_score(ms, ts, ex.score, ex.weight_mode, W_MBTI, W_TAG, W_EX)
        append_score_row(
            {
                "row_num": n,
                "male_id": mid,
                "female_id": wid,
                "status": "scored",
                "mbti_score": f"{ms:.6f}",
                "tag_score": f"{ts:.6f}",
                "ex_score": f"{ex.score:.6f}",
                "ex_detail": _dump_ex_detail(ex.detail),
                "final_score": f"{final_score:.6f}",
                "tag_min_incl": f"{tag_inclusion_min(*tag_args):.6f}",
            }
        )
        print(
            f"[{n}/{total}] FINAL {final_score:.2f} "
            f"(mbti={ms:.2f}, tag={ts:.2f}, ex={ex.score:.2f}, {ex.weight_mode})"
        )


def commit_selected_matches(conn, matches: list[dict], students: dict) -> None:
    _atomic_write(PHASE_FILE, "committing")
    write_selected_csv(matches)
    write_output_csv(matches, students)
    done_ranks = already_committed_ranks(conn)
    for item in matches:
        rank = item["순위"]
        if rank in done_ranks:
            print(f"순위 {rank} 이미 DB에 있음 → 건너뜀")
            continue
        insert_match_result(conn, item)
        print(
            f"순위 {rank} 커밋: {item['male_id']} × {item['female_id']} "
            f"({item['final_score']:.4f})"
        )
    _atomic_write(PHASE_FILE, "done")


def main() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    with connect() as conn:
        students, male_ids, female_ids = fetch_students(conn)
        expected_total = len(male_ids) * len(female_ids)
        phase = _read_text_file(PHASE_FILE)
        stored_total = (
            _read_int_file(TOTAL_FILE) if os.path.exists(TOTAL_FILE) else count_csv_rows(PAIRS_CSV)
        )
        pairs_ok = os.path.exists(PAIRS_CSV) and stored_total == expected_total
        fresh = phase in {"", "done"} or not pairs_ok

        # 추출 결과 확인과 항목 미리 채점은 match_result를 지우기 전에 끝낸다.
        # 여기서 실패해도 기존 결과는 그대로 남는다.
        ex_ctx = None
        if fresh or phase == "scoring":
            if not fresh:
                ensure_scores_header_current()
            ex_ctx = prepare_ex_scoring(conn, students, male_ids, female_ids)

        if fresh:
            total = start_fresh_job(male_ids, female_ids, conn)
            start = 1
            phase = "scoring"
        else:
            total = stored_total
            start = resume_row()
            print(f"이전 작업 재개: phase={phase}, {start}/{total}행부터")

        if phase == "scoring" and start <= total:
            score_remaining_pairs(students, start, total, ex_ctx)

        if last_score_row_num() < total:
            raise RuntimeError("점수 CSV가 곱집합보다 짧습니다. 다시 실행하면 이어서 계산합니다.")

        if os.path.exists(SELECTED_CSV) and _read_text_file(PHASE_FILE) in {
            "committing",
            "done",
        }:
            matches = load_selected_csv()
        else:
            matches = pick_unique_matches(load_scored_rows())
            for rank, item in enumerate(matches, start=1):
                item["순위"] = rank

        commit_selected_matches(conn, matches, students)

        matched_ids = {item["male_id"] for item in matches} | {item["female_id"] for item in matches}
        unmatched = [
            students[sid]
            for sid in list(male_ids) + list(female_ids)
            if sid not in matched_ids
        ]
        print(f"최종 매칭 {len(matches)}쌍 → {OUTPUT_CSV} / match_result")
        if unmatched:
            names = ", ".join(f"{s['name']}({s['registration_id']})" for s in unmatched)
            print(f"미매칭 {len(unmatched)}명: {names}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
