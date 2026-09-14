import csv
import json
import os
from typing import Iterator

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

load_dotenv()

MATCH_ROUND = int(os.getenv('MATCH_ROUND', 1))
if MATCH_ROUND not in (1, 2):
    raise ValueError("MATCH_ROUND는 1 또는 2여야 합니다.")

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
    ("student_id", "ID"),
    ("name", "이름"),
    ("gender", "성별"),
    ("age", "나이"),
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
                }
            )
    return rows


def pick_unique_matches(scored_rows: list[dict]) -> list[dict]:
    ranked = sorted(
        scored_rows,
        key=lambda item: (
            -item["final_score"],
            item["male_id"],
            item["female_id"],
        ),
    )
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
    'student_id', s.student_id,
    'name', s.name,
    'gender', s.gender,
    'age', s.age,
    'mbti', s.mbti,
    'phone', s.phone,
    'want', COALESCE(
        (
        SELECT jsonb_agg(c.name ORDER BY c.name)
        FROM want w
        JOIN charm c ON c.charm_id = w.charm_id
        WHERE w.student_id = s.student_id
        ),
        '[]'::jsonb
    ),
    'have', COALESCE(
        (
        SELECT jsonb_agg(c.name ORDER BY c.name)
        FROM have h
        JOIN charm c ON c.charm_id = h.charm_id
        WHERE h.student_id = s.student_id
        ),
        '[]'::jsonb
    ),
    'age_pref', COALESCE(
        (
        SELECT jsonb_agg(ap.name ORDER BY ap.sort_order)
        FROM prefer_age pa
        JOIN age_pref ap ON ap.age_pref_id = pa.age_pref_id
        WHERE pa.student_id = s.student_id
        ),
        '[]'::jsonb
    ),
    'ex_want', ew.charm,
    'ex_have', eh.charm
    ) AS student
    FROM student s
    LEFT JOIN ex_want ew ON ew.student_id = s.student_id
    LEFT JOIN ex_have eh ON eh.student_id = s.student_id
    WHERE s.round = %s
    ORDER BY s.student_id;
            """,
            (MATCH_ROUND,),
        )
        rows = [row[0] for row in cur.fetchall()]
    if not rows:
        raise Exception("쿼리 결과가 없음!")
    students = {r["student_id"]: r for r in rows}
    male_ids = [r["student_id"] for r in rows if r["gender"] is False]
    female_ids = [r["student_id"] for r in rows if r["gender"] is True]
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


def score_remaining_pairs(students: dict, start_row: int, total: int) -> None:
    from score_functions.age import age_accepts
    from score_functions.mbti import mbti_score
    from score_functions.tag import tag_score
    from score_functions.ex import ex_score

    for pair in iter_pairs_from(start_row):
        n = int(pair["row_num"])
        mid = pair["male_id"]
        wid = pair["female_id"]
        _atomic_write(PROGRESS_FILE, str(n))
        print(f"[{n}/{total}] 매칭 중: {mid} × {wid}")

        m = students[mid]
        w = students[wid]
        age_ok = age_accepts(m["age"], w["age"], m["age_pref"] or []) and age_accepts(
            w["age"], m["age"], w["age_pref"] or []
        )
        if not age_ok:
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
                }
            )
            print(f"[{n}/{total}] 연령 조건 불일치 → 건너뜀")
            continue

        ms = mbti_score(m["mbti"], w["mbti"])
        ts = tag_score(m["want"] or [], m["have"] or [], w["want"] or [], w["have"] or [])
        es, ex_detail = ex_score(m["ex_want"], m["ex_have"], w["ex_want"], w["ex_have"])
        final_score = ms * 0.1 + ts * 0.6 + es * 0.3
        append_score_row(
            {
                "row_num": n,
                "male_id": mid,
                "female_id": wid,
                "status": "scored",
                "mbti_score": f"{ms:.6f}",
                "tag_score": f"{ts:.6f}",
                "ex_score": f"{es:.6f}",
                "ex_detail": _dump_ex_detail(ex_detail),
                "final_score": f"{final_score:.6f}",
            }
        )
        print(
            f"[{n}/{total}] FINAL {final_score:.2f} "
            f"(mbti={ms:.2f}, tag={ts:.2f}, ex={es:.2f})"
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
    with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
        students, male_ids, female_ids = fetch_students(conn)
        expected_total = len(male_ids) * len(female_ids)
        phase = _read_text_file(PHASE_FILE)
        stored_total = (
            _read_int_file(TOTAL_FILE) if os.path.exists(TOTAL_FILE) else count_csv_rows(PAIRS_CSV)
        )
        pairs_ok = os.path.exists(PAIRS_CSV) and stored_total == expected_total

        if phase in {"", "done"} or not pairs_ok:
            total = start_fresh_job(male_ids, female_ids, conn)
            start = 1
            phase = "scoring"
        else:
            total = stored_total
            start = resume_row()
            print(f"이전 작업 재개: phase={phase}, {start}/{total}행부터")

        if phase == "scoring" and start <= total:
            score_remaining_pairs(students, start, total)

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
            names = ", ".join(f"{s['name']}({s['student_id']})" for s in unmatched)
            print(f"미매칭 {len(unmatched)}명: {names}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
