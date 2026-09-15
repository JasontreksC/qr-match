"""match_result를 사람 기준으로 조인해 한 장의 CSV로 저장한다."""

import argparse
import csv
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
KIND_LABEL = {
    "incomparable": "비교불가",
    "opposed": "반대",
    "similar": "비슷함",
    "match": "맞음",
}
HEADERS = [
    "차수",
    "순위",
    "최종점수",
    "MBTI점수",
    "매력태그점수",
    "기타매력점수",
    "남_이름",
    "남_나이",
    "남_전공",
    "남_MBTI",
    "남_전화번호",
    "남_매력",
    "남_이상형(태그)",
    "남_나이선호",
    "남_기타매력",
    "남_기타이상형",
    "여_이름",
    "여_나이",
    "여_전공",
    "여_MBTI",
    "여_전화번호",
    "여_매력",
    "여_이상형(태그)",
    "여_나이선호",
    "여_기타매력",
    "여_기타이상형",
    "남이 본 여_기타매력채점",
    "여가 본 남_기타매력채점",
]


def resolve_round(cli_round: int | None) -> int:
    if cli_round is not None:
        round_no = cli_round
    else:
        round_no = int(os.getenv("MATCH_ROUND", "1"))
    if round_no not in (1, 2):
        raise ValueError("MATCH_ROUND는 1 또는 2여야 합니다.")
    return round_no


def _join_names(value) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return str(value)


def _fmt_score(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _fmt_ex_item(item: dict, name: str = "") -> str:
    kind = KIND_LABEL.get(item.get("match_kind"), item.get("match_kind") or "")
    reason = (item.get("reason") or "").replace("\n", " ").strip()
    want_ev = (item.get("want_evidence") or "").replace("\n", " / ").strip()
    have_ev = (item.get("have_evidence") or "").replace("\n", " / ").strip()
    evidence = (item.get("evidence") or "").replace("\n", " / ").strip()
    parts = [f"{name}: {kind}"] if name else [kind]
    if reason:
        parts.append(reason)
    if want_ev or have_ev:
        if want_ev:
            parts.append(f"이상형: {want_ev}")
        if have_ev:
            parts.append(f"매력: {have_ev}")
    elif evidence:
        parts.append(f"근거: {evidence}")
    return " — ".join(parts)


def _fmt_ex_side(detail: dict | None, key: str) -> str:
    if not isinstance(detail, dict):
        return ""
    side = detail.get(key) or {}
    if not isinstance(side, dict):
        return ""
    lines = []
    score = side.get("score")
    if score is not None:
        lines.append(f"점수 {_fmt_score(score)}")
    if side.get("criteria"):
        for item in side["criteria"]:
            lines.append(_fmt_ex_item(item, item.get("criterion") or ""))
    elif side.get("match_kind") or side.get("reason"):
        lines.append(_fmt_ex_item(side))
    return "\n".join(lines)


def fetch_rows(conn, round_no: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                mr.round,
                mr.rank,
                mr.final_score,
                mr.mbti_score,
                mr.tag_score,
                mr.ex_score,
                mr.ex_detail,
                ms.name,
                ms.age,
                COALESCE(mm.name, ''),
                ms.mbti,
                ms.phone,
                COALESCE(
                    (
                        SELECT string_agg(c.name, ', ' ORDER BY c.name)
                        FROM have h
                        JOIN charm c ON c.charm_id = h.charm_id
                        WHERE h.student_id = ms.student_id
                    ),
                    ''
                ),
                COALESCE(
                    (
                        SELECT string_agg(c.name, ', ' ORDER BY c.name)
                        FROM want w
                        JOIN charm c ON c.charm_id = w.charm_id
                        WHERE w.student_id = ms.student_id
                    ),
                    ''
                ),
                COALESCE(
                    (
                        SELECT string_agg(ap.name, ', ' ORDER BY ap.sort_order)
                        FROM prefer_age pa
                        JOIN age_pref ap ON ap.age_pref_id = pa.age_pref_id
                        WHERE pa.student_id = ms.student_id
                    ),
                    ''
                ),
                COALESCE(meh.charm, ''),
                COALESCE(mew.charm, ''),
                fs.name,
                fs.age,
                COALESCE(fm.name, ''),
                fs.mbti,
                fs.phone,
                COALESCE(
                    (
                        SELECT string_agg(c.name, ', ' ORDER BY c.name)
                        FROM have h
                        JOIN charm c ON c.charm_id = h.charm_id
                        WHERE h.student_id = fs.student_id
                    ),
                    ''
                ),
                COALESCE(
                    (
                        SELECT string_agg(c.name, ', ' ORDER BY c.name)
                        FROM want w
                        JOIN charm c ON c.charm_id = w.charm_id
                        WHERE w.student_id = fs.student_id
                    ),
                    ''
                ),
                COALESCE(
                    (
                        SELECT string_agg(ap.name, ', ' ORDER BY ap.sort_order)
                        FROM prefer_age pa
                        JOIN age_pref ap ON ap.age_pref_id = pa.age_pref_id
                        WHERE pa.student_id = fs.student_id
                    ),
                    ''
                ),
                COALESCE(feh.charm, ''),
                COALESCE(few.charm, '')
            FROM match_result mr
            JOIN student ms ON ms.student_id = mr.male_id
            JOIN student fs ON fs.student_id = mr.female_id
            LEFT JOIN major mm ON mm.major_id = ms.major_id
            LEFT JOIN major fm ON fm.major_id = fs.major_id
            LEFT JOIN ex_have meh ON meh.student_id = ms.student_id
            LEFT JOIN ex_want mew ON mew.student_id = ms.student_id
            LEFT JOIN ex_have feh ON feh.student_id = fs.student_id
            LEFT JOIN ex_want few ON few.student_id = fs.student_id
            WHERE mr.round = %s
            ORDER BY mr.rank
            """,
            (round_no,),
        )
        rows = []
        for row in cur.fetchall():
            detail = row[6]
            rows.append(
                {
                    "차수": row[0],
                    "순위": row[1],
                    "최종점수": _fmt_score(row[2]),
                    "MBTI점수": _fmt_score(row[3]),
                    "매력태그점수": _fmt_score(row[4]),
                    "기타매력점수": _fmt_score(row[5]),
                    "남_이름": row[7],
                    "남_나이": row[8],
                    "남_전공": row[9],
                    "남_MBTI": row[10],
                    "남_전화번호": row[11],
                    "남_매력": _join_names(row[12]),
                    "남_이상형(태그)": _join_names(row[13]),
                    "남_나이선호": _join_names(row[14]),
                    "남_기타매력": row[15],
                    "남_기타이상형": row[16],
                    "여_이름": row[17],
                    "여_나이": row[18],
                    "여_전공": row[19],
                    "여_MBTI": row[20],
                    "여_전화번호": row[21],
                    "여_매력": _join_names(row[22]),
                    "여_이상형(태그)": _join_names(row[23]),
                    "여_나이선호": _join_names(row[24]),
                    "여_기타매력": row[25],
                    "여_기타이상형": row[26],
                    "남이 본 여_기타매력채점": _fmt_ex_side(detail, "male_to_female"),
                    "여가 본 남_기타매력채점": _fmt_ex_side(detail, "female_to_male"),
                }
            )
        return rows


def write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="매칭 결과를 소개팅용 한 장 CSV로 저장한다. UUID 등 DB 식별자는 넣지 않는다."
    )
    parser.add_argument("--round", type=int, choices=(1, 2), help="접수 차수. 생략하면 MATCH_ROUND 또는 1.")
    parser.add_argument("--output", help="저장 경로. 생략하면 매칭결과_{차수}차.csv")
    args = parser.parse_args()

    round_no = resolve_round(args.round)
    output = args.output or os.path.join(ROOT, f"매칭결과_{round_no}차.csv")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL이 없습니다.")

    with psycopg.connect(database_url) as conn:
        rows = fetch_rows(conn, round_no)

    if not rows:
        raise RuntimeError(f"{round_no}차 match_result가 없습니다.")

    write_csv(rows, output)
    print(f"{round_no}차 매칭 {len(rows)}쌍 → {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        raise SystemExit(1)
