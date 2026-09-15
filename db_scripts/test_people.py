"""가상 남·여 10명씩 테스트 접수 데이터를 넣는다.

사람(`student`)은 slot(sm01/sw01)으로 고정하고, 차수별 접수(`registration`)만
다시 넣기 때문에 여러 번 실행해도 쌓이지 않는다.

  python db_scripts/test_people.py
  python db_scripts/test_people.py --round 2
  python db_scripts/test_people.py --purge
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

CONSENT_VERSION = "2026.09.09-3"
THIRD_PARTY_CONSENT_VERSION = "2026.09.09-3-tp"
USER_AGENT = "qr-match/db_scripts/test_people.py"

# 사람 PK는 sm01 / sw01. 1차 접수는 같은 id, 2차는 sm01-r2.
PEOPLE: list[dict] = [
    {
        "slot": "sm01",
        "name": "김민준",
        "gender": False,
        "age": 24,
        "mbti": "ENTJ",
        "phone": "010-1000-0001",
        "major_id": "computer-software",
        "have": ["운동하는", "유머러스한"],
        "want": ["다정한", "성격이 밝은"],
        "age_prefs": ["same", "younger"],
        "ex_have": "주말에 같이 등산하고 맛집 찾아다니는 걸 좋아함. 리액션이 크고 대화를 잘 이끌어 줘.",
        "ex_want": "잘 들어주고 다정하게 챙기는 사람. 분위기를 밝게 만드는 사람.",
    },
    {
        "slot": "sm02",
        "name": "이서준",
        "gender": False,
        "age": 22,
        "mbti": "ISFJ",
        "phone": "010-1000-0002",
        "major_id": "electrical",
        "have": ["조용한", "다정한"],
        "want": ["유머러스한", "성격이 밝은"],
        "age_prefs": ["any"],
        "ex_have": "조용히 옆에 있어주고 세심하게 챙김. 말보다 행동으로 보여주는 타입.",
        "ex_want": "유머 감각 있고 분위기를 띄워주는 사람. 너무 무겁지 않았으면 함.",
    },
    {
        "slot": "sm03",
        "name": "박도윤",
        "gender": False,
        "age": 26,
        "mbti": "ESTP",
        "phone": "010-1000-0003",
        "major_id": "sports-rehab",
        "have": ["운동하는", "키가 큰"],
        "want": ["귀여운", "단발 머리"],
        "age_prefs": ["younger"],
        "ex_have": "헬스와 축구를 자주 함. 키 크고 활동적인 데이트를 좋아함.",
        "ex_want": "귀엽고 단발인 스타일. 같이 운동하거나 밖에서 놀 수 있는 사람.",
    },
    {
        "slot": "sm04",
        "name": "최하준",
        "gender": False,
        "age": 23,
        "mbti": "INFP",
        "phone": "010-1000-0004",
        "major_id": "hotel-culinary",
        "have": ["요리를 잘하는", "솔직한"],
        "want": ["대화를 잘하는", "성격이 밝은"],
        "age_prefs": ["same", "older"],
        "ex_have": "집에서 요리해서 대접하는 걸 좋아함. 감정은 솔직하게 말하는 편.",
        "ex_want": "대화가 잘 통하고 성격이 밝은 사람. 깊은 이야기도 나눌 수 있으면 좋음.",
    },
    {
        "slot": "sm05",
        "name": "정우진",
        "gender": False,
        "age": 25,
        "mbti": "ENTP",
        "phone": "010-1000-0005",
        "major_id": "game-contents",
        "have": ["유머러스한", "대화를 잘하는"],
        "want": ["요리를 잘하는", "다정한"],
        "age_prefs": ["any"],
        "ex_have": "드립과 유머로 분위기를 잘 띄움. 처음 만난 사람과도 대화가 잘 됨.",
        "ex_want": "요리를 잘하고 다정하게 챙겨주는 사람.",
    },
    {
        "slot": "sm06",
        "name": "강지호",
        "gender": False,
        "age": 21,
        "mbti": "ISTP",
        "phone": "010-1000-0006",
        "major_id": "electronics",
        "have": ["운전을 잘하는", "운동하는"],
        "want": ["키가 큰", "옷을 잘 입는"],
        "age_prefs": ["older"],
        "ex_have": "운전 자신 있고 주말 드라이브를 좋아함. 운동도 꾸준히 함.",
        "ex_want": "키가 크고 옷 잘 입는 사람. 같이 멀리 놀러 갈 수 있으면 좋음.",
    },
    {
        "slot": "sm07",
        "name": "윤태양",
        "gender": False,
        "age": 27,
        "mbti": "ESFJ",
        "phone": "010-1000-0007",
        "major_id": "food-nutrition",
        "have": ["다정한", "요리를 잘하는"],
        "want": ["조용한"],
        "age_prefs": ["younger", "same"],
        "ex_have": "다정하고 집밥을 잘 해줌. 상대 기분을 잘 살핌.",
        "ex_want": "너무 시끄럽지 않고 차분한 사람.",
    },
    {
        "slot": "sm08",
        "name": "장현우",
        "gender": False,
        "age": 24,
        "mbti": "INTJ",
        "phone": "010-1000-0008",
        "major_id": "business",
        "have": ["솔직한", "조용한"],
        "want": ["운동하는"],
        "age_prefs": ["any"],
        "ex_have": "솔직하고 말수가 적음. 관심사 이야기할 때는 깊게 들어감.",
        "ex_want": "운동을 좋아하고 자기관리를 하는 사람.",
    },
    {
        "slot": "sm09",
        "name": "한승민",
        "gender": False,
        "age": 22,
        "mbti": "ENFP",
        "phone": "010-1000-0009",
        "major_id": "k-pop",
        "have": ["노래를 잘하는", "성격이 밝은"],
        "want": ["유머러스한", "대화를 잘하는"],
        "age_prefs": ["same"],
        "ex_have": "노래 잘하고 분위기 메이커. 사람들 사이에서 밝은 편.",
        "ex_want": "유머러스하고 대화가 끊기지 않는 사람.",
    },
    {
        "slot": "sm10",
        "name": "오준서",
        "gender": False,
        "age": 28,
        "mbti": "ISTJ",
        "phone": "010-1000-0010",
        "major_id": "visual-design",
        "have": ["키가 큰", "옷을 잘 입는"],
        "want": ["귀여운", "긴 머리"],
        "age_prefs": ["younger"],
        "ex_have": "키 크고 옷을 깔끔하게 입음. 약속 시간을 잘 지킴.",
        "ex_want": "긴 머리에 귀여운 인상. 다정한 사람.",
    },
    {
        "slot": "sw01",
        "name": "서윤아",
        "gender": True,
        "age": 23,
        "mbti": "INFJ",
        "phone": "010-2000-0001",
        "major_id": "nursing",
        "have": ["다정한", "성격이 밝은", "긴 머리"],
        "want": ["솔직한", "조용한"],
        "age_prefs": ["older", "same"],
        "ex_have": "다정하고 성격이 밝음. 긴 머리. 상대 이야기를 잘 들어줘.",
        "ex_want": "솔직하고 조용한 사람. 감정 기복이 크지 않았으면 함.",
    },
    {
        "slot": "sw02",
        "name": "정하은",
        "gender": True,
        "age": 24,
        "mbti": "ESFP",
        "phone": "010-2000-0002",
        "major_id": "hair-design",
        "have": ["귀여운", "단발 머리", "성격이 밝은"],
        "want": ["운동하는", "키가 큰"],
        "age_prefs": ["any"],
        "ex_have": "귀엽다는 말을 많이 들음. 단발. 성격이 밝고 즉흥적인 데이트를 좋아함.",
        "ex_want": "운동하고 키 큰 사람. 같이 밖에서 활동할 수 있으면 좋음.",
    },
    {
        "slot": "sw03",
        "name": "김수빈",
        "gender": True,
        "age": 22,
        "mbti": "ENFJ",
        "phone": "010-2000-0003",
        "major_id": "early-childhood",
        "have": ["대화를 잘하는", "성격이 밝은"],
        "want": ["요리를 잘하는", "다정한"],
        "age_prefs": ["same", "older"],
        "ex_have": "대화를 잘 이끌어 주고 리액션이 좋음. 성격이 밝은 편.",
        "ex_want": "요리를 잘하고 다정하게 챙기는 사람.",
    },
    {
        "slot": "sw04",
        "name": "이지민",
        "gender": True,
        "age": 26,
        "mbti": "ISTJ",
        "phone": "010-2000-0004",
        "major_id": "fashion-design-business",
        "have": ["요리를 잘하는", "옷을 잘 입는"],
        "want": ["유머러스한", "대화를 잘하는"],
        "age_prefs": ["younger", "same"],
        "ex_have": "요리를 잘하고 옷을 잘 입음. 계획적인 데이트를 좋아함.",
        "ex_want": "유머러스하고 대화가 잘 되는 사람.",
    },
    {
        "slot": "sw05",
        "name": "박소율",
        "gender": True,
        "age": 21,
        "mbti": "INFP",
        "phone": "010-2000-0005",
        "major_id": "webtoon",
        "have": ["조용한", "키가 작은"],
        "want": ["다정한", "요리를 잘하는"],
        "age_prefs": ["older"],
        "ex_have": "조용하고 키가 작은 편. 카페에서 오래 이야기하는 걸 좋아함.",
        "ex_want": "다정하고 요리를 잘하는 사람. 세심하게 챙겨주면 좋음.",
    },
    {
        "slot": "sw06",
        "name": "최예린",
        "gender": True,
        "age": 25,
        "mbti": "ESTJ",
        "phone": "010-2000-0006",
        "major_id": "sports-rehab",
        "have": ["운동하는", "키가 큰"],
        "want": ["유머러스한", "솔직한"],
        "age_prefs": ["any"],
        "ex_have": "운동을 꾸준히 하고 키가 큼. 자기관리에 신경 씀.",
        "ex_want": "유머 있고 솔직한 사람. 돌려 말하지 않았으면 함.",
    },
    {
        "slot": "sw07",
        "name": "한지우",
        "gender": True,
        "age": 24,
        "mbti": "ISFP",
        "phone": "010-2000-0007",
        "major_id": "k-pop",
        "have": ["노래를 잘하는", "긴 머리", "귀여운"],
        "want": ["운전을 잘하는", "운동하는"],
        "age_prefs": ["same", "older"],
        "ex_have": "노래 잘하고 긴 머리에 귀엽다는 말 들음. 감성적인 데이트를 좋아함.",
        "ex_want": "운전 잘하고 운동하는 사람. 같이 여행 가고 싶음.",
    },
    {
        "slot": "sw08",
        "name": "윤채원",
        "gender": True,
        "age": 27,
        "mbti": "ENTP",
        "phone": "010-2000-0008",
        "major_id": "business",
        "have": ["유머러스한", "대화를 잘하는"],
        "want": ["요리를 잘하는", "다정한"],
        "age_prefs": ["younger"],
        "ex_have": "유머러스하고 대화를 잘함. 처음 만난 자리에서도 어색하지 않음.",
        "ex_want": "요리를 잘하고 다정한 사람.",
    },
    {
        "slot": "sw09",
        "name": "강다은",
        "gender": True,
        "age": 22,
        "mbti": "ESFJ",
        "phone": "010-2000-0009",
        "major_id": "food-nutrition",
        "have": ["다정한", "요리를 잘하는"],
        "want": ["성격이 밝은", "노래를 잘하는"],
        "age_prefs": ["same"],
        "ex_have": "다정하고 요리를 잘함. 상대를 잘 챙김.",
        "ex_want": "성격이 밝고 노래 잘하는 사람. 분위기가 즐겁웠으면 함.",
    },
    {
        "slot": "sw10",
        "name": "오하늘",
        "gender": True,
        "age": 23,
        "mbti": "INTP",
        "phone": "010-2000-0010",
        "major_id": "computer-software",
        "have": ["솔직한", "조용한", "옷을 잘 입는"],
        "want": ["운동하는", "유머러스한"],
        "age_prefs": ["older", "same"],
        "ex_have": "솔직하고 조용함. 옷을 잘 입는다는 말 들음.",
        "ex_want": "운동하고 유머러스한 사람. 너무 무겁지 않았으면 함.",
    },
]

def student_id_for(slot: str) -> str:
    return slot


def registration_id_for(slot: str, round_no: int) -> str:
    return slot if round_no == 1 else f"{slot}-r2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="테스트용 가상 접수 20명을 넣거나 지운다.")
    parser.add_argument(
        "--round",
        type=int,
        choices=(1, 2),
        default=None,
        help="접수 차수. 생략하면 MATCH_ROUND 또는 1.",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="테스트 접수를 모든 차수에서 삭제만 하고 끝낸다.",
    )
    return parser.parse_args()


def resolve_round(cli_round: int | None) -> int:
    if cli_round is not None:
        return cli_round
    raw = os.getenv("MATCH_ROUND", "1")
    round_no = int(raw)
    if round_no not in (1, 2):
        raise ValueError("MATCH_ROUND는 1 또는 2여야 합니다.")
    return round_no


def fetch_charm_ids(cur) -> dict[str, str]:
    cur.execute("SELECT charm_id::text, name FROM charm")
    return {name: charm_id for charm_id, name in cur.fetchall()}


def require_charms(charm_ids: dict[str, str]) -> None:
    needed: set[str] = set()
    for person in PEOPLE:
        needed.update(person["have"])
        needed.update(person["want"])
    missing = sorted(needed - set(charm_ids))
    if missing:
        raise RuntimeError(f"charm 테이블에 없는 태그: {', '.join(missing)}")


def require_majors(cur) -> None:
    wanted = sorted({person["major_id"] for person in PEOPLE})
    cur.execute("SELECT major_id FROM major WHERE major_id = ANY(%s)", (wanted,))
    found = {row[0] for row in cur.fetchall()}
    missing = [mid for mid in wanted if mid not in found]
    if missing:
        raise RuntimeError(f"major 테이블에 없는 학과: {', '.join(missing)}")


def require_consent_notices(cur) -> dict[str, tuple[str, str]]:
    versions = [CONSENT_VERSION, THIRD_PARTY_CONSENT_VERSION]
    cur.execute(
        """
        SELECT version, body, body_hash
        FROM consent_notice
        WHERE version = ANY(%s)
        """,
        (versions,),
    )
    notices = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    missing = [v for v in versions if v not in notices]
    if missing:
        raise RuntimeError(f"consent_notice에 없는 버전: {', '.join(missing)}")
    return notices


def test_registration_ids(cur, round_no: int | None) -> list[str]:
    if round_no is None:
        cur.execute(
            """
            SELECT registration_id
            FROM registration
            WHERE registration_id ~ '^(sm|sw)[0-9]{2}(-r2)?$'
               OR student_id ~ '^(sm|sw)[0-9]{2}$'
            """
        )
    else:
        ids = [registration_id_for(person["slot"], round_no) for person in PEOPLE]
        cur.execute(
            """
            SELECT registration_id
            FROM registration
            WHERE round = %s
              AND (
                registration_id = ANY(%s)
                OR student_id ~ '^(sm|sw)[0-9]{2}$'
              )
            """,
            (round_no, ids),
        )
    return [row[0] for row in cur.fetchall()]


def delete_registrations(cur, registration_ids: list[str]) -> int:
    if not registration_ids:
        return 0
    cur.execute(
        "DELETE FROM consent WHERE registration_id = ANY(%s)",
        (registration_ids,),
    )
    cur.execute(
        """
        DELETE FROM registration
        WHERE registration_id = ANY(%s)
        RETURNING registration_id
        """,
        (registration_ids,),
    )
    deleted = len(cur.fetchall())
    cur.execute(
        """
        DELETE FROM student s
        WHERE s.student_id ~ '^(sm|sw)[0-9]{2}$'
          AND NOT EXISTS (
            SELECT 1 FROM registration r WHERE r.student_id = s.student_id
          )
        """
    )
    return deleted


def birth_from_age(age: int) -> str:
    year = 2026 - int(age)
    return f"{year % 100:02d}0101"


def upsert_students(cur) -> None:
    rows = [
        (
            student_id_for(person["slot"]),
            person["name"],
            person["phone"],
            person["gender"],
            birth_from_age(person["age"]),
            person["major_id"],
        )
        for person in PEOPLE
    ]
    cur.executemany(
        """
        INSERT INTO student (student_id, name, phone, gender, birth, major_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (student_id) DO UPDATE
        SET name = EXCLUDED.name,
            phone = EXCLUDED.phone,
            gender = EXCLUDED.gender,
            birth = EXCLUDED.birth,
            major_id = EXCLUDED.major_id
        """,
        rows,
    )


def insert_registrations(
    cur, round_no: int, charm_ids: dict[str, str], notices: dict[str, tuple[str, str]]
) -> None:
    registrations = []
    prefs = []
    haves = []
    wants = []
    ex_haves = []
    ex_wants = []
    consents = []

    for person in PEOPLE:
        sid = student_id_for(person["slot"])
        rid = registration_id_for(person["slot"], round_no)
        registrations.append((rid, sid, round_no, person["mbti"]))
        for pref_id in person["age_prefs"]:
            prefs.append((rid, pref_id))
        for name in person["have"]:
            haves.append((rid, charm_ids[name]))
        for name in person["want"]:
            wants.append((rid, charm_ids[name]))
        ex_haves.append((rid, person["ex_have"]))
        ex_wants.append((rid, person["ex_want"]))
        for version in (CONSENT_VERSION, THIRD_PARTY_CONSENT_VERSION):
            body, body_hash = notices[version]
            consents.append((rid, version, True, body, body_hash, USER_AGENT))

    cur.executemany(
        """
        INSERT INTO registration (registration_id, student_id, round, mbti)
        VALUES (%s, %s, %s, %s)
        """,
        registrations,
    )
    cur.executemany(
        "INSERT INTO prefer_age (registration_id, age_pref_id) VALUES (%s, %s)",
        prefs,
    )
    cur.executemany(
        "INSERT INTO have (registration_id, charm_id) VALUES (%s, %s)",
        haves,
    )
    cur.executemany(
        "INSERT INTO want (registration_id, charm_id) VALUES (%s, %s)",
        wants,
    )
    cur.executemany(
        "INSERT INTO ex_have (registration_id, charm) VALUES (%s, %s)",
        ex_haves,
    )
    cur.executemany(
        "INSERT INTO ex_want (registration_id, charm) VALUES (%s, %s)",
        ex_wants,
    )
    cur.executemany(
        """
        INSERT INTO consent (
            registration_id, notice_version, agreed, consent_text_snapshot,
            consent_hash, user_agent
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        consents,
    )


def main() -> None:
    args = parse_args()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL이 없습니다.")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            if args.purge:
                ids = test_registration_ids(cur, None)
                deleted = delete_registrations(cur, ids)
                conn.commit()
                print(f"테스트 접수 {deleted}건 삭제")
                return

            round_no = resolve_round(args.round)
            charm_ids = fetch_charm_ids(cur)
            require_charms(charm_ids)
            require_majors(cur)
            notices = require_consent_notices(cur)

            ids = test_registration_ids(cur, round_no)
            deleted = delete_registrations(cur, ids)
            upsert_students(cur)
            insert_registrations(cur, round_no, charm_ids, notices)
            conn.commit()

            male = sum(1 for p in PEOPLE if p["gender"] is False)
            female = sum(1 for p in PEOPLE if p["gender"] is True)
            print(
                f"{round_no}차 테스트 접수 {male}남 {female}여 넣음"
                + (f" (기존 {deleted}건 교체)" if deleted else "")
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
