import argparse
import http.client
import os
import re
import urllib.parse
from typing import Any

import psycopg
from dotenv import load_dotenv

load_dotenv()

API_HOST = "munjanara.co.kr"
API_SUCCESS_CODE = "9"
API_CODE_LABELS = {
    "9": "성공",
    "1": "필수전달값이 빠짐",
    "2": "존재하지 않는 아이디",
    "3": "비밀번호 인증실패",
    "4": "잔액 부족",
    "6": "발신번호가 숫자가 아님",
    "7": "사용 중지된 아이디",
    "11": "중복 전송",
    "12": "발신번호 형식 오류",
    "13": "발신번호 미등록",
    "14": "허용되지 않은 아이피",
    "15": "비정상적인 반복 접속",
    "16": "예약시간 설정오류",
}

DEFAULT_MESSAGE = (
    "[{round}차 매칭] {receiver_name}님, 소개팅 매칭 결과가 나왔습니다. "
    "상대: {partner_name} / {partner_phone}"
)


def resolve_round(cli_round: int | None) -> int:
    if cli_round is not None:
        round_no = cli_round
    else:
        round_no = int(os.getenv("MATCH_ROUND", "1"))
    if round_no not in (1, 2):
        raise ValueError("MATCH_ROUND는 1 또는 2여야 합니다.")
    return round_no


def digits_only(phone: str | None) -> str:
    return re.sub(r"\D", "", phone or "")


def send_mesg(
    userid: str,
    passwd: str,
    hp_sender: str,
    hp_receiver: str,
    hp_mesg: str,
    end_alert: int = 0,
) -> tuple[int | None, str]:
    encoded_msg = urllib.parse.quote(hp_mesg)
    url_path = (
        f"/send.sys?userid={urllib.parse.quote(userid)}"
        f"&passwd={urllib.parse.quote(passwd)}"
        f"&sender={urllib.parse.quote(hp_sender)}"
        f"&receiver={urllib.parse.quote(hp_receiver)}"
        f"&encode=1&end_alert={end_alert}&message={encoded_msg}"
    )
    conn = http.client.HTTPConnection(API_HOST, 80, timeout=10)
    try:
        conn.request("GET", url_path, headers={"Host": API_HOST})
        response = conn.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        return response.status, body
    except Exception as e:
        return None, f"Error: {e}"
    finally:
        conn.close()


def parse_api_body(body: str) -> dict[str, str | None]:
    first_line = (body or "").strip().splitlines()[0] if body else ""
    parts = first_line.split("|")
    api_code = (parts[0] or "").strip() if parts else ""
    return {
        "api_code": api_code or None,
        "remaining_balance": parts[1].strip() if len(parts) > 1 else None,
        "provider_send_count": parts[2].strip() if len(parts) > 2 else None,
        "label": API_CODE_LABELS.get(api_code, "알 수 없는 응답"),
    }


def is_success(http_status: int | None, api_code: str | None) -> bool:
    return http_status == 200 and api_code == API_SUCCESS_CODE


def build_message(template: str, round_no: int, receiver_name: str, partner_name: str, partner_phone: str) -> str:
    return template.format(
        round=round_no,
        receiver_name=receiver_name or "참가자",
        partner_name=partner_name or "상대",
        partner_phone=partner_phone or "-",
    )


def fetch_matches(conn, round_no: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                mr.round,
                mr.rank,
                mr.male_id,
                ms.name AS male_name,
                ms.phone AS male_phone,
                mr.female_id,
                fs.name AS female_name,
                fs.phone AS female_phone
            FROM match_result mr
            JOIN student ms ON ms.student_id = mr.male_id
            JOIN student fs ON fs.student_id = mr.female_id
            WHERE mr.round = %s
            ORDER BY mr.rank
            """,
            (round_no,),
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def already_sent_keys(conn, round_no: int) -> set[tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT rank, student_id
            FROM match_message
            WHERE round = %s AND success = TRUE
            """,
            (round_no,),
        )
        return {(row[0], row[1]) for row in cur.fetchall()}


def insert_message_log(conn, row: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO match_message (
                round, rank, student_id, role,
                receiver_phone, sender_phone, message_body,
                success, http_status, api_code, api_response,
                remaining_balance, provider_send_count, error_text
            ) VALUES (
                %(round)s, %(rank)s, %(student_id)s, %(role)s,
                %(receiver_phone)s, %(sender_phone)s, %(message_body)s,
                %(success)s, %(http_status)s, %(api_code)s, %(api_response)s,
                %(remaining_balance)s, %(provider_send_count)s, %(error_text)s
            )
            """,
            row,
        )
    conn.commit()


def recipients_for(match: dict) -> list[dict]:
    return [
        {
            "role": "male",
            "student_id": match["male_id"],
            "receiver_name": match["male_name"],
            "receiver_phone": match["male_phone"],
            "partner_name": match["female_name"],
            "partner_phone": match["female_phone"],
        },
        {
            "role": "female",
            "student_id": match["female_id"],
            "receiver_name": match["female_name"],
            "receiver_phone": match["female_phone"],
            "partner_name": match["male_name"],
            "partner_phone": match["male_phone"],
        },
    ]


def send_for_round(round_no: int, dry_run: bool, force: bool) -> None:
    userid = os.getenv("MUNJANARA_USERID", "").strip()
    passwd = os.getenv("MUNJANARA_PASSWD", "").strip()
    sender = digits_only(os.getenv("MUNJANARA_SENDER", ""))
    template = os.getenv("MATCH_SMS_TEMPLATE", DEFAULT_MESSAGE)
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL이 없습니다.")
    if not dry_run and (not userid or not passwd or not sender):
        raise RuntimeError(
            "실제 발송에는 MUNJANARA_USERID, MUNJANARA_PASSWD, MUNJANARA_SENDER가 필요합니다."
        )

    with psycopg.connect(database_url) as conn:
        matches = fetch_matches(conn, round_no)
        sent = set() if force else already_sent_keys(conn, round_no)
        if not matches:
            print(f"{round_no}차 match_result가 없습니다.")
            return

        ok = fail = skip = 0
        for match in matches:
            for person in recipients_for(match):
                key = (match["rank"], person["student_id"])
                label = (
                    f"{round_no}차 #{match['rank']} "
                    f"{person['receiver_name']}({person['student_id']})"
                )
                if key in sent:
                    skip += 1
                    print(f"건너뜀(이미 성공) {label}")
                    continue

                receiver = digits_only(person["receiver_phone"])
                message = build_message(
                    template,
                    round_no,
                    person["receiver_name"],
                    person["partner_name"],
                    digits_only(person["partner_phone"]),
                )
                if dry_run:
                    skip += 1
                    print(f"[dry-run] {label} → {receiver or '(번호 없음)'}: {message}")
                    continue

                if len(receiver) < 10:
                    log = {
                        "round": round_no,
                        "rank": match["rank"],
                        "student_id": person["student_id"],
                        "role": person["role"],
                        "receiver_phone": receiver or (person["receiver_phone"] or ""),
                        "sender_phone": sender,
                        "message_body": message,
                        "success": False,
                        "http_status": None,
                        "api_code": None,
                        "api_response": None,
                        "remaining_balance": None,
                        "provider_send_count": None,
                        "error_text": "수신번호가 없거나 형식이 올바르지 않음",
                    }
                    insert_message_log(conn, log)
                    fail += 1
                    print(f"실패 {label}: 수신번호 없음")
                    continue

                http_status, body = send_mesg(
                    userid, passwd, sender, receiver, message
                )
                parsed = parse_api_body(body)
                success = is_success(http_status, parsed["api_code"])
                error_text = None
                if not success:
                    if parsed["api_code"]:
                        error_text = parsed["label"]
                    else:
                        error_text = body[:500] if body else "빈 응답"

                insert_message_log(
                    conn,
                    {
                        "round": round_no,
                        "rank": match["rank"],
                        "student_id": person["student_id"],
                        "role": person["role"],
                        "receiver_phone": receiver,
                        "sender_phone": sender,
                        "message_body": message,
                        "success": success,
                        "http_status": http_status,
                        "api_code": parsed["api_code"],
                        "api_response": body[:2000] if body else None,
                        "remaining_balance": parsed["remaining_balance"],
                        "provider_send_count": parsed["provider_send_count"],
                        "error_text": error_text,
                    },
                )
                if success:
                    ok += 1
                    sent.add(key)
                    print(f"성공 {label} → {receiver}")
                else:
                    fail += 1
                    print(f"실패 {label} → {receiver}: {error_text}")

        print(f"완료: 성공 {ok}, 실패 {fail}, 건너뜀 {skip} (총 매칭 {len(matches)}쌍)")


def main() -> None:
    parser = argparse.ArgumentParser(description="match_result 기준으로 매칭 안내 문자를 발송한다.")
    parser.add_argument("--round", type=int, choices=(1, 2), help="접수 차수. 생략하면 MATCH_ROUND 또는 1.")
    parser.add_argument("--dry-run", action="store_true", help="API를 호출하지 않고 발송 대상만 출력한다.")
    parser.add_argument("--force", action="store_true", help="이미 성공 기록이 있어도 다시 보낸다.")
    args = parser.parse_args()
    send_for_round(resolve_round(args.round), args.dry_run, args.force)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
