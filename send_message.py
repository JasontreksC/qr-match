import argparse
import http.client
import json
import os
import re
import socket
import ssl
from typing import Any
from urllib.parse import urlparse

import psycopg
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_URL = "https://api.munjaon.co.kr/api/send/sendMsg"
SUCCESS_RESULT_CODES = {"0", "00"}
MULTIPART_BOUNDARY = "____boundary____"

DEFAULT_MESSAGE = (
    "[QRrious - QR소개팅]\n"
    "{이름}님, 축하드립니다! {n}차 접수에서 매칭되셨습니다.\n"
    "아래 링크에서 로그인하여 매칭된 상대방을 확인해보세요.\n"
    "https://qrious-ysu.vercel.app\n"
    "접수하셨던 폼과 같은 주소이니 안심하셔도 좋아요."
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


class HTTPSConnectionIPv4(http.client.HTTPSConnection):
    """문자온 IP 화이트리스트는 IPv4만 등록되므로 AAAA(IPv6)로 붙지 않게 한다."""

    def connect(self) -> None:
        infos = socket.getaddrinfo(self.host, self.port, socket.AF_INET, socket.SOCK_STREAM)
        sock = socket.create_connection(infos[0][4], self.timeout)
        context = self._context or ssl.create_default_context()
        self.sock = context.wrap_socket(sock, server_hostname=self.host)


def lookup_public_ipv4() -> str:
    conn = HTTPSConnectionIPv4("api.ipify.org", 443, timeout=8)
    try:
        conn.request("GET", "/", headers={"Host": "api.ipify.org"})
        return conn.getresponse().read().decode("utf-8", errors="replace").strip()
    finally:
        conn.close()


def _multipart_body(fields: dict[str, str]) -> bytes:
    # Apache HttpClient BROWSER_COMPATIBLE 실제 전송과 같게:
    # 텍스트 파트에는 Content-Disposition만 넣고, part Content-Type은 생략한다.
    chunks: list[bytes] = []
    for key, value in fields.items():
        if value is None:
            continue
        chunks.append(f"--{MULTIPART_BOUNDARY}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    chunks.append(f"--{MULTIPART_BOUNDARY}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def send_mesg(
    mber_id: str,
    access_key: str,
    call_from: str,
    call_to: str,
    sms_txt: str,
    name_str: str = "",
    test_yn: str = "",
    api_url: str = DEFAULT_API_URL,
) -> tuple[int | None, str]:
    fields = {
        "mberId": mber_id,
        "accessKey": access_key,
        "callFrom": call_from,
        "callToList": call_to,
        "smsTxt": sms_txt,
    }
    if name_str:
        fields["nameStr"] = name_str
    if test_yn:
        fields["test_yn"] = test_yn
    body = _multipart_body(fields)
    parsed = urlparse(api_url)
    host = parsed.hostname or "api.munjaon.co.kr"
    port = parsed.port or 443
    path = parsed.path or "/api/send/sendMsg"
    conn = HTTPSConnectionIPv4(host, port, timeout=15)
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Host": host,
                "User-Agent": "Apache-HttpClient/4.5.13 (Java/1.8.0_202)",
                "Content-Type": f"multipart/form-data; boundary={MULTIPART_BOUNDARY}",
            },
        )
        response = conn.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        return response.status, raw
    except Exception as e:
        return None, json.dumps({"resultCode": "error", "data": str(e)}, ensure_ascii=False)
    finally:
        conn.close()


def _first(*values):
    for value in values:
        if value is not None:
            return value
    return None


def parse_api_body(body: str) -> dict[str, Any]:
    try:
        parsed = json.loads(body) if body else {}
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    data = parsed.get("data")
    data_obj = data if isinstance(data, dict) else {}
    result_code = str(_first(parsed.get("resultCode"), parsed.get("result_code"), data_obj.get("resultCode"), "") )
    error_from_data = data if isinstance(data, str) else None
    if not error_from_data:
        error_from_data = parsed.get("message") or data_obj.get("message")
    success_cnt_raw = _first(data_obj.get("successCnt"), parsed.get("successCnt"))
    fail_cnt_raw = _first(data_obj.get("failCnt"), parsed.get("failCnt"))
    success_cnt = "" if success_cnt_raw is None else str(success_cnt_raw)
    fail_cnt = "" if fail_cnt_raw is None else str(fail_cnt_raw)
    success = result_code in SUCCESS_RESULT_CODES
    if success and success_cnt == "0" and fail_cnt not in ("", "0"):
        success = False
    block_cnt = _first(data_obj.get("blockCnt"), parsed.get("blockCnt"))
    test_yn = _first(data_obj.get("test_yn"), parsed.get("test_yn"))
    return {
        "result_code": result_code or None,
        "msg_group_id": _first(data_obj.get("msgGroupId"), parsed.get("msgGroupId")),
        "msg_type": _first(data_obj.get("msgType"), parsed.get("msgType")),
        "block_cnt": None if block_cnt is None else str(block_cnt),
        "fail_cnt": None if fail_cnt == "" else fail_cnt,
        "success_cnt": None if success_cnt == "" else success_cnt,
        "test_yn": None if test_yn is None else str(test_yn),
        "error_text": error_from_data,
        "success": success,
    }


def build_message(
    template: str,
    round_no: int,
    receiver_name: str,
    partner_name: str,
    partner_phone: str,
) -> str:
    name = receiver_name or "참가자"
    return template.format(
        이름=name,
        n=round_no,
        round=round_no,
        receiver_name=name,
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
                msg_group_id, msg_type, block_cnt, fail_cnt,
                success_cnt, test_yn, error_text
            ) VALUES (
                %(round)s, %(rank)s, %(student_id)s, %(role)s,
                %(receiver_phone)s, %(sender_phone)s, %(message_body)s,
                %(success)s, %(http_status)s, %(api_code)s, %(api_response)s,
                %(msg_group_id)s, %(msg_type)s, %(block_cnt)s, %(fail_cnt)s,
                %(success_cnt)s, %(test_yn)s, %(error_text)s
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


def empty_log(
    round_no: int,
    match: dict,
    person: dict,
    sender: str,
    receiver: str,
    message: str,
    test_yn: str,
    error_text: str,
) -> dict[str, Any]:
    return {
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
        "msg_group_id": None,
        "msg_type": None,
        "block_cnt": None,
        "fail_cnt": None,
        "success_cnt": None,
        "test_yn": test_yn or None,
        "error_text": error_text,
    }


def send_for_round(round_no: int, dry_run: bool, force: bool, test_send: bool) -> None:
    mber_id = os.getenv("MUNJAON_MBER_ID", "").strip()
    access_key = os.getenv("MUNJAON_ACCESS_KEY", "").strip()
    sender = digits_only(os.getenv("MUNJAON_SENDER", ""))
    test_yn = "Y" if test_send else os.getenv("MUNJAON_TEST_YN", "").strip()
    api_url = os.getenv("MUNJAON_API_URL", DEFAULT_API_URL).strip() or DEFAULT_API_URL
    template = os.getenv("MATCH_SMS_TEMPLATE", DEFAULT_MESSAGE)
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL이 없습니다.")
    if not dry_run and (not mber_id or not access_key or not sender):
        raise RuntimeError(
            "실제 발송에는 MUNJAON_MBER_ID, MUNJAON_ACCESS_KEY, MUNJAON_SENDER가 필요합니다."
        )

    if not dry_run:
        try:
            egress = lookup_public_ipv4()
            print(f"이 프로그램의 출구 IPv4: {egress}  (문자온 등록 IP와 같아야 함)")
        except Exception as e:
            print(f"출구 IPv4 확인 실패: {e}")

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
                    insert_message_log(
                        conn,
                        empty_log(
                            round_no,
                            match,
                            person,
                            sender,
                            receiver,
                            message,
                            test_yn,
                            "수신번호가 없거나 형식이 올바르지 않음",
                        ),
                    )
                    fail += 1
                    print(f"실패 {label}: 수신번호 없음")
                    continue

                name_str = person["receiver_name"] or "" if "*이름*" in message else ""
                http_status, body = send_mesg(
                    mber_id,
                    access_key,
                    sender,
                    receiver,
                    message,
                    name_str=name_str,
                    test_yn=test_yn,
                    api_url=api_url,
                )
                parsed = parse_api_body(body)
                success = bool(http_status == 200 and parsed["success"])
                error_text = None
                if not success:
                    error_text = parsed["error_text"] or parsed["result_code"] or (body[:500] if body else "빈 응답")

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
                        "api_code": parsed["result_code"],
                        "api_response": body[:2000] if body else None,
                        "msg_group_id": parsed["msg_group_id"],
                        "msg_type": parsed["msg_type"],
                        "block_cnt": parsed["block_cnt"],
                        "fail_cnt": parsed["fail_cnt"],
                        "success_cnt": parsed["success_cnt"],
                        "test_yn": parsed["test_yn"] if parsed["test_yn"] is not None else (test_yn or None),
                        "error_text": error_text,
                    },
                )
                if success:
                    ok += 1
                    sent.add(key)
                    print(f"성공 {label} → {receiver} (group={parsed['msg_group_id']})")
                else:
                    fail += 1
                    print(f"실패 {label} → {receiver}: {error_text}")

        print(f"완료: 성공 {ok}, 실패 {fail}, 건너뜀 {skip} (총 매칭 {len(matches)}쌍)")


def main() -> None:
    parser = argparse.ArgumentParser(description="match_result 기준으로 매칭 안내 문자를 발송한다.")
    parser.add_argument("--round", type=int, choices=(1, 2), help="접수 차수. 생략하면 MATCH_ROUND 또는 1.")
    parser.add_argument("--dry-run", action="store_true", help="API를 호출하지 않고 발송 대상만 출력한다.")
    parser.add_argument("--force", action="store_true", help="이미 성공 기록이 있어도 다시 보낸다.")
    parser.add_argument("--test", action="store_true", help="문자온 test_yn=Y 로 테스트 발송한다.")
    args = parser.parse_args()
    send_for_round(resolve_round(args.round), args.dry_run, args.force, args.test)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
