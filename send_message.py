import argparse
import http.client
import json
import os
import re
import socket
import ssl
from contextlib import nullcontext
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

import db

load_dotenv()

DEFAULT_API_URL = "https://api.munjaon.co.kr/api/send/sendMsg"
SUCCESS_RESULT_CODES = {"0", "00"}
MULTIPART_BOUNDARY = "____boundary____"

DEFAULT_MESSAGE = (
    "[QRious - QR소개팅]\n"
    "지금 매칭 결과를 확인해보세요!\n"
    "https://qrious-ysu.vercel.app"
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


def fetch_recipients(conn, round_no: int) -> list[dict]:
    """해당 차수 접수자 전원. 매칭된 사람만 상대 이름·번호와 순위가 채워진다."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                r.registration_id,
                s.name,
                s.phone,
                s.gender,
                mr.rank,
                ps.name AS partner_name,
                ps.phone AS partner_phone
            FROM registration r
            JOIN student s ON s.student_id = r.student_id
            LEFT JOIN match_result mr
              ON mr.round = r.round
             AND (mr.male_id = r.registration_id OR mr.female_id = r.registration_id)
            LEFT JOIN registration pr
              ON pr.registration_id = CASE
                   WHEN mr.male_id = r.registration_id THEN mr.female_id
                   ELSE mr.male_id
                 END
            LEFT JOIN student ps ON ps.student_id = pr.student_id
            WHERE r.round = %s
            ORDER BY mr.rank NULLS LAST, r.registration_id
            """,
            (round_no,),
        )
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    recipients = []
    for row in rows:
        if row["gender"] is True:
            role = "female"
        elif row["gender"] is False:
            role = "male"
        else:
            role = None
        recipients.append(
            {
                "role": role,
                "registration_id": row["registration_id"],
                "receiver_name": row["name"],
                "receiver_phone": row["phone"],
                "rank": row["rank"],
                "partner_name": row["partner_name"],
                "partner_phone": row["partner_phone"],
            }
        )
    return recipients


def already_sent_ids(conn, round_no: int) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT registration_id
            FROM match_message
            WHERE round = %s AND success = TRUE
            """,
            (round_no,),
        )
        return {row[0] for row in cur.fetchall()}


def insert_message_log(conn, row: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO match_message (
                round, rank, registration_id, role,
                receiver_phone, sender_phone, message_body,
                success, http_status, api_code, api_response,
                msg_group_id, msg_type, block_cnt, fail_cnt,
                success_cnt, test_yn, error_text
            ) VALUES (
                %(round)s, %(rank)s, %(registration_id)s, %(role)s,
                %(receiver_phone)s, %(sender_phone)s, %(message_body)s,
                %(success)s, %(http_status)s, %(api_code)s, %(api_response)s,
                %(msg_group_id)s, %(msg_type)s, %(block_cnt)s, %(fail_cnt)s,
                %(success_cnt)s, %(test_yn)s, %(error_text)s
            )
            """,
            row,
        )
    conn.commit()


def empty_log(
    round_no: int,
    person: dict,
    sender: str,
    receiver: str,
    message: str,
    test_yn: str,
    error_text: str,
) -> dict[str, Any]:
    return {
        "round": round_no,
        "rank": person["rank"],
        "registration_id": person["registration_id"],
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

    with db.connect_reader() as reader:
        recipients = fetch_recipients(reader, round_no)
        sent = set() if force else already_sent_ids(reader, round_no)
    if not recipients:
        print(f"{round_no}차 접수가 없습니다.")
        return
    matched = sum(1 for person in recipients if person["rank"] is not None)
    print(f"{round_no}차 접수 {len(recipients)}명 (매칭 {matched}명, 미매칭 {len(recipients) - matched}명)")

    # 리더 엔드포인트는 쓰기를 거부하므로 발송 기록(match_message)은 라이터에 남긴다.
    with (nullcontext() if dry_run else db.connect()) as conn:
        ok = fail = skip = 0
        for person in recipients:
            place = f"#{person['rank']}" if person["rank"] is not None else "미매칭"
            label = f"{round_no}차 {place} {person['receiver_name']}({person['registration_id']})"
            if person["registration_id"] in sent:
                skip += 1
                print(f"건너뜀(이미 성공) {label}")
                continue
            if person["role"] is None:
                fail += 1
                print(f"실패 {label}: 성별이 없어 발송 기록을 남길 수 없음")
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
                    "rank": person["rank"],
                    "registration_id": person["registration_id"],
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
                sent.add(person["registration_id"])
                print(f"성공 {label} → {receiver} (group={parsed['msg_group_id']})")
            else:
                fail += 1
                print(f"실패 {label} → {receiver}: {error_text}")

        print(f"완료: 성공 {ok}, 실패 {fail}, 건너뜀 {skip} (접수 {len(recipients)}명)")


def main() -> None:
    parser = argparse.ArgumentParser(description="해당 차수 접수자 전원에게 문자를 발송한다.")
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
