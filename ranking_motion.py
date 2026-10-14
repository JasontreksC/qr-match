"""시간 순서(1시간 단위) 학과별 접수 랭킹 모션 그래픽 로컬 웹 서버.

중요:
  - 데이터 수정이 절대 일어나지 않도록 .env의 AURORA_READER_ENDPOINT(읽기 전용 인스턴스)로만 연결한다.
  - db.connect_reader()를 통해 default_transaction_read_only=on 및 conn.read_only=True를 이중 강제한다.
  - 학생 개인식별정보(이름, 전화번호, 학번, 생년월일 등)는 조회·반환하지 않고 학과별·시간대별 집계만 다룬다.

실행:
  python ranking_motion.py
  python ranking_motion.py --port 8765
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
from urllib.parse import parse_qs, urlparse
import webbrowser
from zoneinfo import ZoneInfo

import db

ROOT = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(ROOT, "ranking_motion.html")
KST = ZoneInfo("Asia/Seoul")
WEEKDAY_KO = ("월", "화", "수", "목", "금", "토", "일")


def _format_hour_label(dt: datetime) -> dict[str, str]:
    end_dt = dt + timedelta(hours=1)
    wd = WEEKDAY_KO[dt.weekday()]
    return {
        "iso": dt.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "date": f"{dt.year}. {dt.month:02d}. {dt.day:02d}. ({wd})",
        "short_date": f"{dt.month:02d}/{dt.day:02d} ({wd})",
        "hour_range": f"{dt.hour:02d}:00 ~ {end_dt.hour:02d}:00",
        "hour_short": f"{dt.hour:02d}:00",
        "full": f"{dt.year}.{dt.month:02d}.{dt.day:02d} ({wd}) {dt.hour:02d}:00~{end_dt.hour:02d}:00",
    }


def _mask_endpoint(endpoint: str) -> str:
    """전체 호스트명을 노출하되 너무 길면 앞부분과 리전/도메인을 읽기 쉽게 표시."""
    parts = endpoint.split(".")
    if len(parts) >= 4:
        return f"{parts[0]}...{'.'.join(parts[-3:])}"
    return endpoint


def fetch_hourly_ranking_data(round_filter: int | None = None) -> dict:
    """Aurora Reader 인스턴스에서 학과 목록과 접수 시각(1시간 단위) 집계를 읽어온다."""
    reader_host = db.aurora_reader_host()

    with db.connect_reader() as conn:
        with conn.cursor() as cur:
            # 1) 읽기 전용 상태 확인
            cur.execute(
                "SELECT current_setting('transaction_read_only'), pg_is_in_recovery()"
            )
            tx_read_only, is_replica = cur.fetchone()
            if tx_read_only != "on":
                raise RuntimeError("데이터베이스 연결이 읽기 전용(transaction_read_only=on)이 아닙니다.")

            # 2) 전체 학과 목록 조회
            cur.execute(
                """
                SELECT major_id, name, COALESCE(short_name, name) AS short_name
                FROM major
                ORDER BY name ASC
                """
            )
            majors_raw = cur.fetchall()

            # 3) 접수별 최초 동의 시각(KST) + 학과 + 성별 + 차수 조회 (개인정보 제외)
            cur.execute(
                """
                WITH reg_time AS (
                    SELECT
                        r.registration_id,
                        r.round,
                        COALESCE(s.major_id, 'undeclared') AS major_id,
                        s.gender,
                        MIN(c.consented_at) AT TIME ZONE 'Asia/Seoul' AS registered_at
                    FROM registration r
                    JOIN student s ON s.student_id = r.student_id
                    JOIN consent c ON c.registration_id = r.registration_id
                    WHERE (%s::smallint IS NULL OR r.round = %s::smallint)
                    GROUP BY r.registration_id, r.round, s.major_id, s.gender
                )
                SELECT
                    major_id,
                    round,
                    gender,
                    registered_at,
                    date_trunc('hour', registered_at) AS hour_bucket
                FROM reg_time
                ORDER BY registered_at ASC, registration_id ASC
                """,
                (round_filter, round_filter),
            )
            reg_rows = cur.fetchall()

            # 4) 차수별 총 인원 요약
            cur.execute(
                """
                SELECT round, COUNT(*)
                FROM registration
                GROUP BY round
                ORDER BY round
                """
            )
            round_counts = {int(r): int(cnt) for r, cnt in cur.fetchall()}

    majors: list[dict] = []
    major_map: dict[str, dict] = {}
    for major_id, name, short_name in majors_raw:
        item = {
            "major_id": major_id,
            "name": name,
            "short_name": short_name,
        }
        majors.append(item)
        major_map[major_id] = item

    # 혹시 major 테이블에 없는 major_id가 접수에 있다면 보정
    for major_id, _round, _gender, _reg_at, _hb in reg_rows:
        if major_id not in major_map:
            item = {
                "major_id": major_id,
                "name": major_id,
                "short_name": major_id,
            }
            majors.append(item)
            major_map[major_id] = item

    now_kst = datetime.now(KST)
    if not reg_rows:
        return {
            "meta": {
                "fetched_at": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
                "reader_endpoint": _mask_endpoint(reader_host),
                "reader_endpoint_full": reader_host,
                "transaction_read_only": tx_read_only == "on",
                "is_replica": bool(is_replica),
                "round_filter": round_filter or "all",
                "round_counts": round_counts,
                "total_registrations": 0,
                "total_hours": 0,
                "active_hours": 0,
                "total_majors": len(majors),
                "participating_majors": 0,
            },
            "majors": majors,
            "frames": [],
        }

    # 시간 버킷별 이벤트 묶기
    min_hour: datetime = reg_rows[0][4]
    max_hour: datetime = reg_rows[-1][4]

    bucket_events: dict[datetime, list[tuple[str, bool, int]]] = {}
    for seq_idx, (major_id, _round, gender, _reg_at, hour_bucket) in enumerate(reg_rows, start=1):
        bucket_events.setdefault(hour_bucket, []).append((major_id, bool(gender), seq_idx))

    # 학과별 누적 상태
    totals: dict[str, int] = {m["major_id"]: 0 for m in majors}
    males: dict[str, int] = {m["major_id"]: 0 for m in majors}
    females: dict[str, int] = {m["major_id"]: 0 for m in majors}
    # 동점일 때 먼저 해당 인원수에 도달한 학과가 상위 랭크를 유지하도록 순서 기록
    last_reached_seq: dict[str, int] = {m["major_id"]: 10**9 for m in majors}
    first_reached_seq: dict[str, int] = {m["major_id"]: 10**9 for m in majors}
    major_name_order: dict[str, str] = {m["major_id"]: m["name"] for m in majors}

    frames: list[dict] = []

    # 시작 직전 0점 프레임 (재생 시 0에서 출발해 첫 시간대로 자연스럽게 차오르도록 구성)
    pre_hour = min_hour - timedelta(hours=1)
    pre_label = _format_hour_label(pre_hour)
    frames.append(
        {
            "frame_index": 0,
            "is_initial": True,
            "has_activity": False,
            "label": {
                **pre_label,
                "hour_range": "접수 시작 전 (0명)",
            },
            "total": 0,
            "male": 0,
            "female": 0,
            "delta_total": 0,
            "delta_male": 0,
            "delta_female": 0,
            "participating_majors": 0,
            "max_count": 0,
            "items": [],
        }
    )

    cur_hour = min_hour
    frame_idx = 1
    active_hours_count = 0
    cum_total = 0
    cum_male = 0
    cum_female = 0

    while cur_hour <= max_hour:
        events = bucket_events.get(cur_hour, [])
        deltas: dict[str, int] = {}
        delta_male_map: dict[str, int] = {}
        delta_female_map: dict[str, int] = {}
        hr_delta_total = 0
        hr_delta_male = 0
        hr_delta_female = 0

        for major_id, is_female, seq_idx in events:
            totals[major_id] += 1
            deltas[major_id] = deltas.get(major_id, 0) + 1
            last_reached_seq[major_id] = seq_idx
            if first_reached_seq[major_id] == 10**9:
                first_reached_seq[major_id] = seq_idx

            hr_delta_total += 1
            cum_total += 1
            if is_female:
                females[major_id] += 1
                delta_female_map[major_id] = delta_female_map.get(major_id, 0) + 1
                hr_delta_female += 1
                cum_female += 1
            else:
                males[major_id] += 1
                delta_male_map[major_id] = delta_male_map.get(major_id, 0) + 1
                hr_delta_male += 1
                cum_male += 1

        if hr_delta_total > 0:
            active_hours_count += 1

        # 현재 시간까지 1명 이상 접수한 학과들을 랭킹순 정렬
        active_major_ids = [mid for mid, cnt in totals.items() if cnt > 0]
        active_major_ids.sort(
            key=lambda mid: (
                -totals[mid],
                last_reached_seq[mid],
                first_reached_seq[mid],
                major_name_order[mid],
            )
        )

        items = []
        for rank_idx, mid in enumerate(active_major_ids, start=1):
            items.append(
                {
                    "major_id": mid,
                    "rank": rank_idx,
                    "count": totals[mid],
                    "male": males[mid],
                    "female": females[mid],
                    "delta": deltas.get(mid, 0),
                    "delta_male": delta_male_map.get(mid, 0),
                    "delta_female": delta_female_map.get(mid, 0),
                }
            )

        max_cnt = items[0]["count"] if items else 0
        frames.append(
            {
                "frame_index": frame_idx,
                "is_initial": False,
                "has_activity": hr_delta_total > 0,
                "label": _format_hour_label(cur_hour),
                "total": cum_total,
                "male": cum_male,
                "female": cum_female,
                "delta_total": hr_delta_total,
                "delta_male": hr_delta_male,
                "delta_female": hr_delta_female,
                "participating_majors": len(items),
                "max_count": max_cnt,
                "items": items,
            }
        )

        cur_hour += timedelta(hours=1)
        frame_idx += 1

    participating_majors = sum(1 for cnt in totals.values() if cnt > 0)

    return {
        "meta": {
            "fetched_at": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
            "reader_endpoint": _mask_endpoint(reader_host),
            "reader_endpoint_full": reader_host,
            "transaction_read_only": tx_read_only == "on",
            "is_replica": bool(is_replica),
            "round_filter": round_filter or "all",
            "round_counts": round_counts,
            "total_registrations": cum_total,
            "total_male": cum_male,
            "total_female": cum_female,
            "total_hours": len(frames) - 1,
            "active_hours": active_hours_count,
            "total_majors": len(majors),
            "participating_majors": participating_majors,
            "start_hour": _format_hour_label(min_hour)["full"],
            "end_hour": _format_hour_label(max_hour)["full"],
        },
        "majors": majors,
        "frames": frames,
    }


class RankingMotionHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[ranking-motion] {self.address_string()} - {fmt % args}\n")

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, path: str) -> None:
        if not os.path.exists(path):
            self.send_error(HTTPStatus.NOT_FOUND, "ranking_motion.html not found")
            return
        with open(path, "rb") as f:
            raw = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_html(HTML_PATH)
            return

        if parsed.path == "/api/rankings":
            qs = parse_qs(parsed.query)
            raw_round = (qs.get("round", ["all"])[0] or "all").strip().lower()
            round_filter: int | None = None
            if raw_round in ("1", "2"):
                round_filter = int(raw_round)
            try:
                data = fetch_hourly_ranking_data(round_filter=round_filter)
                self._send_json(data, status=HTTPStatus.OK)
            except Exception as exc:
                self._send_json(
                    {
                        "error": str(exc),
                        "message": "Aurora Reader 인스턴스 조회 중 오류가 발생했습니다.",
                    },
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aurora Reader(읽기 전용) 기반 1시간 단위 학과별 접수 랭킹 모션 그래픽 웹 서버"
    )
    parser.add_argument("--host", default="127.0.0.1", help="바인딩 호스트 (기본: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="바인딩 포트 (기본: 8765)")
    parser.add_argument(
        "--open",
        action="store_true",
        help="서버 시작 시 기본 브라우저를 자동으로 엽니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader_host = db.aurora_reader_host()
    url = f"http://{args.host}:{args.port}"
    print("=" * 68)
    print("  학과별 접수 랭킹 모션 그래픽 서버 (READ-ONLY)")
    print(f"  - Aurora Reader : {reader_host}")
    print(f"  - 로컬 웹 주소  : {url}")
    print("=" * 68)

    server = ThreadingHTTPServer((args.host, args.port), RankingMotionHandler)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
