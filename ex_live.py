"""ex_pair_score에 쌓이는 방향 평가를 점수 순으로 보여주는 로컬 웹 페이지.

중요:
  - Aurora 리더(AURORA_READER_ENDPOINT)로만 연결한다. db.connect_reader()가
    default_transaction_read_only=on 과 conn.read_only=True를 걸어서 쓰기가 거부된다.
  - 이름, 전화번호, 학번은 조회하지 않는다. 차트에 나오는 글은 평가에 쓰인 이상형/매력 원문뿐이다.

실행:
  python ex_live.py
  python ex_live.py --port 8770 --open
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import webbrowser
from collections import Counter
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from psycopg import errors as pg_errors

import db
from score_functions.ex import NOT_REQUESTED, SCORE_BY_LABEL, denominator_k
from score_functions.ex_traits import CRITERIA, CRITERION_KO

ROOT = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(ROOT, "ex_live.html")
KST = ZoneInfo("Asia/Seoul")
BINS = 10
TOP_N = 30
RECENT_N = 8
CURVE_DIGITS = 4


def direction_from_criteria(criteria: dict, k: float) -> tuple[float | None, int, list[dict]]:
    """direction_score와 같은 식. 요구 없음은 빼고, 점수 = 합계 / (n + k). 요구가 0개면 점수가 없다."""
    chosen = []
    for key in CRITERIA:
        item = (criteria or {}).get(key) or {}
        label = item.get("label")
        if not label or label == NOT_REQUESTED:
            continue
        score = item.get("score")
        if score is None:
            score = SCORE_BY_LABEL.get(label)
        if score is None:
            continue
        chosen.append((float(score), {
            "name": CRITERION_KO.get(key, key),
            "label": label,
            "score": round(float(score), 4),
        }))
    n = len(chosen)
    if n == 0:
        return None, 0, []
    total = sum(score for score, _item in chosen)
    return total / (n + k), n, [item for _score, item in chosen]


def _clip(text: str | None, limit: int = 90) -> str:
    flat = " ".join((text or "").split())
    if not flat:
        return "(적지 않음)"
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _mask_endpoint(endpoint: str) -> str:
    parts = endpoint.split(".")
    if len(parts) >= 4:
        return f"{parts[0]}…{'.'.join(parts[-3:])}"
    return endpoint


def _as_criteria(value) -> dict:
    if isinstance(value, str):
        return json.loads(value)
    return value or {}


class LiveStore:
    """리더를 주기적으로 읽어 스냅샷을 메모리에 둔다. HTTP 요청은 DB에 닿지 않는다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, dict] = {}
        self._watermark: tuple[datetime, str] | None = None
        self._k = denominator_k()
        self.ready = threading.Event()
        self.snapshot: dict = {
            "ok": False,
            "error": None,
            "count": 0,
            "defined": 0,
            "curve": [],
            "histogram": [],
            "minutes": [],
            "top": [],
            "recent": [],
        }
        self.error: str | None = None

    def run(self, interval: float) -> None:
        conn = None
        while True:
            started = time.monotonic()
            try:
                if conn is None or conn.closed:
                    conn = self._connect()
                added = self._fetch_new(conn)
                self.error = None
                self._publish(added)
            except Exception as exc:  # noqa: BLE001 — 다음 주기에 다시 연결한다.
                self.error = f"{type(exc).__name__}: {exc}"
                self._publish(0)
                if conn is not None and not conn.closed:
                    conn.close()
                conn = None
            finally:
                self.ready.set()
            time.sleep(max(0.2, interval - (time.monotonic() - started)))

    def _connect(self):
        conn = db.connect_reader()
        with conn.cursor() as cur:
            cur.execute("SELECT current_setting('transaction_read_only'), pg_is_in_recovery()")
            read_only, is_replica = cur.fetchone()
        conn.rollback()
        if read_only != "on":
            conn.close()
            raise RuntimeError("데이터베이스 연결이 읽기 전용(transaction_read_only=on)이 아닙니다.")
        self._read_only = read_only == "on"
        self._is_replica = bool(is_replica)
        self._reader = _mask_endpoint(db.aurora_reader_host())
        return conn

    def _fetch_new(self, conn) -> int:
        sql = """
            SELECT cache_key, want_text, have_text, criteria,
                   jsonb_array_length(samples), resampled, created_at
            FROM ex_pair_score
        """
        params: tuple = ()
        if self._watermark is not None:
            ts, key = self._watermark
            sql += " WHERE created_at > %s OR (created_at = %s AND cache_key > %s)"
            params = (ts, ts, key)
        sql += " ORDER BY created_at, cache_key"
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                fetched = cur.fetchall()
        except pg_errors.UndefinedTable:
            conn.rollback()
            raise RuntimeError("ex_pair_score 테이블이 없습니다.") from None
        conn.rollback()
        for cache_key, want, have, criteria, calls, resampled, created_at in fetched:
            score, n, detail = direction_from_criteria(_as_criteria(criteria), self._k)
            self._rows[cache_key] = {
                "want": want or "",
                "have": have or "",
                "score": None if score is None else round(score, 6),
                "n": n,
                "calls": int(calls or 0),
                "resampled": bool(resampled),
                "created_at": created_at,
                "criteria": detail,
            }
        if fetched:
            self._watermark = (fetched[-1][6], fetched[-1][0])
        return len(fetched)

    def _publish(self, added: int) -> None:
        rows = list(self._rows.values())
        defined = [row for row in rows if row["score"] is not None]
        defined.sort(key=lambda row: (-row["score"], -row["n"], row["created_at"]))
        scores = [row["score"] for row in defined]
        now = datetime.now(KST)
        recent_cut = now - timedelta(seconds=60)
        per_minute = sum(1 for row in rows if row["created_at"] >= recent_cut)
        snapshot = {
            "ok": self.error is None,
            "error": self.error,
            "read_only": getattr(self, "_read_only", False),
            "replica": getattr(self, "_is_replica", False),
            "reader": getattr(self, "_reader", ""),
            "k": self._k,
            "fetched_at": now.strftime("%H:%M:%S"),
            "count": len(rows),
            "defined": len(defined),
            "undefined": len(rows) - len(defined),
            "resampled": sum(1 for row in rows if row["resampled"]),
            "calls": sum(row["calls"] for row in rows),
            "added": added,
            "mean": round(statistics.fmean(scores), 4) if scores else None,
            "median": round(statistics.median(scores), 4) if scores else None,
            "maximum": scores[0] if scores else None,
            "per_minute": per_minute,
            "curve": [round(score, CURVE_DIGITS) for score in scores],
            "histogram": _histogram(scores),
            "minutes": _minutes(rows, now),
            "top": [_public(row, rank) for rank, row in enumerate(defined[:TOP_N], start=1)],
            "recent": [
                _public(row, 0)
                for row in sorted(rows, key=lambda item: item["created_at"], reverse=True)[:RECENT_N]
            ],
        }
        with self._lock:
            self.snapshot = snapshot
            self.error = None if snapshot["ok"] else self.error
        if added:
            print(f"[ex-live] 저장 {snapshot['count']}건 (+{added}, 최근 1분 {per_minute}건)", file=sys.stderr)

    def payload(self) -> dict:
        with self._lock:
            return self.snapshot


def _histogram(scores: list[float]) -> list[dict]:
    width = 1 / BINS
    counts = [0] * BINS
    for score in scores:
        index = min(BINS - 1, max(0, int(score / width)))
        counts[index] += 1
    return [
        {"lo": round(i * width, 2), "hi": round((i + 1) * width, 2), "n": counts[i]}
        for i in range(BINS)
    ]


def _minutes(rows: list[dict], now: datetime) -> list[dict]:
    if not rows:
        return []
    end = now.astimezone(KST).replace(second=0, microsecond=0)
    start = end - timedelta(minutes=29)
    counts: Counter[datetime] = Counter()
    for row in rows:
        minute = row["created_at"].astimezone(KST).replace(second=0, microsecond=0)
        if start <= minute <= end:
            counts[minute] += 1
    out = []
    cursor = start
    while cursor <= end:
        out.append({"t": cursor.strftime("%H:%M"), "n": counts[cursor]})
        cursor += timedelta(minutes=1)
    return out


def _public(row: dict, rank: int) -> dict:
    created = row["created_at"].astimezone(KST)
    return {
        "rank": rank,
        "score": row["score"],
        "n": row["n"],
        "calls": row["calls"],
        "resampled": row["resampled"],
        "want": _clip(row["want"]),
        "have": _clip(row["have"]),
        "at": created.strftime("%H:%M:%S"),
        "criteria": row["criteria"],
    }


class LiveHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        message = fmt % args
        if "/api/snapshot" in message and " 200 " in f" {message} ":
            return
        sys.stderr.write(f"[ex-live] {self.address_string()} - {message}\n")

    def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        store: LiveStore = self.server.store  # type: ignore[attr-defined]
        if path in ("/", "/index.html"):
            if not os.path.exists(HTML_PATH):
                self.send_error(HTTPStatus.NOT_FOUND, "ex_live.html not found")
                return
            with open(HTML_PATH, "rb") as handle:
                self._send(handle.read(), "text/html; charset=utf-8")
            return
        if path == "/api/snapshot":
            self._send(
                json.dumps(store.payload(), ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ex_pair_score 실시간 점수 차트 (Aurora 리더, 읽기 전용)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--interval", type=float, default=2.0, help="리더를 다시 읽는 간격(초)")
    parser.add_argument("--open", action="store_true", help="서버가 준비되면 브라우저를 연다.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = LiveStore()
    threading.Thread(target=store.run, args=(args.interval,), daemon=True).start()
    store.ready.wait(30)

    server = ThreadingHTTPServer((args.host, args.port), LiveHandler)
    server.store = store  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}"
    print("=" * 68)
    print("  ex 평가 실시간 차트 (READ-ONLY)")
    print(f"  - Aurora Reader : {db.aurora_reader_host()}")
    print(f"  - 로컬 웹 주소  : {url}")
    print("=" * 68)
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
