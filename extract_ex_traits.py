"""모든 registration의 ex_want / ex_have를 7개 기준으로 추출해 ex_trait을 채운다.

  python extract_ex_traits.py [옵션]

옵션
  (없음)           점검만 합니다. 읽기 전용 연결로 추출 대상 수만 세고, LLM을 부르지도 쓰지도 않습니다.
  --execute        실제로 추출해서 ex_trait에 저장합니다. 여러 번 실행해도 안전합니다(UPSERT).
                   이미 최신인 행은 건드리지 않고, 아래 행만 (다시) 추출합니다.
                     - ex_trait 행이 없음
                     - source_text가 현재 원문과 다름
                     - extraction_ok가 false
                     - prompt_version이 현재와 다름
  --round N        해당 차수(1 또는 2)만 대상으로 합니다. 기본은 모든 차수.
  --limit N        추출 대상이 많아도 앞에서부터 N건만 처리합니다. (시험 실행용)
  --concurrency N  동시에 부르는 LLM 호출 수. 기본 4.
  --force          최신인 행도 모두 다시 추출합니다.

필요한 환경변수
  AURORA_*          이 저장소의 .env (db.py 참고)
  OPENAI_API_KEY    --execute에서만 필요
  OPENAI_MODEL      --execute에서만 필요. 날짜가 고정된 스냅샷 이름. latest 별칭은 거부합니다.
  (선택) OPENAI_REASONING_EFFORT, OPENAI_TEMPERATURE

제약
  - 쓰기는 Aurora의 ex_trait 한 테이블에만 합니다. 접수 데이터(registration, ex_want, ex_have 등)는 읽기만 합니다.
  - ex_trait 테이블이 없으면 QRious/migrations/aurora/02_ex_trait.sql을 먼저 적용해야 합니다.
"""

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg
from psycopg import errors as pg_errors
from psycopg.types.json import Jsonb

import db
from score_functions.ex_traits import (
    CRITERIA,
    CRITERION_KO,
    EXTRACT_PROMPT_VERSION,
    extract_traits,
)
from score_functions.llm import get_client, resolve_model

MIGRATION_HINT = "QRious/migrations/aurora/02_ex_trait.sql"

SOURCE_SQL = """
SELECT r.registration_id, r.round, 'want'::text AS side, ew.charm AS text
FROM registration r
JOIN ex_want ew ON ew.registration_id = r.registration_id
UNION ALL
SELECT r.registration_id, r.round, 'have'::text AS side, eh.charm AS text
FROM registration r
JOIN ex_have eh ON eh.registration_id = r.registration_id
"""

_COLUMNS = ", ".join(CRITERIA)
UPSERT_SQL = f"""
INSERT INTO ex_trait (
  registration_id, side, {_COLUMNS},
  extraction_ok, source_text, error_message, raw_response, model, prompt_version, attempts, extracted_at
) VALUES (%s, %s, {", ".join(["%s"] * len(CRITERIA))}, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (registration_id, side) DO UPDATE SET
  {", ".join(f"{key} = EXCLUDED.{key}" for key in CRITERIA)},
  extraction_ok = EXCLUDED.extraction_ok,
  source_text = EXCLUDED.source_text,
  error_message = EXCLUDED.error_message,
  raw_response = EXCLUDED.raw_response,
  model = EXCLUDED.model,
  prompt_version = EXCLUDED.prompt_version,
  attempts = EXCLUDED.attempts,
  extracted_at = now()
"""

REASON_LABEL = {
    "missing": "추출 결과 없음",
    "source_changed": "원문이 바뀜",
    "failed": "이전 추출 실패",
    "prompt_changed": "프롬프트 버전이 다름",
    "forced": "--force",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ex_want/ex_have를 7개 기준으로 추출해 ex_trait을 채웁니다.")
    parser.add_argument("--execute", action="store_true", help="실제로 추출해서 저장합니다. 없으면 점검만 합니다.")
    parser.add_argument("--round", type=int, choices=(1, 2), help="해당 차수만 대상으로 합니다.")
    parser.add_argument("--limit", type=int, help="앞에서부터 N건만 처리합니다.")
    parser.add_argument("--concurrency", type=int, default=4, help="동시 LLM 호출 수 (기본 4).")
    parser.add_argument("--force", action="store_true", help="최신인 행도 모두 다시 추출합니다.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit은 1 이상이어야 합니다.")
    if args.concurrency < 1:
        parser.error("--concurrency는 1 이상이어야 합니다.")
    return args


def table_state(conn) -> tuple[bool, list[str]]:
    """(ex_trait 테이블이 있는가, 빠진 컬럼 이름들)"""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.ex_trait') IS NOT NULL")
        exists = cur.fetchone()[0]
        if not exists:
            return False, []
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'ex_trait'
            """
        )
        have = {row[0] for row in cur.fetchall()}
    return True, [key for key in CRITERIA if key not in have]


def fetch_candidates(conn, round_no: int | None, with_traits: bool) -> list[dict]:
    """원문이 있는 (registration, side)와 현재 ex_trait 상태."""
    if with_traits:
        trait_cols = "t.registration_id IS NOT NULL, t.source_text, t.extraction_ok, t.prompt_version"
        join = "LEFT JOIN ex_trait t ON t.registration_id = s.registration_id AND t.side = s.side"
    else:
        trait_cols = "false, NULL::text, NULL::boolean, NULL::text"
        join = ""
    sql = f"""
        SELECT s.registration_id, s.round, s.side, s.text, {trait_cols}
        FROM ({SOURCE_SQL}) s
        {join}
        WHERE s.text IS NOT NULL AND btrim(s.text) <> ''
          AND (%(round)s::smallint IS NULL OR s.round = %(round)s::smallint)
        ORDER BY s.round, s.registration_id, s.side
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"round": round_no})
        rows = cur.fetchall()
    return [
        {
            "registration_id": r[0],
            "round": r[1],
            "side": r[2],
            "text": r[3],
            "has_row": r[4],
            "source_text": r[5],
            "extraction_ok": r[6],
            "prompt_version": r[7],
        }
        for r in rows
    ]


def reason_to_extract(row: dict, force: bool) -> str | None:
    if not row["has_row"]:
        return "missing"
    if row["source_text"] != row["text"]:
        return "source_changed"
    if not row["extraction_ok"]:
        return "failed"
    if row["prompt_version"] != EXTRACT_PROMPT_VERSION:
        return "prompt_changed"
    if force:
        return "forced"
    return None


def distribution(conn) -> list[tuple]:
    """ex_trait에 쌓인 추출 결과의 side별 분포."""
    counts = ", ".join(f"count({key})" for key in CRITERIA)
    all_null = " AND ".join(f"{key} IS NULL" for key in CRITERIA)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT side, count(*), count(*) FILTER (WHERE extraction_ok),
                   count(*) FILTER (WHERE extraction_ok AND {all_null}), {counts}
            FROM ex_trait GROUP BY side ORDER BY side
            """
        )
        return cur.fetchall()


def print_distribution(conn) -> None:
    rows = distribution(conn)
    if not rows:
        return
    print("\n== ex_trait 분포 (기준별로 값이 있는 행 수) ==")
    header = ["side", "전체", "성공", "성공·전부 없음"] + [CRITERION_KO[key] for key in CRITERIA]
    print("  " + " | ".join(header))
    for row in rows:
        print("  " + " | ".join(str(value) for value in row))


def save(conn, target: dict, outcome, model: str) -> None:
    traits = outcome.traits
    with conn.cursor() as cur:
        cur.execute(
            UPSERT_SQL,
            (
                target["registration_id"],
                target["side"],
                *[traits.get(key) for key in CRITERIA],
                outcome.ok,
                target["text"],
                outcome.error,
                Jsonb(outcome.raw),
                model,
                EXTRACT_PROMPT_VERSION,
                outcome.attempts,
            ),
        )
    conn.commit()


def main() -> int:
    args = parse_args()
    mode = "실행" if args.execute else "점검만 (쓰기 없음)"
    print(f"== ex_trait 추출 [{mode}] ==")
    print(f"대상 DB: Aurora {db.aurora_host()}")
    print(f"추출 프롬프트 버전: {EXTRACT_PROMPT_VERSION}")

    if args.execute:
        model = resolve_model()
        get_client()  # OPENAI_API_KEY 확인
        print(f"모델: {model}")
    else:
        try:
            print(f"모델: {resolve_model()}")
        except RuntimeError as err:
            print(f"모델: (--execute 때 필요) {err}")

    conn = db.connect(read_only=not args.execute)
    try:
        exists, missing_columns = table_state(conn)
        if missing_columns:
            print(
                f"\n오류: ex_trait에 컬럼이 빠져 있습니다: {', '.join(missing_columns)}\n"
                f"{MIGRATION_HINT}을 적용해서 컬럼과 제약을 맞춰 주세요. (여러 번 실행해도 안전합니다)"
            )
            return 1
        if not exists:
            print(f"\n알림: ex_trait 테이블이 아직 없습니다. 모든 원문을 추출 대상으로 셉니다. ({MIGRATION_HINT} 적용 필요)")
            if args.execute:
                print(f"\n오류: 먼저 {MIGRATION_HINT}을 Aurora에 적용해 주세요.")
                return 1

        candidates = fetch_candidates(conn, args.round, with_traits=exists)
        conn.commit()

        targets = []
        reasons: Counter = Counter()
        for row in candidates:
            reason = reason_to_extract(row, args.force)
            if reason:
                targets.append({**row, "reason": reason})
                reasons[reason] += 1

        by_side = Counter(row["side"] for row in candidates)
        scope = f"{args.round}차" if args.round else "모든 차수"
        print(f"\n[{scope}] 원문이 있는 항목: {len(candidates)}건 (want {by_side['want']}, have {by_side['have']})")
        print(f"이미 최신: {len(candidates) - len(targets)}건")
        print(f"추출 필요: {len(targets)}건")
        for key, label in REASON_LABEL.items():
            if reasons[key]:
                print(f"  - {label}: {reasons[key]}건")

        if args.limit is not None and len(targets) > args.limit:
            print(f"--limit {args.limit}: 앞에서부터 {args.limit}건만 처리합니다.")
            targets = targets[: args.limit]

        if not args.execute:
            if exists:
                print_distribution(conn)
            print("\n점검만 했습니다. 아무것도 쓰지 않았습니다. 실제로 추출하려면 --execute를 붙이세요.")
            return 0

        if not targets:
            print("\n추출할 항목이 없습니다.")
            print_distribution(conn)
            return 0

        failed: list[tuple[dict, str]] = []
        skipped = 0
        done = 0
        pool = ThreadPoolExecutor(max_workers=args.concurrency)
        try:
            futures = {pool.submit(extract_traits, t["side"], t["text"], model): t for t in targets}
            for future in as_completed(futures):
                target = futures[future]
                done += 1
                outcome = future.result()
                try:
                    save(conn, target, outcome, model)
                except pg_errors.ForeignKeyViolation:
                    conn.rollback()
                    skipped += 1
                    print(f"[{done}/{len(targets)}] {target['registration_id']} {target['side']}: 접수가 사라져 건너뜀")
                    continue
                state = "성공" if outcome.ok else "실패"
                filled = sum(1 for key in CRITERIA if outcome.traits.get(key))
                print(
                    f"[{done}/{len(targets)}] {target['registration_id']} {target['side']}: "
                    f"{state} (시도 {outcome.attempts}회, 기준 {filled}개)"
                )
                if not outcome.ok:
                    failed.append((target, outcome.error or ""))
        except KeyboardInterrupt:
            print("\n중단합니다. 이미 저장된 행은 그대로이고, 다시 실행하면 이어서 합니다.")
            pool.shutdown(wait=False, cancel_futures=True)
            return 130
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        print(f"\n완료: 성공 {done - skipped - len(failed)}건, 실패 {len(failed)}건, 건너뜀 {skipped}건")
        for target, error in failed:
            print(f"  실패 {target['registration_id']} {target['side']}: {error[:200]}")
        if failed:
            print("실패한 행은 extraction_ok=false로 저장했습니다. 다시 실행하면 재시도합니다.")
        print_distribution(conn)
        return 1 if failed else 0
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, psycopg.Error) as err:
        print(f"\n오류: {err}")
        sys.exit(1)
