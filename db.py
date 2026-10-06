"""Aurora PostgreSQL 연결.

이 저장소의 .env에 있는 AURORA_* 환경 변수만 쓴다. (QRious의 .env.local은 읽지 않는다.)
Neon(DATABASE_URL)에는 접속하지 않는다.
"""

import os
import tempfile

import psycopg
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(
            f"{name}이(가) 설정되지 않았습니다. 이 저장소의 .env에 Aurora 연결 정보를 넣어 주세요."
        )
    return value


def _ssl_options() -> dict:
    mode = (os.getenv("AURORA_DB_SSL") or "").strip().lower()
    if mode == "disable":
        return {"sslmode": "disable"}
    ca = (os.getenv("AURORA_DB_SSL_CA") or "").strip()
    if ca:
        # PEM 내용을 그대로 넣거나 파일 경로를 넣을 수 있다. (QRious 앱과 같은 규칙)
        if "BEGIN CERTIFICATE" in ca:
            handle = tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False)
            handle.write(ca.replace("\\n", "\n"))
            handle.close()
            ca = handle.name
        return {"sslmode": "verify-full", "sslrootcert": ca}
    # 암호화는 하되 인증서는 검증하지 않는다. 검증하려면 AURORA_DB_SSL_CA를 지정한다.
    return {"sslmode": "require"}


def aurora_host() -> str:
    return _require("AURORA_WRITER_ENDPOINT")


def connect(*, read_only: bool = False) -> psycopg.Connection:
    """Aurora 라이터 엔드포인트에 연결한다. read_only=True이면 쓰기가 서버에서 거부된다."""
    port = int(_require("AURORA_DB_PORT"))
    conn = psycopg.connect(
        host=_require("AURORA_WRITER_ENDPOINT"),
        port=port,
        user=_require("AURORA_DB_USER"),
        password=_require("AURORA_DB_PASSWORD"),
        dbname=_require("AURORA_DB_NAME"),
        connect_timeout=60,  # Serverless v2가 깨어나는 시간을 고려한다.
        application_name="qr-match",
        **_ssl_options(),
    )
    if read_only:
        conn.read_only = True
    return conn
