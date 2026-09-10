import os
from dotenv import load_dotenv
import psycopg

load_dotenv()

try:
    # 각 참가자의 모든 정보를 가져오는 SQL믄
    with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE student CASCADE")

except Exception as e:
    print(e)


