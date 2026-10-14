-- 미매칭 참가자 문자도 match_message에 남긴다.
-- rank는 매칭된 사람만 채우고, 미매칭이면 NULL이다.
-- 여러 번 실행해도 안전하다.

ALTER TABLE match_message DROP CONSTRAINT IF EXISTS match_message_result_fkey;
ALTER TABLE match_message ALTER COLUMN rank DROP NOT NULL;
