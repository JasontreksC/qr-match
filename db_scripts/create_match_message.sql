CREATE TABLE match_message (
  message_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  round smallint NOT NULL,
  rank integer,                          -- 미매칭 안내 문자는 순위가 없다
  registration_id text NOT NULL REFERENCES registration(registration_id) ON DELETE CASCADE,
  role text NOT NULL CHECK (role IN ('male', 'female')),
  receiver_phone text NOT NULL,
  sender_phone text NOT NULL,
  message_body text NOT NULL,
  success boolean NOT NULL,
  http_status integer,
  api_code text,
  api_response text,
  msg_group_id text,
  msg_type text,
  block_cnt text,
  fail_cnt text,
  success_cnt text,
  test_yn text,
  error_text text,
  sent_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_match_message_result ON match_message (round, rank);
CREATE INDEX idx_match_message_registration ON match_message (registration_id);
CREATE INDEX idx_match_message_success ON match_message (round, registration_id, success);
