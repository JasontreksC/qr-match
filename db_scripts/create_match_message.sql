CREATE TABLE match_message (
  message_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  round smallint NOT NULL,
  rank integer NOT NULL,
  student_id text NOT NULL REFERENCES student(student_id) ON DELETE CASCADE,
  role text NOT NULL CHECK (role IN ('male', 'female')),
  receiver_phone text NOT NULL,
  sender_phone text NOT NULL,
  message_body text NOT NULL,
  success boolean NOT NULL,
  http_status integer,
  api_code text,
  api_response text,
  remaining_balance text,
  provider_send_count text,
  error_text text,
  sent_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT match_message_result_fkey
    FOREIGN KEY (round, rank) REFERENCES match_result(round, rank) ON DELETE CASCADE
);

CREATE INDEX idx_match_message_result ON match_message (round, rank);
CREATE INDEX idx_match_message_student ON match_message (student_id);
CREATE INDEX idx_match_message_success ON match_message (round, student_id, success);
