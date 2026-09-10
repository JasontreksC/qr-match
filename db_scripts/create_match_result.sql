CREATE TABLE match_result (
  rank integer PRIMARY KEY,
  male_id text NOT NULL REFERENCES student(student_id) ON DELETE CASCADE,
  female_id text NOT NULL REFERENCES student(student_id) ON DELETE CASCADE,
  mbti_score double precision NOT NULL,
  tag_score double precision NOT NULL,
  ex_score double precision NOT NULL,
  final_score double precision NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT match_result_male_unique UNIQUE (male_id),
  CONSTRAINT match_result_female_unique UNIQUE (female_id)
);

CREATE INDEX match_result_final_score_idx ON match_result (final_score DESC);
