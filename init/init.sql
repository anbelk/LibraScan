CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    hash TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    img_url TEXT NOT NULL,
    model_answer TEXT NOT NULL
);

CREATE TABLE user_answers (
    id SERIAL PRIMARY KEY,
    prediction_hash TEXT,
    user_answer TEXT NOT NULL,
    answered_at TIMESTAMP NOT NULL
);
