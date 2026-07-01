-- Users
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    password_hash TEXT,
    role TEXT DEFAULT 'user',
    oidc_sub TEXT UNIQUE,
    display_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- LLM Models
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    api_key TEXT NOT NULL,
    model_id TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Test Runs
CREATE TABLE IF NOT EXISTS test_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    status TEXT DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    total_questions INTEGER DEFAULT 0,
    passed_questions INTEGER DEFAULT 0,
    avg_score REAL DEFAULT 0.0,
    error_message TEXT,
    created_by INTEGER,
    test_suite_hash TEXT,
    total_prompt_tokens INTEGER DEFAULT 0,
    total_completion_tokens INTEGER DEFAULT 0,
    FOREIGN KEY (model_id) REFERENCES models(id),
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- Test Results
CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    test_id TEXT NOT NULL,
    category TEXT NOT NULL,
    prompt TEXT,
    response TEXT,
    score REAL DEFAULT 0.0,
    detail TEXT,
    evaluator TEXT,
    question_index INTEGER,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES test_runs(id)
);

-- Active benchmark sessions (for progress tracking)
CREATE TABLE IF NOT EXISTS benchmark_progress (
    run_id INTEGER PRIMARY KEY,
    current_test TEXT,
    current_index INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    status_message TEXT,
    FOREIGN KEY (run_id) REFERENCES test_runs(id)
);
