-- Users
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    password_hash TEXT,
    role TEXT DEFAULT 'user',
    oidc_sub TEXT UNIQUE,
    display_name TEXT,
    token_version INTEGER NOT NULL DEFAULT 0,
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
    temperature REAL NOT NULL DEFAULT 0,
    reasoning_effort TEXT,
    b300_metrics_model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Test Runs
CREATE TABLE IF NOT EXISTS test_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
    -- Quality
    weighted_score REAL DEFAULT 0.0,   -- difficulty-weighted average score
    scored_questions INTEGER DEFAULT 0, -- questions that produced a model answer
    error_count INTEGER DEFAULT 0,      -- transport failures, excluded from scores
    -- Performance
    workers INTEGER DEFAULT 1,          -- concurrent workers used for the run
    duration_ms REAL DEFAULT 0.0,       -- wall clock for the question phase
    latency_p50_ms REAL,
    latency_p95_ms REAL,
    latency_p99_ms REAL,
    ttft_p50_ms REAL,
    ttft_p95_ms REAL,
    output_tokens_per_sec REAL,         -- effective throughput for this run
    perf_json TEXT,                     -- serialised PerfReport, when the perf suite ran
    quality_config_json TEXT,
    quality_json TEXT,
    run_options_json TEXT,                -- full start parameters, so the queue can re-dispatch a pending run
    decoding_config_json TEXT,            -- model settings captured when the run is submitted
    metrics_config_json TEXT,             -- B300 telemetry selectors, without credentials
    plan_id INTEGER,                      -- set when the run belongs to a run_plans group
    repeat_group_id INTEGER,              -- first run of a repeat group; NULL for single runs
    repeat_index INTEGER,                 -- 0-based position in the group; also its model seed
    repeat_count INTEGER,                 -- planned runs in the group
    canary_id INTEGER,                    -- set for scheduled canary runs
    canary_status TEXT,                   -- ok | warning | alert once the canary run is evaluated
    canary_json TEXT,                     -- canary evaluation detail
    FOREIGN KEY (model_id) REFERENCES models(id),
    FOREIGN KEY (created_by) REFERENCES users(id),
    FOREIGN KEY (plan_id) REFERENCES run_plans(id)
);

-- Run plans: a named group of runs scheduled together (now or later)
CREATE TABLE IF NOT EXISTS run_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    created_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    scheduled_at TEXT,                    -- UTC ISO; NULL = as soon as the queue is free
    status TEXT DEFAULT 'active',         -- 'active' | 'cancelled'
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- Use-case suites uploaded by administrators; every changed upload is a new immutable version
CREATE TABLE IF NOT EXISTS usecase_suites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL,
    version INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    owner TEXT,
    source_yaml TEXT NOT NULL,            -- the uploaded document, packed like other payloads
    suite_hash TEXT NOT NULL,
    question_count INTEGER NOT NULL,
    warnings_json TEXT,
    created_by INTEGER,
    created_at TIMESTAMP,
    archived INTEGER NOT NULL DEFAULT 0,  -- hidden from the run form; runs keep working
    UNIQUE (slug, version),
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- Deployment snapshots, canaries and monitoring events (docs/design-deployment-monitoring.md)
CREATE TABLE IF NOT EXISTS deployment_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    run_id INTEGER REFERENCES test_runs(id) ON DELETE SET NULL,  -- NULL for on-demand checks
    revision TEXT NOT NULL,
    created_at TIMESTAMP,
    ok INTEGER NOT NULL,
    error TEXT,
    fingerprint TEXT,
    snapshot_json TEXT NOT NULL,
    previous_id INTEGER REFERENCES deployment_checks(id) ON DELETE SET NULL,
    changed TEXT                          -- NULL | configuration | deployment
);
CREATE INDEX IF NOT EXISTS idx_deployment_checks_model ON deployment_checks(model_id, id);

CREATE TABLE IF NOT EXISTS canaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    suite TEXT NOT NULL,
    max_concurrency INTEGER NOT NULL DEFAULT 4,
    interval_hours INTEGER NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    baseline_run_id INTEGER REFERENCES test_runs(id) ON DELETE SET NULL,
    next_run_at TEXT,                     -- UTC ISO
    webhook_url TEXT,
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitor_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    canary_id INTEGER REFERENCES canaries(id) ON DELETE SET NULL,
    run_id INTEGER REFERENCES test_runs(id) ON DELETE SET NULL,
    check_id INTEGER REFERENCES deployment_checks(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,               -- alert | warning | info
    title TEXT NOT NULL,
    detail_json TEXT,
    notified TEXT,                        -- NULL, 'sent' or the delivery error
    created_at TIMESTAMP,
    acknowledged_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    acknowledged_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_monitor_events_open ON monitor_events(acknowledged_at, severity);

-- Blind A/B studies: pairwise comparison of two runs of one suite (docs/design-ab-studies.md).
-- Studies copy everything they show, so they outlive the runs they compare.
CREATE TABLE IF NOT EXISTS ab_studies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    run_a INTEGER,                        -- informational; the run may be deleted later
    run_b INTEGER,
    label_a TEXT NOT NULL,
    label_b TEXT NOT NULL,
    suite_name TEXT NOT NULL,
    scope TEXT NOT NULL,                  -- open_ended | all
    status TEXT NOT NULL DEFAULT 'open',  -- open | closed
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ab_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id INTEGER NOT NULL REFERENCES ab_studies(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    question_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    category TEXT NOT NULL,
    family TEXT NOT NULL,
    prompt TEXT NOT NULL,                 -- packed
    system_prompt TEXT,                   -- packed
    criteria_json TEXT,
    reference TEXT,                       -- packed
    answer_a TEXT NOT NULL,               -- packed, reasoning removed
    answer_b TEXT NOT NULL,
    outcome_a TEXT,
    outcome_b TEXT
);
CREATE INDEX IF NOT EXISTS idx_ab_pairs_study ON ab_pairs(study_id, position);

CREATE TABLE IF NOT EXISTS ab_judges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id INTEGER NOT NULL REFERENCES ab_studies(id) ON DELETE CASCADE,
    model_id INTEGER,                     -- the judge's models row; may be deleted later
    model_name TEXT NOT NULL,
    model_identifier TEXT NOT NULL,
    revision TEXT NOT NULL,
    status TEXT NOT NULL,                 -- running | completed | failed | cancelled | interrupted
    done INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    started_at TIMESTAMP,
    finished_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ab_judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    judge_id INTEGER NOT NULL REFERENCES ab_judges(id) ON DELETE CASCADE,
    pair_id INTEGER NOT NULL REFERENCES ab_pairs(id) ON DELETE CASCADE,
    first TEXT,                           -- a | b | tie: verdict with A shown first
    second TEXT,                          -- verdict with B shown first
    verdict TEXT,                         -- a | b | tie, NULL on judge_error
    consistent INTEGER,
    reason_first TEXT,
    reason_second TEXT,
    error TEXT,
    UNIQUE (judge_id, pair_id)
);

CREATE TABLE IF NOT EXISTS ab_votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id INTEGER NOT NULL REFERENCES ab_pairs(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,  -- votes of deleted users stay anonymous
    verdict TEXT NOT NULL,                -- a | b | tie | both_bad
    shown_left TEXT NOT NULL,             -- a | b
    comment TEXT,
    created_at TIMESTAMP,
    UNIQUE (pair_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_ab_votes_pair ON ab_votes(pair_id);

-- Decision profiles and decision records (docs/design-decision-dashboard.md).
-- Records copy the evaluation they were made on, so later runs never rewrite them.
CREATE TABLE IF NOT EXISTS decision_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    gates_json TEXT NOT NULL,
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decision_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES decision_profiles(id) ON DELETE CASCADE,
    model_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
    model_name TEXT NOT NULL,
    decision TEXT NOT NULL,               -- approved | conditional | rejected
    note TEXT NOT NULL,
    revision TEXT NOT NULL,
    evaluation_json TEXT NOT NULL,
    fingerprint TEXT,                     -- the model's deployment fingerprint when decided
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_decision_records_profile ON decision_records(profile_id, model_id, id);

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
    -- Pass/fail is decided by the question's own threshold, not a global 0.5,
    -- so it is stored rather than re-derived in every query.
    passed INTEGER DEFAULT 0,
    pass_threshold REAL DEFAULT 1.0,
    difficulty TEXT DEFAULT 'medium',
    weight REAL DEFAULT 1.0,
    -- Per-request performance
    latency_ms REAL,
    ttft_ms REAL,
    request_ok INTEGER DEFAULT 1,       -- 0 when the call failed at the transport layer
    quality_scored INTEGER,             -- NULL on legacy rows; excludes evaluator/unsupported errors
    quality_metadata_json TEXT,
    quality_outcome TEXT,               -- queryable without decompressing audit payloads
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES test_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_test_results_run ON test_results(run_id);
CREATE INDEX IF NOT EXISTS idx_test_results_run_category ON test_results(run_id, category);

-- Each sweep cell is durable without rewriting a growing report after every request.
CREATE TABLE IF NOT EXISTS performance_cells (
    run_id INTEGER NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    effort TEXT NOT NULL,
    context_tokens INTEGER NOT NULL,
    concurrency INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (run_id, effort, context_tokens, concurrency)
);

-- Active benchmark sessions (for progress tracking)
CREATE TABLE IF NOT EXISTS benchmark_progress (
    run_id INTEGER PRIMARY KEY,
    current_test TEXT,
    current_index INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    status_message TEXT,
    phase TEXT DEFAULT 'quality',
    FOREIGN KEY (run_id) REFERENCES test_runs(id)
);
