"""Blind A/B studies: construction, judging state, votes and results.

See docs/design-ab-studies.md. A study copies everything it shows from the two
runs, so later run deletion or suite changes never alter it.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import random
import sqlite3

from app.benchmarking import ab_stats
from app.benchmarking.evaluators import strip_think_blocks
from app.storage import pack_text

SCOPES = {"open_ended": "Open-ended questions", "all": "All paired questions"}
MAX_PAIRS = 1000
NAME_LIMIT = 120
COMMENT_LIMIT = 500
CHOICES = ("left", "right", "tie", "both_bad")
# Outcomes without a usable answer. Truncated or repetitive answers are real
# answers a user would see, so they stay in studies.
NO_ANSWER = {"endpoint_error", "unsupported_context", "cancelled", "missing_answer", "evaluator_error"}


class StudyError(ValueError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _connect():
    from app import config as app_config
    from app.storage import DETECT_TYPES

    db = sqlite3.connect(str(app_config.DATABASE_PATH), detect_types=DETECT_TYPES, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def _rows(db, query, params=()):
    return [dict(row) for row in db.execute(query, params).fetchall()]


def _one(db, query, params=()):
    row = db.execute(query, params).fetchone()
    return dict(row) if row else None


# --- Runs ------------------------------------------------------------------------

def run_suite(config_json):
    """(comparison name, loadable key) of a run's quality suite."""
    try:
        config = json.loads(config_json or "{}")
    except (TypeError, ValueError):
        config = {}
    name = config.get("name") or "rigorous"
    if name.startswith("usecase:") and config.get("slug") and config.get("version"):
        from app.benchmarking.usecase_suites import make_key

        return name, make_key(config["slug"], config["version"])
    return name, name


def run_label(row):
    return f"{row.get('model_name') or 'deleted model'} · run #{row['id']}"


def candidate_runs():
    """Completed runs with quality answers, newest first."""
    db = _connect()
    try:
        rows = _rows(db, """
            SELECT tr.id, tr.created_at, tr.quality_config_json, m.name AS model_name,
                   (SELECT COUNT(*) FROM test_results t WHERE t.run_id = tr.id) AS answers,
                   (SELECT COUNT(*) FROM test_results t WHERE t.run_id = tr.id
                       AND t.quality_outcome = 'recorded') AS recorded
              FROM test_runs tr LEFT JOIN models m ON m.id = tr.model_id
             WHERE tr.status = 'completed'
             ORDER BY tr.id DESC LIMIT 500""")
    finally:
        db.close()
    out = []
    for row in rows:
        if not row["answers"]:
            continue
        name, _key = run_suite(row["quality_config_json"])
        out.append({"id": row["id"], "label": run_label(row), "suite": name, "answers": row["answers"],
                    "recorded": row["recorded"], "created_at": row["created_at"]})
    return out


def _load_run(db, run_id):
    run = _one(db, """SELECT tr.id, tr.status, tr.quality_config_json, m.name AS model_name
                        FROM test_runs tr LEFT JOIN models m ON m.id = tr.model_id WHERE tr.id = ?""",
               (run_id,))
    if run is None:
        raise StudyError(f"Run {run_id} does not exist")
    if run["status"] != "completed":
        raise StudyError(f"Run {run_id} is not completed")
    rows = _rows(db, """SELECT test_id, category, prompt, response, request_ok, quality_metadata_json,
                               quality_outcome
                          FROM test_results WHERE run_id = ? ORDER BY question_index, id""", (run_id,))
    answers = {}
    for row in rows:
        try:
            record = json.loads(row["quality_metadata_json"] or "{}")
        except (TypeError, ValueError):
            continue
        fp = record.get("fingerprint")
        if not fp or fp in answers:
            continue
        answers[fp] = {
            "question_id": row["test_id"], "category": row["category"], "prompt": row["prompt"] or "",
            "family": record.get("family") or row["test_id"], "scope": record.get("scope"),
            "multi_turn": bool((record.get("metadata") or {}).get("protocol")),
            "outcome": row["quality_outcome"] or record.get("outcome"),
            "answer": strip_think_blocks(row["response"] or "").strip() if row["request_ok"] else "",
        }
    run["answers"] = answers
    run["suite_name"], run["suite_key"] = run_suite(run["quality_config_json"])
    return run


def _usable(item, scope):
    if not item["answer"] or item["outcome"] in NO_ANSWER or item["multi_turn"]:
        return False
    return scope == "all" or item["scope"] == "open_ended"


def _suite_questions(*keys):
    from app.benchmarking.quality_suite import fingerprint
    from app.benchmarking.suites import get_suite

    for key in dict.fromkeys(keys):
        try:
            return {fingerprint(q): q for q in get_suite(key).load()}
        except Exception:  # noqa: BLE001 - a removed suite only loses instructions and criteria
            continue
    return {}


def build_pairs(run_a, run_b, scope):
    shared = [fp for fp, item in run_a["answers"].items()
              if fp in run_b["answers"] and _usable(item, scope) and _usable(run_b["answers"][fp], scope)]
    if len(shared) > MAX_PAIRS:
        # A deterministic sample across the whole suite, not its first categories.
        shared = sorted(shared, key=lambda fp: hashlib.sha256(fp.encode()).hexdigest())[:MAX_PAIRS]
    questions = _suite_questions(run_a["suite_key"], run_b["suite_key"]) if shared else {}
    pairs = []
    for fp in shared:
        a, b = run_a["answers"][fp], run_b["answers"][fp]
        q = questions.get(fp)
        criteria = reference = None
        if q is not None and q.evaluator == "open_ended" and isinstance(q.expected, dict):
            criteria, reference = q.expected.get("criteria"), q.expected.get("reference")
        pairs.append({
            "question_id": a["question_id"], "fingerprint": fp, "category": a["category"],
            "family": a["family"], "prompt": q.prompt if q is not None else a["prompt"],
            "system_prompt": q.system_prompt if q is not None else None,
            "criteria": criteria, "reference": reference,
            "answer_a": a["answer"], "answer_b": b["answer"],
            "outcome_a": a["outcome"], "outcome_b": b["outcome"],
        })
    pairs.sort(key=lambda p: (p["category"], p["question_id"]))
    return pairs


def create_study(name, run_a, run_b, scope, user_id):
    name = (name or "").strip()
    if scope not in SCOPES:
        raise StudyError("Unknown scope")
    if run_a == run_b:
        raise StudyError("Choose two different runs")
    db = _connect()
    try:
        a, b = _load_run(db, run_a), _load_run(db, run_b)
        if a["suite_name"] != b["suite_name"]:
            raise StudyError(f"The runs used different suites ({a['suite_name']} and {b['suite_name']}); "
                             "a study compares answers to the same questions")
        pairs = build_pairs(a, b, scope)
        if not pairs:
            raise StudyError("The runs have no answered questions in common for this scope"
                             + (". Choose all paired questions or runs of an open-ended suite."
                                if scope == "open_ended" else "."))
        label_a, label_b = run_label(a), run_label(b)
        name = name[:NAME_LIMIT] or f"{label_a} vs {label_b}"
        cursor = db.execute(
            """INSERT INTO ab_studies (name, run_a, run_b, label_a, label_b, suite_name, scope, status,
                                       created_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)""",
            (name, run_a, run_b, label_a, label_b, a["suite_name"], scope, user_id, _now()))
        study_id = cursor.lastrowid
        db.executemany(
            """INSERT INTO ab_pairs (study_id, position, question_id, fingerprint, category, family, prompt,
                                     system_prompt, criteria_json, reference, answer_a, answer_b,
                                     outcome_a, outcome_b)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(study_id, i, p["question_id"], p["fingerprint"], p["category"], p["family"],
              pack_text(p["prompt"]), pack_text(p["system_prompt"]) if p["system_prompt"] else None,
              json.dumps(p["criteria"]) if p["criteria"] else None,
              pack_text(p["reference"]) if p["reference"] else None,
              pack_text(p["answer_a"]), pack_text(p["answer_b"]), p["outcome_a"], p["outcome_b"])
             for i, p in enumerate(pairs)])
        db.commit()
        return study_id
    finally:
        db.close()


# --- Studies ---------------------------------------------------------------------

def list_studies():
    db = _connect()
    try:
        return _rows(db, """
            SELECT s.*, u.username AS created_by_name,
                   (SELECT COUNT(*) FROM ab_pairs p WHERE p.study_id = s.id) AS pairs,
                   (SELECT COUNT(*) FROM ab_votes v JOIN ab_pairs p ON p.id = v.pair_id
                     WHERE p.study_id = s.id) AS votes,
                   (SELECT COUNT(*) FROM ab_judges j WHERE j.study_id = s.id) AS judges
              FROM ab_studies s LEFT JOIN users u ON u.id = s.created_by
             ORDER BY s.id DESC""")
    except sqlite3.OperationalError:
        return []
    finally:
        db.close()


def get_study(study_id):
    db = _connect()
    try:
        return _one(db, """SELECT s.*, u.username AS created_by_name FROM ab_studies s
                             LEFT JOIN users u ON u.id = s.created_by WHERE s.id = ?""", (study_id,))
    finally:
        db.close()


def _pair(row):
    row = dict(row)
    row["criteria"] = json.loads(row.pop("criteria_json") or "null")
    return row


def study_pairs(study_id):
    db = _connect()
    try:
        return [_pair(r) for r in db.execute(
            "SELECT * FROM ab_pairs WHERE study_id = ? ORDER BY position", (study_id,)).fetchall()]
    finally:
        db.close()


def get_pair(study_id, pair_id):
    db = _connect()
    try:
        row = db.execute("SELECT * FROM ab_pairs WHERE study_id = ? AND id = ?", (study_id, pair_id)).fetchone()
        return _pair(row) if row else None
    finally:
        db.close()


def set_status(study_id, status):
    if status not in {"open", "closed"}:
        raise StudyError("Unknown status")
    db = _connect()
    try:
        db.execute("UPDATE ab_studies SET status = ? WHERE id = ?", (status, study_id))
        db.commit()
    finally:
        db.close()


def delete_study(study_id):
    db = _connect()
    try:
        if db.execute("SELECT 1 FROM ab_judges WHERE study_id = ? AND status = 'running'",
                      (study_id,)).fetchone():
            raise StudyError("Cancel the running judge before deleting the study")
        db.execute("DELETE FROM ab_studies WHERE id = ?", (study_id,))
        db.commit()
    finally:
        db.close()


def contestant_models(study):
    """Model ids (rows and identifiers) of the two runs, while the runs exist."""
    db = _connect()
    try:
        rows = _rows(db, """SELECT m.id, m.model_id FROM test_runs tr JOIN models m ON m.id = tr.model_id
                             WHERE tr.id IN (?, ?)""", (study["run_a"] or 0, study["run_b"] or 0))
    finally:
        db.close()
    return {r["id"] for r in rows}, {r["model_id"] for r in rows}


# --- Judges ----------------------------------------------------------------------

def judges(study_id):
    db = _connect()
    try:
        return _rows(db, "SELECT * FROM ab_judges WHERE study_id = ? ORDER BY id", (study_id,))
    finally:
        db.close()


def start_judge(study_id, model, spawn=None):
    """Start (or resume) judging a study with a configured model; returns the judge id."""
    from app.benchmarking import pairwise_judge
    from app.services.url_guard import validate_endpoint

    validate_endpoint(model["base_url"])
    config = pairwise_judge.judge_config(model)
    db = _connect()
    try:
        if _one(db, "SELECT id FROM ab_studies WHERE id = ?", (study_id,)) is None:
            raise StudyError("Study not found")
        total = db.execute("SELECT COUNT(*) FROM ab_pairs WHERE study_id = ?", (study_id,)).fetchone()[0]
        existing = _one(db, """SELECT id, status FROM ab_judges
                                WHERE study_id = ? AND model_id = ? AND model_identifier = ? AND revision = ?
                                ORDER BY id DESC LIMIT 1""",
                        (study_id, model["id"], model["model_id"], pairwise_judge.REVISION))
        if existing and existing["status"] == "running":
            raise StudyError("This judge is already running")
        if existing:
            judge_id = existing["id"]
            db.execute("""UPDATE ab_judges SET status = 'running', error = NULL, started_at = ?,
                                 finished_at = NULL, total = ?, model_name = ? WHERE id = ?""",
                       (_now(), total, model["name"], judge_id))
        else:
            judge_id = db.execute(
                """INSERT INTO ab_judges (study_id, model_id, model_name, model_identifier, revision, status,
                                         total, started_at)
                   VALUES (?, ?, ?, ?, ?, 'running', ?, ?)""",
                (study_id, model["id"], model["name"], model["model_id"], pairwise_judge.REVISION, total,
                 _now())).lastrowid
        db.commit()
    finally:
        db.close()
    (spawn or pairwise_judge.start_judge_thread)(judge_id, config)
    return judge_id


def cancel_judge(study_id, judge_id):
    db = _connect()
    try:
        db.execute("""UPDATE ab_judges SET status = 'cancelled', finished_at = ?
                       WHERE id = ? AND study_id = ? AND status = 'running'""", (_now(), judge_id, study_id))
        db.commit()
    finally:
        db.close()


def judge_status(judge_id):
    db = _connect()
    try:
        row = db.execute("SELECT status FROM ab_judges WHERE id = ?", (judge_id,)).fetchone()
        return row[0] if row else None
    finally:
        db.close()


def pending_pairs(judge_id):
    db = _connect()
    try:
        return [_pair(r) for r in db.execute(
            """SELECT p.* FROM ab_pairs p JOIN ab_judges j ON j.study_id = p.study_id
                WHERE j.id = ? AND NOT EXISTS (
                      SELECT 1 FROM ab_judgments x WHERE x.judge_id = j.id AND x.pair_id = p.id
                         AND x.verdict IS NOT NULL)
                ORDER BY p.position""", (judge_id,)).fetchall()]
    finally:
        db.close()


def _refresh_counts(db, judge_id):
    db.execute("""UPDATE ab_judges
                     SET done = (SELECT COUNT(*) FROM ab_judgments WHERE judge_id = ? AND verdict IS NOT NULL),
                         errors = (SELECT COUNT(*) FROM ab_judgments WHERE judge_id = ? AND verdict IS NULL)
                   WHERE id = ?""", (judge_id, judge_id, judge_id))


def store_judgment(judge_id, pair_id, outcome):
    db = _connect()
    try:
        consistent = outcome.get("consistent")
        db.execute(
            """INSERT INTO ab_judgments (judge_id, pair_id, first, second, verdict, consistent,
                                         reason_first, reason_second, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (judge_id, pair_id) DO UPDATE SET
                   first = excluded.first, second = excluded.second, verdict = excluded.verdict,
                   consistent = excluded.consistent, reason_first = excluded.reason_first,
                   reason_second = excluded.reason_second, error = excluded.error""",
            (judge_id, pair_id, outcome.get("first"), outcome.get("second"), outcome.get("verdict"),
             None if consistent is None else int(consistent), outcome.get("reason_first"),
             outcome.get("reason_second"), outcome.get("error")))
        _refresh_counts(db, judge_id)
        db.commit()
    finally:
        db.close()


def finish_judge(judge_id, stopped=None):
    db = _connect()
    try:
        _refresh_counts(db, judge_id)
        row = _one(db, "SELECT done, errors, total FROM ab_judges WHERE id = ?", (judge_id,))
        if row is None:
            return
        if stopped or (row["errors"] and not row["done"]):
            status = "failed"
        else:
            status = "completed"
        plural = "pair" if row["errors"] == 1 else "pairs"
        error = stopped or (f"{row['errors']} {plural} had judge errors; start the judge again to retry them."
                            if row["errors"] else None)
        # A cancelled judge keeps its status; only a running one finishes here.
        db.execute("""UPDATE ab_judges SET status = ?, error = ?, finished_at = ?
                       WHERE id = ? AND status = 'running'""", (status, error, _now(), judge_id))
        db.commit()
    finally:
        db.close()


def judgments(judge_id):
    db = _connect()
    try:
        return _rows(db, "SELECT * FROM ab_judgments WHERE judge_id = ?", (judge_id,))
    finally:
        db.close()


# --- Votes -----------------------------------------------------------------------

def vote_progress(study_id, user_id):
    db = _connect()
    try:
        total = db.execute("SELECT COUNT(*) FROM ab_pairs WHERE study_id = ?", (study_id,)).fetchone()[0]
        voted = db.execute("""SELECT COUNT(*) FROM ab_votes v JOIN ab_pairs p ON p.id = v.pair_id
                               WHERE p.study_id = ? AND v.user_id = ?""", (study_id, user_id)).fetchone()[0]
        return voted, total
    finally:
        db.close()


def next_pair(study_id, user_id, skip=None, rng=None):
    """The pair with the fewest votes this user has not voted on, with random sides.

    Returns ``(pair, shown_left)`` or ``(None, None)`` when the user is done."""
    rng = rng or random.SystemRandom()
    db = _connect()
    try:
        rows = _rows(db, """
            SELECT p.id, (SELECT COUNT(*) FROM ab_votes v WHERE v.pair_id = p.id) AS votes
              FROM ab_pairs p
             WHERE p.study_id = ?
               AND NOT EXISTS (SELECT 1 FROM ab_votes v WHERE v.pair_id = p.id AND v.user_id = ?)""",
                     (study_id, user_id))
    finally:
        db.close()
    if not rows:
        return None, None
    choices = [r for r in rows if r["id"] != skip] or rows
    fewest = min(r["votes"] for r in choices)
    chosen = rng.choice([r for r in choices if r["votes"] == fewest])
    return get_pair(study_id, chosen["id"]), rng.choice(("a", "b"))


def record_vote(study_id, pair_id, user_id, choice, shown_left, comment=""):
    if choice not in CHOICES:
        raise StudyError("Choose left, right, tie or both bad")
    if shown_left not in {"a", "b"}:
        raise StudyError("Invalid answer order")
    comment = (comment or "").strip()
    if len(comment) > COMMENT_LIMIT:
        raise StudyError(f"Comments are limited to {COMMENT_LIMIT} characters")
    right = "b" if shown_left == "a" else "a"
    verdict = {"left": shown_left, "right": right}.get(choice, choice)
    db = _connect()
    try:
        study = _one(db, "SELECT status FROM ab_studies WHERE id = ?", (study_id,))
        if study is None:
            raise StudyError("Study not found")
        if study["status"] != "open":
            raise StudyError("This study is closed for voting")
        if not db.execute("SELECT 1 FROM ab_pairs WHERE id = ? AND study_id = ?", (pair_id, study_id)).fetchone():
            raise StudyError("This pair does not belong to the study")
        try:
            db.execute("""INSERT INTO ab_votes (pair_id, user_id, verdict, shown_left, comment, created_at)
                          VALUES (?, ?, ?, ?, ?, ?)""",
                       (pair_id, user_id, verdict, shown_left, comment or None, _now()))
        except sqlite3.IntegrityError as error:
            raise StudyError("You have already voted on this pair") from error
        db.commit()
        return verdict
    finally:
        db.close()


def votes(study_id):
    db = _connect()
    try:
        return _rows(db, """SELECT v.pair_id, v.verdict, v.shown_left, v.comment, v.created_at
                              FROM ab_votes v JOIN ab_pairs p ON p.id = v.pair_id
                             WHERE p.study_id = ? ORDER BY v.id""", (study_id,))
    finally:
        db.close()


# --- Results ---------------------------------------------------------------------

def results(study_id, pairs=None):
    pairs = pairs if pairs is not None else study_pairs(study_id)
    all_votes = votes(study_id)
    human = ab_stats.human_items(pairs, all_votes)
    lengths = {p["id"]: (len(p["answer_a"]), len(p["answer_b"])) for p in pairs}
    out = {
        "revision": ab_stats.REVISION,
        "pairs": len(pairs),
        "people": {**ab_stats.preference(human), "votes": len(all_votes),
                   "both_bad": sum(v["verdict"] == "both_bad" for v in all_votes),
                   "comments": sum(bool(v["comment"]) for v in all_votes)},
        "judges": [],
    }
    for judge in judges(study_id):
        rows = judgments(judge["id"])
        items = ab_stats.judge_items(pairs, rows)
        out["judges"].append({
            **judge,
            "preference": ab_stats.preference(items),
            "diagnostics": ab_stats.judge_diagnostics(items, lengths),
            "agreement": ab_stats.agreement(items, human),
            "by_pair": {r["pair_id"]: r for r in rows},
        })
    out["human_by_pair"] = {i["pair_id"]: i for i in human}
    return out


def export(study_id):
    """Portable study record: anonymous votes, judgments and both answers per pair."""
    study = get_study(study_id)
    pairs = study_pairs(study_id)
    summary = results(study_id, pairs)
    by_pair = {}
    for vote in votes(study_id):
        by_pair.setdefault(vote["pair_id"], []).append({"verdict": vote["verdict"], "comment": vote["comment"]})
    fields = ("question_id", "fingerprint", "category", "family", "prompt", "system_prompt", "criteria",
              "reference", "answer_a", "answer_b", "outcome_a", "outcome_b")
    return {
        "kind": "ab_study", "revision": ab_stats.REVISION,
        "study": {k: study[k] for k in ("id", "name", "label_a", "label_b", "suite_name", "scope", "status",
                                         "created_at")},
        "people": summary["people"],
        "judges": [{k: j[k] for k in ("model_name", "model_identifier", "revision", "status", "done", "errors",
                                      "preference", "diagnostics", "agreement")} for j in summary["judges"]],
        "pairs": [{**{k: p[k] for k in fields}, "votes": by_pair.get(p["id"], []),
                   "judgments": [{"judge": j["model_name"], **{k: j["by_pair"][p["id"]][k] for k in (
                       "first", "second", "verdict", "consistent", "reason_first", "reason_second", "error")}}
                                 for j in summary["judges"] if p["id"] in j["by_pair"]]}
                  for p in pairs],
    }
