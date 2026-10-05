# Design: LLM judge and blind A/B studies

Status: accepted (recommended decisions), 2026-10-05. Item 3 of the evaluation roadmap.

## Problem

Every existing quality signal is deterministic: exact answers, schemas, simulations and tool calls. That is right for capability measurement, but most company use is open-ended. People ask for emails, summaries, explanations, advice and rewrites, and no answer key separates a good reply from a mediocre one. Two questions follow:

1. *Which of two deployments do our users prefer on our kind of requests?* This is the decision for swapping a model, a quantization or a serving configuration.
2. *Can an LLM judge stand in for people on these comparisons, and how far can we trust it?*

## Decisions

1. **Pairwise preference is the unit.**
   - An **A/B study** compares two completed runs of the *same* quality suite, question by question.
   - LLM judges and people both give pairwise verdicts on the same pairs, so their results aggregate the same way and their agreement can be measured.
   - Absolute 1–10 rubric scores are out of scope. They are less reliable and harder to calibrate than pairwise choices.
2. **Open-ended questions are a question kind, not an evaluator score.**
   - A question with `evaluator: open_ended` carries `expected: {criteria: [...], reference?: "..."}`.
   - In a run it is executed as usual (bounded protocol, final-answer recovery). Its completed answer gets the outcome **`recorded`** and the scope **`open_ended`**.
   - It is not "scored": it is excluded from capability, compliance, pass counts and paired score comparisons. Truncated, empty or failed answers keep their usual failure outcomes.
   - `EVALUATOR_VERSIONS` and every existing fingerprint are unchanged.
   - Use-case suites may contain open-ended questions, and a suite made only of open-ended questions completes normally.
3. **Built-in open-ended suite.** `assistant-open` (revision `assistant-open-v1`): 30 workplace assistant requests in English, Czech and German across six categories:
   - writing
   - summarising
   - explaining
   - advising
   - transforming data
   - reviewing text or code
   
   Each request has explicit criteria. It gives studies a useful default before teams upload their own.
4. **Study construction.**
   - An admin picks run A, run B and a scope:
     - *open-ended questions* (the default when any exist)
     - *all paired questions*
   - Pairs are questions with the same fingerprint where both requests returned an answer, capped at 1,000 per study.
   - Each study copies prompts, system prompts, criteria, references and both final answers (reasoning blocks removed), so it survives run deletion and later suite changes.
   - Run labels are stored for the results page.
5. **LLM judge (`pairwise-judge-v1`).**
   - **Judge model:** any configured model can judge. If it is also a contestant, the page warns about self-preference.
   - **Prompt:** the user request, the assistant instructions, the criteria (general defaults when absent), the optional reference and two answers. Each answer is truncated to 24,000 characters, and truncation is noted to the judge.
   - **Output:** strict JSON `{"reason": "...", "winner": "1"|"2"|"tie"}` at temperature 0.
   - **Position bias:** every pair is judged in both orders. A=1, tie=½, B=0 per order; the mean decides the verdict (A, B or tie). A pair is *position-consistent* when both orders agree.
   - **Failures:** malformed output or a failed request is a recorded `judge_error`, excluded from statistics and visible on the page.
   - **Execution:** judging runs in the background (4 workers) and saves each judgment as it finishes. Several judge models can judge the same study. Starting a judge again resumes the missing pairs, and judges interrupted by a restart are marked *interrupted*.
6. **Blind human votes.**
   - Any signed-in user can vote in an open study.
   - **What the voter sees:** the request (instructions collapsed), the criteria and two answers in random left/right order. Model names, run ids and statistics are not shown.
   - **Pair selection:** the pair with the fewest votes that the user hasn't voted on, with a random tie-break.
   - **Choices:** left, right, tie or both bad, plus an optional comment of up to 500 characters. There is one immutable vote per user and pair.
   - **Rendering:** answers are plain text with preserved whitespace; no HTML or Markdown is rendered.
   - **Deleted users:** their votes are kept anonymously.
7. **Statistics (`ab-stats-v1`), for each judge model and for people:**
   - **Preference for B:** wins = 1, ties = ½. Multiple human votes on a pair are averaged first.
   - **Interval:** a 95% bootstrap interval resampling families within categories (fixed seed, 2,000 draws).
   - **Test:** an exact two-sided sign test on decisive pairs.
   - **Verdict:** B or A is *preferred* when the interval excludes ½ and p < 0.05; otherwise *no clear preference*.
   - **Breakdowns:** per category; outcome counts (A / B / tie / both bad).
   - **Judge diagnostics:** position-consistency rate, and the share of decisive verdicts won by the longer answer (length bias).
   - **Judge vs people:** on pairs with both a judge verdict and human votes, the raw agreement and Cohen's κ over {A, B, tie}. Ties among votes on a pair count as a tie.
8. **Access.** Admins create, close, reopen and delete studies and start or cancel judges. Every signed-in user can vote and see results. Results never show who voted.

## Data model

```
ab_studies   id, name, run_a, run_b, label_a, label_b, suite_name, scope, status(open|closed), created_by, created_at
ab_pairs     id, study_id→cascade, position, question_id, fingerprint, category, family, prompt*, system_prompt*,
             criteria_json, reference*, answer_a*, answer_b*, outcome_a, outcome_b          (* packed text)
ab_judges    id, study_id→cascade, model_id, model_name, model_identifier, revision, status(running|completed|
             failed|cancelled|interrupted), done, total, errors, error, started_at, finished_at
ab_judgments id, judge_id→cascade, pair_id→cascade, first, second, verdict, consistent, reason_first,
             reason_second, error, UNIQUE(judge_id, pair_id)
ab_votes     id, pair_id→cascade, user_id→users ON DELETE SET NULL, verdict(a|b|tie|both_bad), shown_left(a|b),
             comment, created_at, UNIQUE(pair_id, user_id)
```

Runs referenced by a study may be deleted. The study keeps its copies and labels.

## UI

- **Navigation:** "Blind A/B", visible to all users.
- **`/studies`:**
  - a list with pair, vote and judge counts and status;
  - for admins, a create form (two completed runs of the same suite plus a scope);
  - a "Create blind A/B study" link from the compare page when exactly two runs are selected.
- **`/studies/{id}`:** results, with one card each for people and every judge, then:
  - per-category preferences
  - judge diagnostics and judge–human agreement
  - for admins: judge controls (model select, self-judging warning, start/resume/cancel, progress), close/reopen and delete
  - a pair browser that shows judge reasons (unblinded; results page only)
- **`/studies/{id}/vote`:** the blind voting page, with progress for the user and a "skip" link.

## Verification

`selftest_ab_studies.py`:

- **Run and suite handling:**
  - open-ended scoring outcome and scope, and summaries excluding it
  - an open-ended-only run completing
  - use-case `open_ended` validation
  - the built-in suite's validity and hash stability
- **Studies:**
  - study creation (same-suite rule, pairing by fingerprint, failed answers excluded, scope, cap, copied answers stripped of reasoning)
- **Judging:**
  - judge prompt contents and truncation
  - verdict parsing and both-order combination
  - background judging with a fake judge: progress, errors, resume, cancel, interruption marking
- **Voting:**
  - blind voting: assignment order, random sides mapped back correctly, one vote per user, closed studies
  - non-admin restrictions
- **Statistics and pages:**
  - statistics: preference, interval, sign test, κ and length bias on hand-computed cases
  - pages render, including the compare-page entry point

## Implementation notes

- **Excluded from studies:** multi-turn tasks (native-tool and simulated-action protocols), because their answers are transcripts rather than replies. Truncated and repetitive answers stay, because users would see them.
- **Unblinded pages:**
  - each pair has a page (`/studies/{id}/pairs/{pair}`) with both answers, votes, comments and both judge reasons;
  - the whole study downloads as JSON (`/studies/{id}/export.json`), with anonymous votes.
- **Judge failures:**
  - after eight consecutive judge errors the judge stops as *failed*;
  - a judge that finishes with some errors is *completed*, notes the errors, and can be resumed to retry them.
- **Endpoint load:** judging uses the judge's endpoint outside the run queue, so the page warns against judging on a deployment under performance test.

## Out of scope (v1)

- More than two runs per study (Bradley–Terry / Elo ranking across many models).
- Absolute rubric scoring.
- Rendering Markdown in answers.
- Calibrating judges against stored human labels from other studies.
- Multi-turn conversations.
