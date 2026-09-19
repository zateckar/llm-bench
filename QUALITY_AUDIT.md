# Quality audit — 19 September 2026

The latest comparable database runs contain 1,380 answers across five models,
all using suite `dc1efed08e5f310e`. The stored results show substantial output-budget
and grading confounds. They do not establish that Gemma has stronger general
capability than Kimi or Qwen. Model names below are saved endpoint aliases, not
independently verified model identities.

| Run | Saved model name | Average with partial credit | Full passes | Output-limit failures |
|---|---|---:|---:|---:|
| 28 | Qwen3.8-27B.long | 69.38% | 185/276 | 72 |
| 29 | Qwen3.8-27B-FP8.long | 65.44% | 172/276 | 81 |
| 31 | Qwen3.8-27B-NVFP4.long | 64.29% | 172/276 | 88 |
| 33 | skoda-llm-primary-test | 79.65% | 206/276 | 38 |
| 34 | gemma4-31 | 82.59% | 198/276 | 1 |

Kimi already had more full passes than Gemma. Its saved advanced-coding and
code-generation category scores were both 100%, and reading comprehension was
100% versus Gemma's 50%. The overall average also includes partial credit for
prose-pattern checks and zeroes for truncated answers. Larger budgets require
fresh model calls; rescoring cannot reconstruct an answer that was never produced.

## Confirmed defects and changes

- **Incorrect word-ladder fixture:** `hit -> hot`, with `hot` in the word list,
  has length 2, not 0. The prompt's invalid `hit -> hot -> cog` example is fixed too.
- **Broken tar harness:** a function declared to accept `bytes` received a JSON
  list. A trusted helper now converts fixture values to bytes. The size field
  is corrected from 11 to 12 bytes; zero padding and a value above 2^64 are tested.
  The format follows the [GNU tar specification](https://www.gnu.org/software/tar/manual/html_node/Standard.html).
- **Wrong rounded constant:** Rydberg's constant rounds to 10,973,732, not
  10,973,731. The old relative tolerance hid the wrong key by accepting both.
  [NIST's value](https://physics.nist.gov/Pubs/AtSpec/node01.html) supports the correction.
- **Impossible letter constraint:** IF-09 required `Yesterday` while forbidding
  `a`. The required opening word is now explicitly exempt; the remaining paragraph
  must still obey the ban. The stored Kimi answer also contains `a` in `steady`,
  so fixing the impossible requirement does not make that complete answer valid.
- **Representation errors:** thousands-separated tungsten temperatures,
  superscript scientific notation, an equivalent Strassen complexity expression,
  and valid German compound wording no longer produce those false failures.
- **Prose-order and vocabulary errors:** S3 setup did not require the incidental
  textual order enforced by its grader. Specific ethics, summary, and premise
  correction patterns missed legitimate phrasing; those regressions are repaired.
- **Ambiguous output contract:** the replica tombstone explicitly requires
  `value:null`. Previously the deleted-value schema was easy to interpret differently.
- **Weak truth checks:** the water enthalpy check now requires a negative sign;
  the near-integer question requires the actual full integer and rejects literal
  equality. Atlantic trench naming no longer requires a disputed subdivision.
- **False numeric passes:** exact JSON and integer fixtures preserve integer
  precision above 2^53. JSON comparison defaults to exact numeric values, so
  a tiny constant cannot match zero through a fixed absolute tolerance.
- **Reasoning-only responses:** provider reasoning fields are retained as marked
  scratchpads, never promoted to final answers. Output-limit failures remain failures.

The read-only replay checked **990 unchanged, completed static responses** against
the revised graders. It recovered 11 full passes for run 33 and three for run 34,
plus partial-credit corrections in other runs. It deliberately skips changed
prompts, incomplete answers, generated tasks, and historically expanded code
fixtures that cannot be reproduced from a static question alone. These are
diagnostic comparisons, not replacement run scores.

## Reliability-v4

The suite now has **315 static questions and 322 questions with default generated
and interactive tasks**. `reliability_cases.py` authors 46 new questions, two in
each static category, with 179 code fixtures. They introduce interacting rules,
negative and boundary cases, evidence reconciliation, scoped authorization,
unsupported claims, conditional probabilities, optimization ties, and longer
record ledgers. Interactive preview now requires reconciling later records that
revoke apparent eligibility because ownership or pinning changed.

The default noninteractive quality cap is **16,384 output tokens**, including
provider reasoning when billed against that cap. It is uniform across models
and configurable through `--quality-max-tokens`, the run form, or a run plan.
Interactive actions have 4,096 tokens per turn and 32,768 total, with the existing
16-turn limit. These settings change the protocol and suite fingerprint. The
higher cap can increase runtime and token usage; it does not guarantee completion.

The capability headline excludes **124 legacy heuristic items** whose keyword,
regex, or prose-format checks cannot establish factual correctness. They remain
available as diagnostics and in explicitly labeled all-item averages. There are
**187 capability items and 11 creative-compliance items** under the default
configuration. Full-pass rates are reported separately from partial-credit means.
Creative constraints do not claim to measure artistic quality, and the new ethics
items measure application of stated principles rather than universal moral truth.

Difficulty labels remain author estimates. The new tasks and their answer keys
are public. Fresh, matched runs are needed to measure discrimination; no model's
identity or desired ranking enters question generation, reference answers, or grading.

## Reproduction and validation

```powershell
.venv/Scripts/python.exe audit_results.py data/bench.db --runs 28 29 31 33 34 --replay --output data/quality-audit-2026-09-19.json
.venv/Scripts/python.exe reliability_cases.py
.venv/Scripts/python.exe validate_suite.py --strict
.venv/Scripts/python.exe selftest_reliability.py
```

The audit opens SQLite in read-only mode and exports per-question original grades,
diagnostics, replay differences, and reasons for skipping replay. It does not
export endpoint credentials or full response text. Historical answers, grades,
and suite hashes remain unchanged.

Validation also includes `selftest_evaluators.py`, `selftest_challenges.py`,
`selftest_quality.py`, `selftest_perf.py`, `selftest_reports.py`,
`selftest_run_queue.py`, and `ruff check .`. Reliability tests verify all new
references, corrupt every structured answer field, reject alternative/duplicate
JSON answers, execute code in the actual sandbox, and check database-discovered
regressions. No live model calls are part of these tests.
