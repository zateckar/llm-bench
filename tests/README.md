# Test suites

Question files for the quality benchmark. Each YAML file in this directory is one
category; `test_loader.py` loads all of them, validates the fields, and produces
a stable `test_suite_hash` recorded on every run so a change to the suite shows up
in the run metadata.

## File format

A file is a list of question mappings:

```yaml
- id: FK-01                    # unique within the file
  category: Factual Knowledge  # groups results on the run page
  difficulty: medium           # easy | medium | hard | expert (sets default weight)
  prompt: "..."
  evaluator: contains_keywords # see evaluators.EVALUATORS
  expected:
    groups:
      - ["fluorite", "fluorspar"]
```

Optional fields (all validated, unknown keys are rejected):

| Field | Meaning |
|-------|---------|
| `id`, `category`, `prompt` | required |
| `evaluator` + `expected` | explicit scorer from `evaluators.EVALUATORS` |
| `criteria` [+ `must_not`] | shorthand for the `security_analysis` evaluator |
| `keywords` | shorthand for `contains_keywords` |
| `system_prompt` | sent with the prompt |
| `difficulty` | `easy`/`medium`/`hard`/`expert` — drives the default weight |
| `weight` | explicit weight, overriding the difficulty weight |
| `pass_threshold` | score needed to count as a pass (default 1.0) |
| `max_tokens` | per-question generation cap |

The loader is strict: a typo in an evaluator name, an unknown difficulty, or a
malformed `expected` value aborts the whole run with a clear message instead of
silently scoring wrong.

## Categories in this directory

`advanced_coding`, `agentic_use_cases`, `classification`, `code_generation`,
`creative_writing`, `ethical_reasoning`, `factual_knowledge`,
`instruction_following`, `logical_reasoning`, `long_context_coherence`,
`mathematical_reasoning`, `needle_retrieval`, `reading_comprehension`,
`security`, `summarization`, `terminal_algorithms`, `terminal_debugging`,
`terminal_file_operations`, `terminal_science`, `terminal_system_admin`,
`tool_using`, `translation`, `truthfulness`.

## Challenge revision

The suite contains 269 questions. Six categories use 56 `challenge-v2` items:

| Category | IDs | Count | Main demands |
|----------|-----|-------|--------------|
| Logical Reasoning | `LR2-*` | 12 | Enumerate consistent models, optimize schedules and assignments, track state |
| Mathematical Reasoning | `MR2-*` | 12 | Constrained counting, conditional probability, exact expectations, symmetry |
| Advanced Coding | `AC2-*` | 8 | Time windows, tombstones, cycles, negative values, idempotency, escaped wildcards |
| Agentic Use Cases | `AU2-*` | 8 | Resource limits, transaction visibility, retries, compensation, replica conflicts |
| Tool Using | `TU2-*` | 8 | Exact arguments, uncertain outcomes, concurrency, scope, and untrusted data |
| Reading Comprehension | `RC2-*` | 8 | Reconcile revisions, exceptions, overlapping intervals, and accounting rules |

These replace the 106 previous questions in those categories, rather than adding
more questions on top of easy ones. Other categories remain as baseline coverage.
The new IDs deliberately do not reuse retired IDs. The suite hash and prompt-based
cache fingerprint change automatically; historical results remain historical.

Prompts define all assumptions, ordering rules, output schemas, and rounding.
Structured answers use exact JSON comparison with no ignored fields; code items
require every fixture to pass. Difficulty comes from solving interacting constraints,
not obscure trivia, arbitrary prose wording, or unexplained precision requirements.
Expert reasoning items have an 8,192-token cap to allow working space.

Check answers and grading offline after edits:

```bash
uv run python validate_suite.py --strict
uv run python selftest_evaluators.py
uv run python selftest_challenges.py
```

`challenge_oracles.py` derives the answer keys using small exhaustive searches,
exact arithmetic, simulations, and reviewed policy traces. Its code reference
solutions are executed in the benchmark's sandbox. `selftest_challenges.py`
checks those derivations against the YAML and rejects independently corrupted
answer fields and flawed implementations. Both are authoring/verification files;
the model receives only the question prompt and optional system prompt.

This is a static, public suite, so it cannot establish freedom from contamination.
Hard/expert labels are provisional author estimates. Measure discrimination with
fresh runs of both baseline and strong models on the same suite hash, temperature,
and token budgets before claiming a particular performance gap.

## Generated and interactive evaluation

`quality_suite.py` assembles the executable suite around these YAML anchors.
The default adds four reasoning instances and three interactive tasks, producing
276 questions, and extends the eight advanced-coding items with 197 extra cases.
`--static-only` preserves the static suite. Long-context accuracy sizes are an
independent, explicit option; the default adds none.

Each generated family uses an independent RNG stream derived from the generator
revision, split, question seed, family, and variant index. Adding a family does
not perturb other streams. Development and evaluation splits generate different
instances; fixed anchors are shared. Neither public split promises secrecy or
freedom from contamination. Use evaluation seeds consistently across models and
avoid tuning on those instances. Increment `quality_suite.REVISION` when changing
generation or simulation semantics. Full executable question fingerprints include
the prompt, answer, fixtures, budgets, metadata, and hidden simulation setup.

Additional code fixtures use `independent_oracles.py`, with different algorithms
from `challenge_oracles.py`. Verification cross-checks both implementations and
metamorphic properties. Model prompts do not include the extra test inputs.

`interactive_tasks.py` has no external side effects. A model sends one JSON action
and observes the result on each turn, including conflicts and uncertain outcomes.
Tasks stop at 16 model turns or 8,192 total output tokens. A failed tool operation
is part of the task; an endpoint failure is a separate excluded result. The full
trajectory, final state, authorization violations, and excess calls are recorded.

Long-context prompts are measured exactly with `cl100k_base`, excluding chat
envelope overhead. Provider token counts are also stored, with estimates clearly
distinguished. The quality JSON records the reference tokenizer, prompt hash,
size, question seed, and outcome so each tested instance can be reproduced.

Run `uv run python selftest_quality.py` after changing these components. This is
also a CI gate. All network model calls are replaced by fake clients in the tests.
