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
