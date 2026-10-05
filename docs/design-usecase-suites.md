# Design: use-case suites contributed by application teams

Roadmap item 2. It builds on the suite registry from [the tool-conformance design](design-tool-conformance-and-repeats.md).

## Goal

The rigorous suite answers "how capable is this model in general". Application teams need to know "does this model do *our* task": extracting fields from our invoices, classifying our tickets, answering in our support tone. They have golden examples but no developers on this repository. This design lets them supply those examples as a versioned suite and run them through the same execution, grading, repeat and comparison machinery as the built-in suites.

## Decisions (recommended options in force)

1. **Storage: database, uploaded through the admin UI.** The suite is one YAML document. Every upload that changes the content creates a new immutable version; a byte-identical (same fingerprint) re-upload is rejected as a no-op. The alternative, a `suites/` directory in the repository, would need repository access and a redeploy per edit, and would put possibly internal examples into source control. Versions can be downloaded as YAML, so teams can still keep them in their own repositories.
2. **Deterministic evaluators only, with no code execution.** Allowed: `exact_match`, `mcq`, `numeric_match`, `numeric_set`, `contains_keywords`, `regex_all`, `json_match`, `set_match`, `format_check`, `ordered_labels`, `refusal_calibration`, `admits_uncertainty`. `code_exec` and the other code- or simulation-based evaluators are excluded: an uploaded suite must not cause server-side execution of fixtures. LLM-judged questions arrive with roadmap item 3 as one more allowed evaluator.
3. **Runs pin an exact version.** Run options store `usecase:<slug>@<version>`, so a queued run, "Run again", clone or edit never silently switches to a newer upload. The run form defaults to the latest version.
4. **Versions of one suite compare on shared questions.** The report's suite name is `usecase:<slug>` and the protocol revision is the format revision (`usecase-v1`), not the version. Paired comparisons across versions therefore use only questions whose fingerprints are identical, and show unmatched counts, which is exactly what the comparison code already does. Repeat groups still require one suite hash.
5. **Same execution protocol as rigorous.** `bounded-quality-v1` (at most two calls, a shared 65,536-token budget unless a question sets `max_tokens`) and the same evaluator versions. Questions that set `response_format` run through the native structured-output path from the tool-conformance work, because for applications that API contract *is* the task.
6. **Admin-only management, visible results.** Upload, archive and download are admin-only. Runs and reports are visible to all signed-in users, as today. Archiving hides a suite from the run form; versions are never deleted while runs reference them. Per-team confidentiality is out of scope (see G).

## A. Suite document

```yaml
suite:
  slug: invoice-extraction          # [a-z0-9][a-z0-9-]{1,47}; stable identity across versions
  name: Invoice extraction (Finance)
  description: Header fields from supplier invoices, CZ/DE/EN.
  owner: finance-apps@example.com    # free text, shown on reports
  system_prompt: You extract data ... # optional default for every question

questions:
  - id: inv-001
    category: Header fields
    prompt: |
      Extract supplier VAT ID, invoice number and total from: ...
    evaluator: json_match
    expected: {value: {vat_id: CZ12345678, number: "2026-0042", total: 1250.5}}
    response_format:                  # optional; json_match only
      type: json_schema
      json_schema: {name: invoice, schema: {...}}
    metadata: {family: supplier-a}    # optional; variants of one case share a family
```

Every question field the rigorous loader accepts is accepted, with the same strict checks (unknown fields, duplicate keys and ids, thresholds, weights, rubrics). `response_format` is the one addition. Limits: 2 MB document, 2,000 questions, ids unique within the suite.

## B. Validation on upload

- Parse with the existing strict loader (`test_loader._parse_question`).
- Run the same static checks as `validate_suite.py` (regexes that match everything, guessable exact-match items, malformed `json_match` keys, and so on). They move from the root script into `app/benchmarking/suite_checks.py`; `validate_suite.py` keeps its command-line interface and re-exports the names the selftests import.
- Errors block the upload; warnings are shown and stored with the version.
- `response_format` must be `json_object` or `json_schema` within the supported schema subset, and the expected value must validate against the schema.
- A **Check** button validates without saving.

## C. Storage

```sql
CREATE TABLE IF NOT EXISTS usecase_suites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL,
    version INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    owner TEXT,
    source_yaml TEXT NOT NULL,      -- packed like other payloads
    suite_hash TEXT NOT NULL,
    question_count INTEGER NOT NULL,
    warnings_json TEXT,
    created_by INTEGER,
    created_at TIMESTAMP,
    archived INTEGER NOT NULL DEFAULT 0,
    UNIQUE (slug, version)
);
```

## D. Registry and execution

- `get_suite("usecase:<slug>@<version>")` loads the version from the database (synchronously, because the runner thread uses it) and returns a `SuiteDef` named `usecase:<slug>`.
- Provenance: format revision, slug, version, display name, owner, suite hash, question count.
- Questions get `scope: capability`, `family` defaulting to the id, `cohort: usecase:<slug>`, and `max_tokens` defaulting to 65,536.
- `response_format` questions become native structured-output questions (`native_structured_output`, `request.response_format`, one turn).
- Run submission validates that the referenced version exists. Reruns of archived versions remain possible.

## E. UI

- **Admin → Use-case suites** (`/admin/suites`): latest version per suite, with owner, question count, fingerprint and run count; an upload form (file or pasted YAML) with Check and Save; a link to the YAML format.
- **Suite page** (`/admin/suites/{slug}`): versions with warnings, YAML download, archive/unarchive, and the latest version's questions grouped by category.
- **Run form:** the suite select lists the built-in suites plus the latest non-archived version of each use-case suite. Pinned older versions in edited plans stay selectable.
- **Reports** show the suite's display name and version instead of the bare revision.

## F. Verification

- **Parser:** a valid document produces stable fingerprints and hash; each invalid case is rejected with a specific message (disallowed evaluator, bad slug, duplicate ids, unknown fields, an everything-matching regex, a `response_format` whose expected value violates its schema, oversize documents).
- **Routes:** upload creates v1, an identical re-upload is rejected, a change creates v2, Check saves nothing, archive hides the suite from the form, and non-admins are redirected.
- **Submission:** pinned options are stored, unknown versions return 422, and rerun/clone keep the pinned version.
- **Runner:** `get_suite(...).load()` works against a patched database path.
- **End to end:** fake-client results produce a report named `usecase:<slug>`, the structured-output question runs natively, and a comparison across two versions pairs only identical questions.

## G. Out of scope here

- Multi-turn or few-shot message lists per question, and file or image inputs.
- Per-team visibility of suites and results.
- LLM-judged questions (roadmap item 3).
- A self-service editor; teams edit YAML and re-upload.
