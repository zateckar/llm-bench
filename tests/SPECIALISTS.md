# Specialist-v1

30 additional questions in `specialists.yaml`, authored by `specialist_cases.py`,
`translation_cases.py` and `review_cases.py`. The loader automatically exposes the
five new category names to category selection and reporting. Security uses the
existing category, so its eight additions run alongside the earlier security items.

| Category | IDs | Count | Coverage |
|---|---|---:|---|
| Legal | SP1-LE-* | 4 | EPC novelty versus inventive step; territorial patent claim charts and freedom to operate; US trademark screening; contract/IP vesting, deadlines and survival |
| Finances | SP1-FI-* | 4 | Restricted cash and revolver ladder; natural FX netting and forwards; accrued interest and leverage covenants; liquidity stress and collateral haircuts |
| R&D | SP1-RD-* | 4 | EV regeneration and battery energy; worst-case mechanical tolerances; robotic transforms, timestamps and stopping envelopes; experimental units and statistical evidence |
| Language Translations | SP1-TR-* | 6 | CS↔EN, DE↔EN, CS↔DE; obligations, negation, uncertainty, conditions, dates, money and technical units |
| Code Review | SP1-CR-* | 4 | Cross-file correctness, security, transactions, concurrency, retries, API/UI semantics, schema/migrations, deployment/rollback, performance, dependencies, tests, privacy, accessibility and observability |
| Security | SP1-SE-* | 8 | Tenant/cache authorization and mass assignment; webhook replay; SSRF redirects and DNS; archive symlinks; OIDC session binding; CI credentials and artifact provenance; delegated tool authorization; ECU update metadata and rollback |

All additions use strict structural JSON grading, with no keyword credit and no
model judge. Every requested field must match to pass. Review tasks require a
complete vector of confirmed/refuted/unknown judgments, exact execution consequences,
and selection of sufficient repairs. Missing findings, false positives, invented
facts and ineffective fixes fail. All review tasks contain both safe behavior and
facts that cannot be established from the evidence. Code Review supplies 7–8 named
files per snapshot; Security supplies 4–6. These are bounded synthetic repositories
with explicit helper contracts, not arbitrary live repository audits or claims of
exhaustive real-world security coverage. Fixtures are inert text and never executed.

Translations test **semantic fidelity review**: three passages per direction, each
with four candidate translations, and all faithful alternatives must be selected.
Several idiomatic phrasings are valid. This avoids rejecting a correct free-form
translation merely because it differs from one reference string. It does not measure
unconstrained translation generation, literary quality, or spoken fluency. The older
`Translation` category remains separate. Authored language keys still benefit from
independent bilingual review; automated checks verify grading, not linguistic truth.

Legal questions state their jurisdiction or fictional contract, dates, evidence and
applicable rule pack. They distinguish screening results from guarantees and omit
unstated doctrines. Finance uses synthetic rates and explicit settlement/day-count
rules; engineering uses bounded physical models. They are exercises, not live advice.

Primary references checked on 2026-09-20:

- [EPC Article 56](https://www.epo.org/en/legal/epc/2020/a56.html): Article 54(3) documents are excluded from inventive-step assessment.
- [WIPO patent FAQ](https://www.wipo.int/en/web/patents/faq_patents): patent rights, territorial protection and licensing context.
- [USPTO likelihood of confusion](https://www.uspto.gov/trademarks/search/likelihood-confusion): similarity and related goods/services in trademark screening.
- [ECB reference exchange rates](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html): informational reference rates are not executable transaction quotes.
- [OWASP SSRF prevention](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html): redirect and DNS validation boundaries.

Regenerate and verify without model/API calls:

```bash
uv run python specialist_cases.py
uv run python validate_suite.py --strict
uv run python selftest_specialists.py
```

The specialist self-test checks shipped fixtures against authoring, category
integration, all six language directions, positive/negative/unknown controls,
corruption of every answer field, and independent arithmetic, enumerations and
state simulations. It runs in CI. Neither fixture equality nor mutation rejection
alone validates an answer's meaning; the independent derivations and explicit
contracts provide additional checks. No benchmark model outputs are rewritten.

Specialist-v1 brought the static suite to 345 questions in 28 categories. Ceiling-v5
subsequently expands it to 397 static / 404 default questions and updates the grading
protocol to quality-v5. The suite hash and cache fingerprint change with new prompts
and grading specifications. Use fresh runs with matching suite hash, budgets and settings
to assess model separation. Expert difficulty is an author estimate, not an empirical
guarantee that any particular model will rank above another.
