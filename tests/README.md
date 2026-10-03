# Fixed rigorous question suite

The canonical run bank is assembled by [quality_suite.py](../quality_suite.py). Every quality run uses the same 291 questions; the test browser displays that complete bank.

| Component | Count |
|---|---:|
| Static questions: one file per question in questions/ | 171 |
| Generated reasoning/code: 23 families × 2 seeds × 2 variants | 92 |
| Interactive simulations: 5 families × 2 seeds × 2 variants | 20 |
| Long-context: 2 families × 2 seeds × 8k/32k sizes | 8 |

Static anchors span reasoning, constrained decisions, knowledge, security, code, translations, legal/finance/engineering policies and instruction compliance. Typed output contracts state the required fields without revealing answer values. Scientific and specialist tasks are self-contained policy/problem fixtures; they are not current professional advice.

Specialist translation tasks review semantic fidelity in CS↔EN, DE↔EN and CS↔DE: select all faithful candidates while preserving obligations, uncertainty, conditions, dates and units. They do not measure unconstrained translation generation; authored keys benefit from independent bilingual review. Code Review and Security use bounded synthetic repository snapshots with explicit helper contracts and confirmed/refuted/unknown judgments. Fixtures are inert text. Their coverage and independent arithmetic/state checks are verified by [selftest_specialists.py](../selftest_specialists.py).

New v10 tasks add robust portfolio ranking, observational versus interventional and counterfactual probabilities, SQL NULL and join cardinality, weighted minimal policy repair, vector-clock conflict frontiers, and Unicode/casefold revision handling. Every new family has independently derived oracle checks in [selftest_frontier.py](../selftest_frontier.py).

V11 adds adaptive minimax policies versus fixed sensor subsets, worst-case graph failure and one-edge repair, conflict versus view serializability and commit safety, and all minimal evidence supports with contradiction-aware retraction. [selftest_adversarial.py](../selftest_adversarial.py) checks their answers with separate algorithms and rejects 942 corrupted answers. Canonical static JSON tasks explicitly reject prose and Markdown fences. Their full-pass criteria and requirement-balanced diagnostic weights remain separate.

Generated tasks include selective-report conditioning, bounded constraint search, minimal inconsistent sets, temporal policies, revised multihop evidence, reliability decisions, string parsers, dependency layers and transactional state. Fixed authoring seeds are 19 and 23, with two variants. Family identifiers group related instances for balanced scoring.

Interactive models choose a JSON action, receive a simulated observation and act again. Tasks include conflict recovery, payment reconciliation after ambiguous outcomes, paginated deletion previews, cross-resource races and permission-sensitive replanning. Gates check final state, required actions and authorization. External services and real payments are never called. The provider's native tool-calling format is not tested.

All passes require every mandatory condition. Strict success is the primary category/family-balanced capability score. Requirement-balanced partial achievement remains diagnostic. The three writing tasks measure verifiable constraints separately. Full JSON comparisons reject incorrect values, omissions and extra keys; missing structures stay in the partial-credit denominator. Code fixture errors earn zero achievement. Canonical code fixtures weight boundary and generated combinations equally, protect input state, and every fixture is required for a full pass.

Long-context questions require indirect site selection, approved temporal revisions and distributed relational or signed aggregation evidence; both lengths share a seed's core problem. Relevant and filler records have the same identifier grammar. The long-context prompts have exact cl100k_base reference lengths. Provider tokenizers can differ; reported prompt-token counts and unsupported-context outcomes are retained. These are quality questions, separate from the fixed 1k performance workload.

Run **python validate_suite.py --strict** before editing the bank. Keep a derivation or independent executable oracle for answer keys. Test plausible wrong answers, omissions, type errors and boundary cases. Keep prompt constraints consistent with evaluation. Update the revision when behavior changes; fresh runs establish empirical difficulty. Public generation is not a secret holdout.

There are no alternative runtime banks, filters, question seeds, splits or context-size controls. Authoring modules retained for oracle checks are not selectable profiles. Git retains previous source versions and historical audits.

Authoring layout is independent of runtime selection. The recursive loader reads every YAML question, rejects duplicate keys and duplicate IDs, and still produces one canonical suite. The design, research sources and review iterations are documented in [BENCHMARK_DESIGN.md](../BENCHMARK_DESIGN.md).
