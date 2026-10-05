"""One fixed, reproducible rigorous question suite for every benchmark run."""

from dataclasses import asdict, replace
import hashlib
import json
import random
from pathlib import Path

REVISION = "rigorous-v15"
# Keep established instances stable when adding families or changing grading.
# New families own their generation version, independently of the run protocol.
GENERATION_REVISION = "rigorous-v11"
MAX_OUTPUT_TOKENS = 65536
QUESTION_SEEDS = (19, 23)
VARIANTS = 2
CONTEXT_SIZES = (8192, 32768)
QUALITY_WORKERS = 4


def question_scope(q):
    if q.evaluator == "open_ended":
        # No answer key: recorded for pairwise A/B studies, never scored.
        return "open_ended"
    return "compliance" if q.category == "Creative Writing" else "capability"


def rng_for(seed, variant, family):
    digest = hashlib.sha256(f"{GENERATION_REVISION}:evaluation:{seed}:{variant}:{family}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def fingerprint(q):
    # The fixed output cap is recorded in the comparison protocol. The question
    # fingerprint covers its prompt, answer key, rubric, request fields and
    # environment. Absent optional fields are omitted so that adding a field
    # never changes the identity of existing questions.
    fields = {
        k: v
        for k, v in asdict(q).items()
        if k != "max_tokens" and not (k in {"rubric", "request"} and v is None)
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def suite_hash(questions):
    return hashlib.sha256(
        "".join(fingerprint(q) for q in sorted(questions, key=lambda q: q.id)).encode()
    ).hexdigest()[:16]


def load_questions(tests_dir=None):
    from app.benchmarking.adversarial_cases import load_adversarial_questions
    from app.benchmarking.compositional_cases import load_compositional_questions
    from app.benchmarking.evidence_cases import load_evidence_questions
    from app.benchmarking.context_cases import FAMILIES, context_question
    from app.benchmarking.frontier_cases import load_frontier_questions
    from app.benchmarking.independent_oracles import CODE, code_cases
    from app.benchmarking.interactive_tasks import make_tasks
    from app.benchmarking.rigorous_cases import load_new_questions
    from app.benchmarking.reasoning_cases import load_reasoning_questions
    from app.benchmarking.reconstruction_cases import make_reconstruction_tasks
    from app.benchmarking.test_loader import load_all_tests

    questions = load_all_tests(tests_dir or Path(__file__).resolve().parents[2] / "tests")
    cases = code_cases(rng_for(QUESTION_SEEDS[0], 0, "code-tests"))
    for q in questions:
        code_id = q.metadata.get("original_anchor_id", q.id)
        if code_id in CODE:
            oracle = CODE[code_id]
            q.expected = list(q.expected) + [
                {
                    "id": f"generated-{i:03d}",
                    "function": oracle.__name__,
                    "args": args,
                    "expected": oracle(*args),
                    "relative": 0,
                    "tolerance": 0,
                    "preserve_inputs": True,
                }
                for i, args in enumerate(cases[code_id])
            ]
    for q in questions:
        if q.evaluator == "json_match":
            q.expected.update(strict_json=True, allow_fence=False)
            q.prompt += "\nReturn exactly one JSON document, without prose or Markdown fences."
        if q.evaluator == "code_exec":
            q.prompt += "\nDo not mutate any positional or keyword inputs."
            for fixture in q.expected:
                fixture["preserve_inputs"] = True
            if not q.rubric:
                boundary = min(5, len(q.expected))
                q.rubric = [
                    {
                        "id": f.get("id", f"fixture-{i + 1:03d}"),
                        "weight": 0.5 / (boundary if i < boundary else len(q.expected) - boundary),
                        "group": "boundary" if i < boundary else "generated-combination",
                        "dimension": "content",
                        "mandatory": True,
                    }
                    for i, f in enumerate(q.expected)
                ]
    for seed in QUESTION_SEEDS:
        questions.extend(load_new_questions(split="evaluation", variants=VARIANTS, seed=seed))
        for variant in range(VARIANTS):
            questions.extend(make_tasks(seed, variant))
            questions.extend(load_frontier_questions(seed, variant))
            questions.extend(load_adversarial_questions(seed, variant))
            questions.extend(load_compositional_questions(seed, variant))
            questions.extend(load_evidence_questions(seed, variant))
            questions.extend(load_reasoning_questions(seed, variant))
            questions.extend(make_reconstruction_tasks(seed, variant))
    questions.extend(
        context_question(size, seed, family)
        for family in FAMILIES
        for seed in QUESTION_SEEDS
        for size in CONTEXT_SIZES
    )
    questions = [
        replace(
            q,
            max_tokens=MAX_OUTPUT_TOKENS,
            metadata={
                **q.metadata,
                "family": q.metadata.get("family", q.id),
                "scope": question_scope(q),
                "cohort": REVISION,
            },
        )
        for q in questions
    ]
    if len({q.id for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs in the canonical suite")
    return questions


def provenance():
    from app.benchmarking.reconstruction_cases import REVISION as reconstruction_revision

    return {
        "revision": REVISION,
        "legacy_generation_revision": GENERATION_REVISION,
        "question_seeds": list(QUESTION_SEEDS),
        "variants": VARIANTS,
        "context_reference_tokens": list(CONTEXT_SIZES),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "behavioral_reconstruction_revision": reconstruction_revision,
    }
