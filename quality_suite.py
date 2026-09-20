"""Seeded quality-suite assembly and reproducibility metadata (no model calls)."""

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import itertools
import json
import math
import random

from models import Question

REVISION = "quality-v5"

# Prose pattern coverage is useful diagnostic information, but cannot establish
# correctness: negating a keyword, omitting an untested claim or using a synonym
# can all change the verdict. Keep these tasks out of the capability headline.
HEURISTIC_EVALUATORS = {
    "contains_keywords", "regex_all", "refusal_calibration", "admits_uncertainty",
    "security_analysis", "command_correctness", "multi_step_solution",
    "file_content_match", "ordered_labels", "set_match", "numeric_set",
}


def question_scope(q):
    if q.category == "Creative Writing":
        return "compliance"
    if q.evaluator in HEURISTIC_EVALUATORS or (
        q.evaluator == "format_check" and q.category != "Instruction Following"
    ):
        return "heuristic"
    return "capability"


@dataclass(frozen=True)
class QualityConfig:
    generated: bool = True
    interactive: bool = True
    strengthen_code: bool = True
    seeds: tuple[int, ...] = (1729,)
    split: str = "development"
    variants: int = 1
    context_sizes: tuple[int, ...] = ()
    input_price: float | None = None
    output_price: float | None = None
    max_output_tokens: int = 16384

    def __post_init__(self):
        if type(self.max_output_tokens) is not int or not 1024 <= self.max_output_tokens <= 65536:
            raise ValueError("max_output_tokens must be an integer in 1024..65536")
        if any(
            type(v) is not bool for v in (self.generated, self.interactive, self.strengthen_code)
        ):
            raise ValueError("quality feature switches must be booleans")
        if self.split not in {"development", "evaluation"}:
            raise ValueError("split must be development or evaluation")
        if type(self.variants) is not int or not 1 <= self.variants <= 10:
            raise ValueError("variants must be between 1 and 10")
        if not 1 <= len(self.seeds) <= 10 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("provide 1..10 distinct integer seeds")
        if any(type(s) is not int or not 0 <= s <= 2**32 - 1 for s in self.seeds):
            raise ValueError("seeds must be integers in 0..4294967295")
        if len(set(self.context_sizes)) != len(self.context_sizes) or len(self.context_sizes) > 5:
            raise ValueError("provide at most 5 distinct context sizes")
        if any(type(n) is not int or not 2048 <= n <= 262144 for n in self.context_sizes):
            raise ValueError("quality context sizes must be 2048..262144 reference tokens")
        for value in (self.input_price, self.output_price):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("prices must be finite, nonnegative dollars per million tokens")
        if (self.input_price is None) != (self.output_price is None):
            raise ValueError("provide both input and output prices, or neither")

    @classmethod
    def from_dict(cls, data):
        data = dict(data or {})
        for key in ["seeds", "context_sizes"]:
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


def parse_ints(value):
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def rng_for(config, seed, variant, family):
    # Independent streams: adding another family never changes existing problems.
    digest = hashlib.sha256(
        f"{REVISION}:{config.split}:{seed}:{variant}:{family}".encode()
    ).digest()
    return random.Random(int.from_bytes(digest, "big"))


def fingerprint(q):
    return hashlib.sha256(
        json.dumps(asdict(q), sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def suite_hash(questions):
    return hashlib.sha256(
        "".join(fingerprint(q) for q in sorted(questions, key=lambda q: q.id)).encode()
    ).hexdigest()[:16]


def generated_question(config, seed, variant, family, category, prompt, answer, parameters):
    return Question(
        id=f"G3-{family}-{config.split}-s{seed}-v{variant}",
        category=category,
        prompt=prompt + "\nReturn only the requested JSON. Arrays must be in the specified order.",
        evaluator="json_match",
        expected={"mode": "exact", "value": answer},
        difficulty="expert",
        max_tokens=8192,
        source=REVISION,
        metadata={
            "family": family,
            "split": config.split,
            "seed": seed,
            "variant": variant,
            "revision": REVISION,
            "parameters": parameters,
        },
    )


def generate_reasoning(config, seed, variant):
    out = []

    def add(family, category, prompt, answer, data):
        out.append(
            generated_question(config, seed, variant, family, category, prompt, answer, data)
        )

    rng = rng_for(config, seed, variant, "assignment")
    n = rng.randint(5, 7)
    costs = [[rng.randint(1, 20) for _ in range(n)] for _ in range(n)]
    forbidden = [rng.randrange(n) for _ in range(n)]
    candidates = [
        (sum(costs[i][p[i]] for i in range(n)), list(p))
        for p in itertools.permutations(range(n))
        if all(p[i] != forbidden[i] for i in range(n)) and p[0] < p[-1]
    ]
    optimum = min(c for c, _ in candidates)
    assignments = [p for c, p in candidates if c == optimum]
    data = {"costs": costs, "forbidden": forbidden}
    add(
        "assignment",
        "Logical Reasoning",
        f"Assign workers 0..{n - 1} bijectively to jobs 0..{n - 1}. Cost matrix by worker: {json.dumps(costs)}. "
        f"Worker i cannot take forbidden[i], where forbidden={forbidden}. Worker 0 must take a smaller job number "
        'than the last worker. Minimize total cost. Return {"cost":integer,"assignments":[all optimal job '
        "arrays in worker order, sorted lexicographically]}.",
        {"cost": optimum, "assignments": assignments},
        data,
    )

    rng = rng_for(config, seed, variant, "automaton")
    n, ones, modulus = rng.randint(12, 16), rng.randint(4, 8), rng.choice([3, 5, 7])
    pattern = rng.choice(["001", "101", "1101", "0100"])
    states = {("", 0, 0): 1}
    for _ in range(n):
        nxt = {}
        for (suffix, k, remainder), count in states.items():
            for digit in [0, 1]:
                word = suffix + str(digit)
                if pattern in word or k + digit > ones:
                    continue
                key = (word[-(len(pattern) - 1) :], k + digit, (2 * remainder + digit) % modulus)
                nxt[key] = nxt.get(key, 0) + count
        states = nxt
    count = sum(c for (_, k, r), c in states.items() if k == ones and r == 0)
    data = {"length": n, "ones": ones, "modulus": modulus, "forbidden": pattern}
    add(
        "automaton",
        "Mathematical Reasoning",
        f"Count length-{n} binary strings, leading zeros allowed, "
        f"with exactly {ones} ones, no substring {pattern}, and binary numeric value divisible by {modulus}. "
        'Return {"count":integer}.',
        {"count": count},
        data,
    )

    rng = rng_for(config, seed, variant, "urn")
    red, blue, draw = rng.randint(4, 8), rng.randint(3, 7), rng.randint(3, 5)
    probabilities = [rng.randint(1, 5) for _ in range(draw + 1)]
    weights = [
        Fraction(math.comb(red, k) * math.comb(blue, draw - k), math.comb(red + blue, draw))
        * Fraction(probabilities[k], 5)
        for k in range(draw + 1)
    ]
    event = sum(weights)
    posterior = sum(k * w for k, w in enumerate(weights)) / event

    def frac(x):
        return [x.numerator, x.denominator]

    data = {"red": red, "blue": blue, "draw": draw, "report_fifths": probabilities}
    add(
        "urn",
        "Mathematical Reasoning",
        f"An urn contains {red} red and {blue} blue balls. Draw {draw} without replacement. "
        f"Given exactly k red draws, a report is sent with probability p[k]/5, where p={probabilities}. "
        'Return {"report_probability":[numerator,denominator],"expected_red_given_report":[numerator,denominator]} '
        "using reduced fractions.",
        {"report_probability": frac(event), "expected_red_given_report": frac(posterior)},
        data,
    )

    rng = rng_for(config, seed, variant, "boolean")
    n = rng.randint(6, 9)
    witness = [rng.randrange(2) for _ in range(n)]
    equations = []
    for _ in range(n - 2):
        indexes = sorted(rng.sample(range(n), rng.randint(2, 4)))
        equations.append([indexes, sum(witness[i] for i in indexes) % 2])
    models = [
        list(p)
        for p in itertools.product([0, 1], repeat=n)
        if all(sum(p[i] for i in indexes) % 2 == rhs for indexes, rhs in equations)
    ]
    data = {"n": n, "equations": equations}
    add(
        "boolean",
        "Logical Reasoning",
        f"Variables x0..x{n - 1} are 0 or 1. Each pair [indexes,rhs] requires "
        f"the sum of those variables modulo 2 to equal rhs: {json.dumps(equations)}. "
        'Return {"count":number of assignments,"first":lexicographically smallest full 0/1 array, '
        '"forced":[[index,value],...] for variables identical across every assignment, sorted by index]}.',
        {
            "count": len(models),
            "first": models[0],
            "forced": [[i, models[0][i]] for i in range(n) if len({p[i] for p in models}) == 1],
        },
        data,
    )
    return out


def long_context_question(config, seed, variant, size):
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    rng = rng_for(config, seed, variant, "context")
    account, region = f"acct-{rng.randrange(10000, 99999)}", f"zone-{rng.randrange(100, 999)}"
    quota, multiplier = rng.randint(21, 79), rng.randint(3, 9)
    evidence = [
        f"RECORD E1: account {account} belongs to region {region}.\n",
        f"RECORD E2: region {region} approved quota revision 1 is {quota - 7}.\n"
        f"RECORD E3: region {region} approved quota revision 2 is {quota}.\n"
        f"RECORD E4: region {region} DRAFT quota revision 3 is {quota + 18}; drafts have no effect.\n",
        f"RECORD E5: account {account} multiplier is {multiplier}.\n"
        f"RECORD E6: account {account}-old multiplier is 1; this is a different account.\n",
    ]
    header = (
        "Reconcile the following archive. Use the highest APPROVED revision for the exact region, "
        "ignoring drafts. Account and region identifiers must match exactly.\n"
    )
    question = (
        f'\nFor account {account}, return JSON {{"region":string,"quota":integer,"multiplier":integer,'
        '"total":quota*multiplier,"evidence":[IDs supporting these facts in lexical order]}}. '
        "Cite only the current approved quota, account-region mapping, and multiplier records.\n"
    )
    # Tokenize complete sections: token boundaries at joins are re-measured below.
    filler = "".join(
        f"ARCHIVE {i}: account other-{rng.randrange(10000, 99999)}; region other-{i}; "
        f"approved quota {rng.randrange(1, 99)}; revision 1; operational note unchanged.\n"
        for i in range(size // 10 + 20)
    )
    tokens = encoding.encode(filler)
    budget = size - len(encoding.encode(header + "".join(evidence) + question)) - 20
    boundaries = [0, budget // 10, budget // 2, 9 * budget // 10, budget]
    parts = [encoding.decode(tokens[a:b]) for a, b in zip(boundaries, boundaries[1:])]
    rng.shuffle(evidence)
    prompt = (
        header
        + "".join(parts[i] + "\n" + record for i, record in enumerate(evidence))
        + parts[3]
        + question
    )
    # Pad with a single-token ASCII word, then measure the full prompt exactly.
    while len(encoding.encode(prompt)) < size:
        prompt = prompt.replace("\nFor account", " pad\nFor account", 1)
    actual = len(encoding.encode(prompt))
    if actual != size:
        raise ValueError(f"Context assembly produced {actual} tokens, expected {size}")
    q = generated_question(
        config,
        seed,
        variant,
        "context",
        "Long Context Quality",
        prompt,
        {
            "region": region,
            "quota": quota,
            "multiplier": multiplier,
            "total": quota * multiplier,
            "evidence": ["E1", "E3", "E5"],
        },
        {},
    )
    # Do not append any unmeasured instructions after constructing the prompt.
    q.prompt = prompt
    q.id += f"-n{size}"
    q.max_tokens = 2048
    q.metadata.update(
        context_tokens=size,
        reference_tokenizer="cl100k_base",
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )
    return q


def assemble_questions(base, config):
    from independent_oracles import CODE, code_cases
    from interactive_tasks import make_tasks

    questions = [
        replace(
            q,
            metadata={
                **q.metadata,
                "family": q.id.rsplit("-", 1)[0] if q.id.startswith("H5-") else q.id,
                "scope": question_scope(q),
                "cohort": "ceiling-v5" if q.id.startswith("H5-") else "anchor",
            },
        )
        for q in base
    ]
    if config.strengthen_code:
        # All model runs using the same suite configuration receive identical tests.
        cases = code_cases(rng_for(config, config.seeds[0], 0, "code-tests"))
        for q in questions:
            if q.id in CODE:
                oracle = CODE[q.id]
                q.expected = list(q.expected) + [
                    {
                        "function": oracle.__name__,
                        "args": args,
                        "expected": oracle(*args),
                        "relative": 0,
                        "tolerance": 0,
                    }
                    for args in cases[q.id]
                ]
                q.metadata.update(
                    revision=REVISION, fixture_seed=config.seeds[0], split=config.split
                )
    for seed in config.seeds:
        for variant in range(config.variants):
            if config.generated:
                questions.extend(generate_reasoning(config, seed, variant))
            if config.interactive:
                questions.extend(make_tasks(config, seed, variant))
            for size in config.context_sizes:
                questions.append(long_context_question(config, seed, variant, size))
    # Every model receives the same declared cap, including baseline anchors.
    # Keep truncations as failures; a larger budget cannot recover past answers.
    return [replace(q, max_tokens=config.max_output_tokens) if not q.interaction else q
            for q in questions]
