#!/usr/bin/env python3
"""Offline integration, differential, property, and regression tests for quality v3.

No LLM endpoints or live user databases are used. The reference tokenizer may
download its verified vocabulary once unless TIKTOKEN_CACHE_DIR is prepopulated.
"""

import asyncio
import copy
from dataclasses import asdict, replace
from fractions import Fraction
from functools import lru_cache
import inspect
import itertools
import json
from pathlib import Path
import random
import re
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from challenge_oracles import CODE as ORIGINAL_CODE
from independent_oracles import CODE, code_cases
from interactive_tasks import Environment, make_tasks, run_interaction
from llm_client import ChatClient, ClientConfig
from models import Question, RequestMetrics, Result, TokenUsage
from quality_execution import score_response
from quality_report import make_report, paired_comparison, result_record, summarize
from quality_suite import (
    QualityConfig,
    assemble_questions,
    generate_reasoning,
    long_context_question,
    suite_hash,
)
from test_loader import load_all_tests

ROOT = Path(__file__).parent


class FakeAgent:
    """Adaptive client using only messages/tool observations, not hidden state."""

    def __init__(self, question):
        self.q = question
        self.history = []
        self.actions = []

    def complete_messages(self, messages, **kwargs):
        self.history.append(copy.deepcopy(messages))
        prompt = messages[1]["content"]
        responses = [
            json.loads(m["content"].removeprefix("Simulated tool result: "))
            for m in messages[2:]
            if m["role"] == "user"
        ]
        last = responses[-1] if responses else None
        action = {"done": True}
        if "get_document" in prompt:
            ident = re.search(r"on document (d\d+)", prompt)[1]
            target = int(re.search(r"limit to (\d+)", prompt)[1])
            if last is None or last.get("status") == 412:
                action = {"tool": "get_document", "args": {"id": ident}}
            elif "body" in last:
                body = copy.deepcopy(last["body"])
                body.update(limit=target, labels=sorted(set(body["labels"]) | {"red"}))
                action = {
                    "tool": "put_document",
                    "args": {"id": ident, "if_match": last["etag"], "body": body},
                }
        elif "lookup_payment" in prompt:
            order, amount = re.search(r"order (o\d+) exactly (\d+)", prompt).groups()
            key = "pay:" + order
            if last is None or last.get("status") == "absent":
                action = {
                    "tool": "charge",
                    "args": {"order": order, "amount": int(amount), "key": key},
                }
            elif last.get("error") == "timeout":
                action = {"tool": "lookup_payment", "args": {"key": key}}
            elif last.get("status") == "succeeded":
                action = {
                    "tool": "record_receipt",
                    "args": {k: last[k] for k in ["order", "payment", "amount"]},
                }
        else:
            if last is None:
                action = {"tool": "list_objects", "args": {"cursor": None}}
            elif last.get("next"):
                action = {"tool": "list_objects", "args": {"cursor": last["next"]}}
            else:
                now = int(re.search(r"Current time is (\d+)", prompt)[1])
                tenant = re.search(r"tenant (t\d+)", prompt)[1]
                objects = {}
                for response in responses:
                    for obj in response.get("rows", []):
                        if (
                            obj["id"] not in objects
                            or obj["version"] > objects[obj["id"]]["version"]
                        ):
                            objects[obj["id"]] = obj
                previewed = {r.get("would_delete") for r in responses}
                for ident, obj in sorted(objects.items()):
                    if (
                        ident not in previewed
                        and obj["tenant"] == tenant
                        and not obj["pinned"]
                        and obj["expires_at"] <= now
                    ):
                        action = {
                            "tool": "delete_object",
                            "args": {"id": ident, "if_version": obj["version"], "dry_run": True},
                        }
                        break
        self.actions.append(action)
        return (
            json.dumps(action),
            TokenUsage(100, 20),
            RequestMetrics(
                latency_ms=10,
                ttft_ms=2,
                prompt_tokens=100,
                completion_tokens=20,
                finish_reason="stop",
            ),
        )


class SuiteTests(unittest.TestCase):
    def test_reproducibility_and_holdout(self):
        config = QualityConfig(seeds=(19, 23), variants=2)
        a = assemble_questions([], config)
        b = assemble_questions([], QualityConfig.from_dict(asdict(config)))
        self.assertEqual(suite_hash(a), suite_hash(b))
        heldout = assemble_questions([], replace(config, split="evaluation"))
        self.assertFalse({q.id for q in a} & {q.id for q in heldout})
        self.assertNotEqual(suite_hash(a), suite_hash(heldout))
        self.assertEqual(len(a), 28)
        # Variant inputs change matrix dimensions, costs, constraints, etc.
        matrices = [
            q.metadata["parameters"]["costs"] for q in a if q.metadata["family"] == "assignment"
        ]
        self.assertGreater(len({json.dumps(m) for m in matrices}), 1)

    def test_config_rejects_invalid_and_unbounded_work(self):
        for kwargs in [
            {"seeds": ()},
            {"seeds": (1, 1)},
            {"seeds": (-1,)},
            {"variants": 0},
            {"variants": True},
            {"variants": 11},
            {"split": "secret"},
            {"context_sizes": (1,)},
            {"context_sizes": (8192, 8192)},
            {"input_price": 1},
            {"input_price": float("nan"), "output_price": 1},
            {"input_price": -1, "output_price": 1},
            {"generated": "yes"},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QualityConfig(**kwargs)

    def test_generated_answers_independently(self):
        for seed in [3, 19, 71]:
            for q in generate_reasoning(QualityConfig(), seed, 0):
                p, family = q.metadata["parameters"], q.metadata["family"]
                answer = q.expected["value"]
                if family == "automaton":
                    count = sum(
                        s.count("1") == p["ones"]
                        and p["forbidden"] not in s
                        and int(s, 2) % p["modulus"] == 0
                        for s in (format(n, f"0{p['length']}b") for n in range(2 ** p["length"]))
                    )
                    self.assertEqual(answer, {"count": count})
                elif family == "assignment":
                    # Minimum-cost subset DP and optimal-path reconstruction,
                    # independent of the generator's permutation enumeration.
                    n = len(p["costs"])

                    def options(mask, first):
                        worker = mask.bit_count()
                        return [
                            j
                            for j in range(n)
                            if not mask & (1 << j)
                            and j != p["forbidden"][worker]
                            and (worker != n - 1 or first < j)
                        ]

                    @lru_cache(None)
                    def best(mask, first):
                        worker = mask.bit_count()
                        if worker == n:
                            return 0
                        return min(
                            (
                                p["costs"][worker][j]
                                + best(mask | (1 << j), j if worker == 0 else first)
                                for j in options(mask, first)
                            ),
                            default=float("inf"),
                        )

                    paths = []

                    def reconstruct(mask, path):
                        if len(path) == n:
                            paths.append(path)
                            return
                        first = path[0] if path else -1
                        for j in options(mask, first):
                            next_first = j if not path else first
                            if p["costs"][len(path)][j] + best(mask | (1 << j), next_first) == best(
                                mask, first
                            ):
                                reconstruct(mask | (1 << j), path + [j])

                    reconstruct(0, [])
                    self.assertEqual(answer, {"cost": best(0, -1), "assignments": sorted(paths)})
                elif family == "urn":
                    # Enumerate labeled draws instead of the hypergeometric formula.
                    reports, red_reports, total = Fraction(0), Fraction(0), 0
                    for draw in itertools.combinations(range(p["red"] + p["blue"]), p["draw"]):
                        red = sum(i < p["red"] for i in draw)
                        prob = Fraction(p["report_fifths"][red], 5)
                        reports += prob
                        red_reports += red * prob
                        total += 1
                    a, b = reports / total, red_reports / reports
                    self.assertEqual(
                        answer,
                        {
                            "report_probability": [a.numerator, a.denominator],
                            "expected_red_given_report": [b.numerator, b.denominator],
                        },
                    )
                else:
                    # Solve GF(2) equations by elimination; count = 2^(n-rank).
                    rows = [
                        sum(1 << i for i in indexes) | (rhs << p["n"])
                        for indexes, rhs in p["equations"]
                    ]
                    rank = 0
                    for column in range(p["n"]):
                        pivot = next(
                            (i for i in range(rank, len(rows)) if rows[i] & (1 << column)), None
                        )
                        if pivot is None:
                            continue
                        rows[rank], rows[pivot] = rows[pivot], rows[rank]
                        for i in range(len(rows)):
                            if i != rank and rows[i] & (1 << column):
                                rows[i] ^= rows[rank]
                        rank += 1
                    self.assertEqual(answer["count"], 2 ** (p["n"] - rank))
                    forced = []
                    for row in rows:
                        mask = row & ((1 << p["n"]) - 1)
                        if mask and mask & (mask - 1) == 0:
                            forced.append([mask.bit_length() - 1, row >> p["n"]])
                    self.assertEqual(answer["forced"], sorted(forced))
                    self.assertTrue(
                        all(
                            sum(answer["first"][i] for i in indexes) % 2 == rhs
                            for indexes, rhs in p["equations"]
                        )
                    )

    def test_differential_code_oracles_and_properties(self):
        cases = code_cases(random.Random(900), count=40)
        base = {q.id: q for q in load_all_tests(ROOT / "tests")}
        for ident, fn in CODE.items():
            for fixture in base[ident].expected:
                self.assertEqual(fn(*fixture["args"]), fixture["expected"], ident)
            for args in cases[ident]:
                self.assertEqual(
                    fn(*copy.deepcopy(args)),
                    ORIGINAL_CODE[ident](*copy.deepcopy(args)),
                    (ident, args),
                )
        # Metamorphic properties: translation, duplication, irrelevant events.
        for intervals in [[], [[0, 3], [1, 5], [5, 7]], [[-4, -1], [-2, 1]]]:
            result = CODE["AC2-02"](intervals)
            self.assertEqual(
                CODE["AC2-02"]([[a + 17, b + 17] for a, b in intervals]),
                [[a + 17, b + 17, c] for a, b, c in result],
            )
            self.assertEqual(CODE["AC2-02"](intervals * 2), [[a, b, c * 2] for a, b, c in result])
        events = [["1", "BANK", "a", 8], ["2", "a", "b", 3]]
        self.assertEqual(CODE["AC2-07"](events), CODE["AC2-07"](events + events))
        self.assertEqual(CODE["AC2-04"]([["a", 1, 2], ["b", 0, 9]], [["a", 2]]), [2])

    def test_expanded_fixtures_execute_in_real_sandbox(self):
        from evaluators import eval_code_exec

        original = [q for q in load_all_tests(ROOT / "tests") if q.id in CODE]
        expanded = assemble_questions(original, QualityConfig(generated=False, interactive=False))
        for q in expanded:
            self.assertGreater(len(q.expected), 30)
            score, detail = eval_code_exec(
                "```python\n" + inspect.getsource(ORIGINAL_CODE[q.id]) + "\n```", q.expected
            )
            self.assertEqual(score, 1, (q.id, detail))

    def test_long_context_exact_lengths_and_evidence(self):
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        for size in [8192, 32768, 131072]:
            q = long_context_question(QualityConfig(), 31, 0, size)
            self.assertEqual(len(enc.encode(q.prompt)), size)
            # Parse facts from the emitted prompt, independent of its answer key.
            account, region = re.search(
                r"RECORD E1: account (\S+) belongs to region (\S+)\.", q.prompt
            ).groups()
            quota = int(
                re.search(
                    rf"RECORD E3: region {region} approved quota revision 2 is (\d+)", q.prompt
                )[1]
            )
            multiplier = int(
                re.search(rf"RECORD E5: account {account} multiplier is (\d+)", q.prompt)[1]
            )
            self.assertEqual(
                q.expected["value"],
                {
                    "region": region,
                    "quota": quota,
                    "multiplier": multiplier,
                    "total": quota * multiplier,
                    "evidence": ["E1", "E3", "E5"],
                },
            )
            positions = [
                len(enc.encode(q.prompt[: q.prompt.index(f"RECORD {i}:")])) / size
                for i in ["E1", "E3", "E5"]
            ]
            self.assertGreater(max(positions) - min(positions), 0.65)
            self.assertEqual(q.prompt, long_context_question(QualityConfig(), 31, 0, size).prompt)


class InteractiveTests(unittest.TestCase):
    def test_adaptive_agents_and_transcripts(self):
        branches = set()
        for seed in range(8):
            for q in make_tasks(QualityConfig(), seed, 0):
                agent = FakeAgent(q)
                result = run_interaction(q, agent)
                self.assertEqual(result.score, 1, (q.id, result.detail, result.response))
                self.assertEqual(result.outcome, "pass")
                self.assertEqual(result.diagnostics["unnecessary_calls"], 0)
                self.assertEqual(result.tokens.prompt_tokens, 100 * len(agent.actions))
                self.assertEqual(result.metrics.attempts, len(agent.actions))
                self.assertGreater(len(agent.history[-1]), len(agent.history[0]))
                if q.interaction["kind"] == "payment":
                    branches.add(q.interaction["commit_on_timeout"])
        self.assertEqual(branches, {True, False})

    def test_violations_cannot_be_repaired_into_a_pass(self):
        q = make_tasks(QualityConfig(), 3, 0)[2]
        env = Environment(q.interaction)
        env.call("delete_object", {"id": "k2", "if_version": 8, "dry_run": False})
        env.call("list_objects", {"cursor": None})
        env.call("list_objects", {"cursor": "page2"})
        for ident, version in [("k2", 8), ("k5", 3)]:
            env.call("delete_object", {"id": ident, "if_version": version, "dry_run": True})
        self.assertFalse(env.verdict()["success"])
        self.assertEqual(env.verdict()["violations"], ["unauthorized_delete"])

    def test_stale_write_loses_no_concurrent_data(self):
        q = make_tasks(QualityConfig(), 3, 0)[0]
        env = Environment(q.interaction)
        p = q.interaction
        old = env.call("get_document", {"id": p["doc"]})
        self.assertEqual(
            env.call(
                "put_document", {"id": p["doc"], "if_match": old["etag"], "body": old["body"]}
            )["status"],
            412,
        )
        current = env.call("get_document", {"id": p["doc"]})
        body = {**old["body"], "limit": p["target"], "labels": ["blue", "red"]}
        env.call("put_document", {"id": p["doc"], "if_match": current["etag"], "body": body})
        self.assertIn("lost_concurrent_update", env.verdict()["violations"])

    def test_bounds_malformed_and_cancelled(self):
        q = make_tasks(QualityConfig(), 3, 0)[0]

        class Client:
            def __init__(self, text):
                self.text = text

            def complete_messages(self, *args, **kwargs):
                return self.text, TokenUsage(1, 1), RequestMetrics(ok=True)

        for text, outcome in [
            ("oops", "formatting"),
            ('{"done":1}', "formatting"),
            ('{"done":true}', "task_failure"),
        ]:
            result = run_interaction(q, Client(text))
            self.assertEqual(result.outcome, outcome)
            self.assertEqual(result.score, 0)
        result = run_interaction(q, Client('{"tool":"get_document","args":{"id":"missing"}}'))
        self.assertEqual(
            len([x for x in result.diagnostics["transcript"] if x["role"] == "assistant"]), 16
        )
        cancelled = run_interaction(q, Client("oops"), cancelled=lambda: True)
        self.assertEqual(cancelled.outcome, "cancelled")
        self.assertFalse(cancelled.is_scored)


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.q = Question(
            "q", "Reasoning", "reply", "json_match", {"value": {"x": 1}}, metadata={"family": "f"}
        )
        self.config = ClientConfig("http://fake.invalid", "unused", "fake")

    def result(self, score=1, **kwargs):
        return Result(
            self.q,
            "{}",
            score,
            tokens=TokenUsage(100, 20),
            metrics=RequestMetrics(latency_ms=50, prompt_tokens=100, completion_tokens=20),
            **kwargs,
        )

    def test_failure_attribution_and_denominators(self):
        from models import CategoryResult

        cases = [
            ('{"x":1}', RequestMetrics(), "pass", True),
            ("no json", RequestMetrics(), "formatting", True),
            ('{"x":2}', RequestMetrics(), "task_failure", True),
            ('{"x":1}', RequestMetrics(finish_reason="length"), "truncation", True),
            ("", RequestMetrics(ok=False, error="HTTP 500"), "endpoint_error", False),
        ]
        results = []
        for response, metrics, outcome, scored in cases:
            r = score_response(self.q, response, TokenUsage(), metrics)
            self.assertEqual((r.outcome, r.is_scored), (outcome, scored))
            results.append(r)
        q = replace(self.q, metadata={"context_tokens": 8192})
        r = score_response(
            q, "", TokenUsage(), RequestMetrics(ok=False, error="HTTP 400 context_length_exceeded")
        )
        self.assertEqual(r.outcome, "unsupported_context")
        self.assertFalse(r.is_scored)
        broken = score_response(
            replace(self.q, evaluator="absent"), "{}", TokenUsage(), RequestMetrics()
        )
        self.assertEqual(broken.outcome, "evaluator_error")
        self.assertFalse(broken.is_scored)
        summary = summarize([result_record(r) for r in results + [r, broken]])
        self.assertEqual(summary["scored"], 4)
        self.assertEqual(len(CategoryResult("test", results + [r, broken]).scored), 4)

    def test_category_balance_compliance_cost_and_pairing(self):
        good = self.result()
        bad = replace(
            self.result(0),
            question=replace(self.q, id="b", category="Code", metadata={"family": "g"}),
        )
        art = replace(
            self.result(0), question=replace(self.q, id="art", category="Creative Writing")
        )
        cached = replace(good, cached=True)
        rows = [result_record(x) for x in [good, bad, art, cached]]
        summary = summarize(rows, 2, 4)
        self.assertEqual(summary["category_balanced"], 0.5)
        self.assertEqual(summary["compliance_score"], 0)
        self.assertAlmostEqual(summary["estimated_cost_usd"], (300 * 2 + 60 * 4) / 1e6)
        self.assertIsNone(summary["category_balanced_ci95"])
        a = make_report([good, bad], QualityConfig(), self.config)
        b = make_report([good, replace(bad, score=1)], QualityConfig(), self.config)
        comparison = paired_comparison(a, b)
        self.assertTrue(comparison["compatible"])
        self.assertEqual(comparison["paired"], 2)
        self.assertEqual(comparison["balanced_difference"], 0.5)
        other = copy.deepcopy(b)
        other["protocol"]["temperature"] = 1
        self.assertFalse(paired_comparison(a, other)["compatible"])
        b["results"][1]["scored"] = False
        self.assertEqual(paired_comparison(a, b)["excluded_pairs"], 1)
        self.assertEqual(paired_comparison(a, b)["paired"], 1)

    def test_cli_cache_preserves_truncation(self):
        import benchmark

        client = ChatClient(self.config)
        with (
            patch.object(benchmark, "MODEL", "fake"),
            patch.object(benchmark, "BASE_URL", "http://fake.invalid"),
        ):
            key = f"fake:{self.q.id}:{benchmark.question_fingerprint(self.q)}"
            cache = {
                key: {
                    "response": '{"x":1}',
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "finish_reason": "length",
                }
            }
            result = benchmark.run_one(self.q, client, cache, persist=False)
            self.assertTrue(result.cached)
            self.assertEqual(result.outcome, "truncation")
            self.assertEqual(result.score, 0)

    def test_client_finish_reason_blocking_and_streamed(self):
        class Response:
            status_code = 200

            def json(self):
                return {
                    "choices": [{"message": {"content": '{"x":1}'}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }

            def iter_lines(self):
                for value in [
                    {"choices": [{"delta": {"content": '{"x":1}'}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "length"}]},
                    {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                ]:
                    yield ("data: " + json.dumps(value)).encode()
                yield b"data: [DONE]"

            def close(self):
                pass

        for streaming in [False, True]:
            client = ChatClient(replace(self.config, stream=streaming))
            with patch.object(client.session, "post", return_value=Response()):
                text, tokens, metrics = client.complete("test")
            self.assertEqual(metrics.finish_reason, "length")
            self.assertEqual(tokens.prompt_tokens, 10)
            self.assertFalse(metrics.prompt_tokens_estimated)
            self.assertEqual(score_response(self.q, text, tokens, metrics).outcome, "truncation")

    def test_complete_cli_and_web_runners_agree(self):
        import benchmark
        from app import config as app_config
        from app.services import benchmark_runner as runner
        from types import SimpleNamespace
        import yaml

        class Client(FakeAgent):
            def __init__(self, config):
                super().__init__(None)
                self.config = config
                self.session = SimpleNamespace(close=lambda: None)

            def complete(self, *args, **kwargs):
                return (
                    '{"x":1}',
                    TokenUsage(100, 20),
                    RequestMetrics(latency_ms=10, finish_reason="stop"),
                )

        options = QualityConfig(
            generated=False, strengthen_code=False, interactive=True, seeds=(19,)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "test.db"
            (root / "one.yaml").write_text(
                yaml.safe_dump(
                    [
                        {
                            "id": "one",
                            "category": "Reasoning",
                            "prompt": "Answer x=1",
                            "evaluator": "json_match",
                            "expected": {"value": {"x": 1}},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            db = sqlite3.connect(path)
            db.executescript((ROOT / "app/schema.sql").read_text(encoding="utf-8"))
            model = {
                "id": 1,
                "name": "fake",
                "base_url": "http://fake.invalid",
                "api_key": "unused",
                "model_id": "fake",
            }
            db.execute(
                "INSERT INTO models(id,name,base_url,api_key,model_id) VALUES(1,'fake','http://fake.invalid','unused','fake')"
            )
            db.execute(
                "INSERT INTO test_runs(id,model_id,status,quality_config_json) VALUES(1,1,'pending',?)",
                (json.dumps(asdict(options)),),
            )
            db.commit()
            db.close()
            with (
                patch.object(app_config, "DATABASE_PATH", path),
                patch.object(app_config, "TESTS_DIR", root),
                patch("app.services.url_guard.validate_endpoint"),
                patch.object(runner, "ChatClient", Client),
            ):
                runner._run_benchmark_impl(
                    run_id=1,
                    model=model,
                    category=None,
                    limit=None,
                    test_ids=None,
                    difficulty=None,
                    workers=1,
                    run_perf=False,
                    concurrency_levels=(1,),
                    run_context=False,
                    context_sizes=(),
                    context_concurrency=1,
                    workload_mix="uniform",
                    shared_prefix=False,
                    slo_ttft_ms=None,
                    slo_tps=None,
                    slo_errors=None,
                    req_per_user_h=None,
                )
            db = sqlite3.connect(path)
            status, raw = db.execute(
                "SELECT status,quality_json FROM test_runs WHERE id=1"
            ).fetchone()
            self.assertEqual(status, "completed")
            web_report = json.loads(raw)
            db.close()
            questions = assemble_questions(load_all_tests(root), options)
            with patch.object(benchmark, "ChatClient", Client):
                categories, _ = benchmark.run_benchmark(questions, self.config, no_cache=True)
            cli_report = make_report(
                [r for c in categories for r in c.results], options, self.config
            )
            self.assertEqual(cli_report["suite_hash"], web_report["suite_hash"])
            self.assertEqual(
                cli_report["summary"]["category_balanced"],
                web_report["summary"]["category_balanced"],
            )
            self.assertEqual(cli_report["summary"]["interactive"]["successes"], 3)
            self.assertEqual(paired_comparison(cli_report, web_report)["balanced_difference"], 0)

    def test_web_form_config_and_comparison_template(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from unittest.mock import AsyncMock
        from app.routes import admin
        from app.templates_config import templates

        app = FastAPI()
        app.include_router(admin.router)
        save = AsyncMock(return_value=1)
        with (
            patch.object(admin, "_admin_required", return_value={"id": 1}),
            patch.object(admin, "fetch_one", new=AsyncMock(return_value={"id": 1})),
            patch.object(admin, "execute", new=save),
            patch("app.services.run_queue.enqueue_run", return_value=True) as enqueue,
        ):
            client = TestClient(app)
            response = client.post(
                "/admin/run",
                data={
                    "model_id": 1,
                    "suite_seeds": "19,23",
                    "suite_split": "evaluation",
                    "quality_context_sizes": "8192,32768",
                    "input_price": "2",
                    "output_price": "4",
                },
                follow_redirects=False,
            )
            self.assertEqual(response.status_code, 302)
            config = json.loads(save.call_args.args[1][-1])
            self.assertEqual(config["seeds"], [19, 23])
            self.assertEqual(config["context_sizes"], [8192, 32768])
            self.assertEqual(config["split"], "evaluation")
            self.assertTrue(enqueue.called)
            save.reset_mock()
            invalid = client.post(
                "/admin/run", data={"model_id": 1, "suite_seeds": "19", "variants": 999}
            )
            self.assertEqual(invalid.status_code, 422)
            save.assert_not_called()
        report = make_report([self.result()], QualityConfig(), self.config)
        run = {
            "id": 1,
            "model_name": "fake",
            "model_id": "fake",
            "avg_score": 0,
            "passed_questions": 0,
            "scored_questions": 0,
            "total_questions": 7,
            "test_suite_hash": report["suite_hash"],
            "quality": report,
            "categories": {"Unscored": {"total": 7, "scored": 0, "passed": 0, "avg_score": None}},
        }
        runs = [run, {**run, "id": 2}, {**run, "id": 3, "quality": None}]
        rendered = templates.env.get_template("compare.html").render(
            request={"url": {"path": "/compare"}},
            user=None,
            completed_runs=runs,
            selected_runs=runs,
            selected_ids=["1", "2", "3"],
            comparison_data=[{"category": "Unscored", "results": [{"test_id": "Q1"}]}],
            quality_comparisons=[
                {"left": 1, "right": 2, "comparison": paired_comparison(report, report)}
            ],
        )
        self.assertIn("identical questions scored by both", rendered)
        self.assertIn("(0/0)", rendered)
        self.assertNotIn("(0/7)", rendered)
        self.assertIn("Test not executed or excluded from quality scoring", rendered)
        # Older runs remain readable without new diagnostics.
        rendered = templates.env.get_template("quality_summary.html").render(
            quality=None, run={"id": 1}
        )
        self.assertNotIn("Quality diagnostics", rendered)

    def test_web_database_and_reporting_integration(self):
        from app.services import benchmark_runner as runner
        from app import config as app_config
        from app.database import _apply_migrations
        import aiosqlite

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.db"
            db = sqlite3.connect(path)
            db.executescript((ROOT / "app/schema.sql").read_text(encoding="utf-8"))
            db.execute(
                "INSERT INTO models(id,name,base_url,api_key,model_id) VALUES(1,'fake','http://fake.invalid','unused','fake')"
            )
            options = QualityConfig(generated=False, interactive=False, strengthen_code=False)
            db.execute(
                "INSERT INTO test_runs(id,model_id,status,quality_config_json) VALUES(1,1,'running',?)",
                (json.dumps(asdict(options)),),
            )
            db.commit()
            db.close()

            async def migrate():
                async with aiosqlite.connect(path) as conn:
                    await _apply_migrations(conn)
                    await _apply_migrations(conn)
                    await conn.commit()

            asyncio.run(migrate())
            with patch.object(app_config, "DATABASE_PATH", path):
                good = self.result()
                broken = replace(
                    self.result(0), outcome="evaluator_error", detail="Evaluator error: example"
                )
                runner._store_result(1, 1, good)
                runner._store_result(1, 2, broken)
                runner._summarise_and_finish(1, [good, broken], 2, 1, 100, None, "")
            db = sqlite3.connect(path)
            run = db.execute(
                "SELECT status,scored_questions,quality_json FROM test_runs WHERE id=1"
            ).fetchone()
            self.assertEqual(run[:2], ("completed", 1))
            report = json.loads(run[2])
            self.assertEqual(report["summary"]["scored"], 1)
            stored = db.execute(
                "SELECT request_ok,quality_scored,quality_metadata_json FROM test_results ORDER BY question_index"
            ).fetchall()
            self.assertEqual(stored[1][:2], (1, 0))
            self.assertEqual(json.loads(stored[1][2])["outcome"], "evaluator_error")
            db.close()
            from app.templates_config import templates

            rendered = templates.env.get_template("quality_summary.html").render(
                quality=report, run={"id": 1}
            )
            self.assertIn("Category/family-balanced capability", rendered)
            self.assertIn("evaluator error: 1", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
