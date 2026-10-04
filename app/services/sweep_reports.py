"""Hydrate incremental sweep reports for pages and self-contained downloads."""

import json

from app.database import fetch_all


async def hydrate_sweep(run_id, perf):
    if perf and perf.get("schema_version") == 4 and perf.get("kind") == "context_sweep":
        rows = await fetch_all(
            "SELECT result_json FROM performance_cells WHERE run_id=? ORDER BY effort,context_tokens,concurrency",
            (run_id,),
        )
        perf["cells"] = [json.loads(row["result_json"]) for row in rows]
        # Counters in the last metadata checkpoint can lag the last cell after
        # a manual stop or process exit. Recompute measured values from storage.
        perf["measured_cells"] = sum(bool(cell.get("requests")) for cell in perf["cells"])
        perf["successful_requests"] = sum(cell.get("completed", 0) for cell in perf["cells"])
    return perf
