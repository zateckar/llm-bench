"""Hydrate incremental sweep reports for pages and self-contained downloads."""

import json

from app.benchmarking.staged_performance import stage
from app.database import fetch_all


async def hydrate_sweep(run_id, perf):
    # A staged report's context stage stores its cells the same way.
    sweep = stage(perf, "context") if isinstance(perf, dict) else None
    if sweep is not None:
        rows = await fetch_all(
            "SELECT result_json FROM performance_cells WHERE run_id=? ORDER BY effort,context_tokens,concurrency",
            (run_id,),
        )
        sweep["cells"] = [json.loads(row["result_json"]) for row in rows]
        # Counters in the last metadata checkpoint can lag the last cell after
        # a manual stop or process exit. Recompute measured values from storage.
        sweep["measured_cells"] = sum(bool(cell.get("requests")) for cell in sweep["cells"])
        sweep["successful_requests"] = sum(cell.get("completed", 0) for cell in sweep["cells"])
    return perf
