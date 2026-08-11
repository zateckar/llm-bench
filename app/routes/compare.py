"""Compare runs route."""

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from app.auth import get_current_user
from app.database import fetch_all, fetch_one
from app.templates_config import templates

router = APIRouter()


@router.get("/compare")
async def compare_page(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    completed_runs = await fetch_all(
        """SELECT tr.*, m.name as model_name, m.model_id
           FROM test_runs tr
           JOIN models m ON tr.model_id = m.id
           WHERE tr.status = 'completed'
           ORDER BY tr.id DESC"""
    )

    # The multi-select form submits repeated params (?runs=8&runs=9), not a
    # comma-separated list. `.get()` would read only the first value and
    # comparison would silently do nothing.
    selected_ids = ",".join(request.query_params.getlist("runs"))
    selected_runs = []
    comparison_data = []

    if selected_ids:
        ids = []
        seen = set()
        for x in selected_ids.split(","):
            trimmed = x.strip()
            if trimmed.isdigit() and int(trimmed) not in seen:
                seen.add(int(trimmed))
                ids.append(int(trimmed))

        for rid in ids:
            run = await fetch_one(
                """SELECT tr.*, m.name as model_name, m.model_id
                   FROM test_runs tr JOIN models m ON tr.model_id = m.id
                   WHERE tr.id = ?""",
                (rid,),
            )
            if run:
                categories = await fetch_all(
                    """SELECT category,
                              COUNT(*) as total,
                              SUM(request_ok) as scored,
                              SUM(passed) as passed,
                              AVG(CASE WHEN request_ok = 1 THEN score END) as avg_score,
                              AVG(latency_ms) as avg_latency_ms
                       FROM test_results WHERE run_id = ?
                       GROUP BY category ORDER BY category""",
                    (rid,),
                )
                run["categories"] = {c["category"]: c for c in categories}
                selected_runs.append(run)

        if len(selected_runs) >= 2:
            all_categories = set()
            for r in selected_runs:
                all_categories.update(r["categories"].keys())

            # Optimize details fetching: query everything in one shot
            selected_run_ids = [r["id"] for r in selected_runs]
            placeholders = ",".join("?" for _ in selected_run_ids)
            all_rows = await fetch_all(
                f"SELECT run_id, category, test_id, score, detail FROM test_results WHERE run_id IN ({placeholders})",
                tuple(selected_run_ids)
            )

            # Reorganize in-memory: run_results[run_id][category][test_id]
            run_results = {rid: {} for rid in selected_run_ids}
            for row in all_rows:
                rid = row["run_id"]
                cat = row["category"]
                tid = row["test_id"]
                if cat not in run_results[rid]:
                    run_results[rid][cat] = {}
                run_results[rid][cat][tid] = row

            for cat in sorted(all_categories):
                cat_results = []
                all_test_ids = set()
                for rid in selected_run_ids:
                    if cat in run_results[rid]:
                        all_test_ids.update(run_results[rid][cat].keys())

                for tid in sorted(all_test_ids):
                    entry = {"test_id": tid}
                    for run in selected_runs:
                        rid = run["id"]
                        r = run_results[rid].get(cat, {}).get(tid, {})
                        # None marks "not executed in this run"; a 0 here would
                        # read as a genuinely zero score and skew the comparison.
                        entry[f"run_{rid}_score"] = r.get("score")
                        entry[f"run_{rid}_detail"] = r.get("detail", "")
                    cat_results.append(entry)

                comparison_data.append({"category": cat, "results": cat_results})

    return templates.TemplateResponse(
        request, "compare.html",
        {
            "completed_runs": completed_runs,
            "selected_runs": selected_runs,
            "comparison_data": comparison_data,
            "selected_ids": selected_ids,
        },
    )
