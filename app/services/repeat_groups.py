"""Load repeat groups and compare them as units."""

from app.database import fetch_all
from app.benchmarking.quality_report import compare_groups, rescore_report
from app.benchmarking.repeat_stats import group_members, group_summary
from app.benchmarking.suites import DEFAULT_SUITE
from app.services.html_reports import _adjust_for_multiplicity, object_json

MAX_COMPARED_GROUPS = 8


async def load_group(group_id: int):
    if not 0 < group_id <= 2**63 - 1:
        return None
    rows = await fetch_all(
        """SELECT tr.id, tr.status, tr.quality_json, tr.repeat_index, tr.repeat_count,
                  tr.run_options_json, tr.created_at, m.name AS model_name,
                  m.model_id AS provider_model_id
             FROM test_runs tr JOIN models m ON m.id = tr.model_id
            WHERE tr.repeat_group_id = ?
            ORDER BY tr.repeat_index, tr.id""",
        (group_id,),
    )
    if not rows:
        return None
    entries = []
    for row in rows:
        report = object_json(row.pop("quality_json"))
        if report.get("schema_version") == 3:
            report = rescore_report(report)
        entries.append({"run_id": row["id"], "status": row["status"], "report": report})
    first = rows[0]
    return {
        "id": group_id,
        "model_name": first["model_name"],
        "provider_model_id": first["provider_model_id"],
        "suite": object_json(first.get("run_options_json")).get("suite", DEFAULT_SUITE),
        "planned": first.get("repeat_count") or len(rows),
        "runs": rows,
        "entries": entries,
        "summary": group_summary(entries),
    }


async def other_groups(group_id: int):
    """Selectable comparison partners, newest first."""
    return await fetch_all(
        """SELECT tr.repeat_group_id AS id, MIN(m.name) AS model_name, COUNT(*) AS runs,
                  SUM(tr.status = 'completed') AS completed, MAX(tr.run_options_json) AS run_options_json
             FROM test_runs tr JOIN models m ON m.id = tr.model_id
            WHERE tr.repeat_group_id IS NOT NULL AND tr.repeat_group_id != ?
            GROUP BY tr.repeat_group_id ORDER BY tr.repeat_group_id DESC""",
        (group_id,),
    )


def compare_to(baseline, others):
    """Baseline-minus-each comparisons of usable members, Holm-adjusted together."""
    left = [entry["report"] for entry in group_members(baseline["entries"])[0]]
    rows = []
    for other in others:
        right = [entry["report"] for entry in group_members(other["entries"])[0]]
        rows.append({"left": baseline["id"], "right": other["id"], "right_model": other["model_name"],
                     "comparison": compare_groups(left, right)})
    _adjust_for_multiplicity([row["comparison"] for row in rows])
    return rows
