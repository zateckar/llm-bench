"""Explicit projections keep compressed audit payloads off interactive hot paths."""

RUN_FIELDS = """id,model_id,status,created_at,started_at,completed_at,total_questions,
passed_questions,avg_score,error_message,created_by,test_suite_hash,total_prompt_tokens,
total_completion_tokens,weighted_score,scored_questions,error_count,workers,duration_ms,
latency_p50_ms,latency_p95_ms,latency_p99_ms,ttft_p50_ms,ttft_p95_ms,output_tokens_per_sec,
perf_json,quality_summary_json,quality_config_json,run_options_json,decoding_config_json,
metrics_config_json,plan_id,repeat_group_id,repeat_index,repeat_count,canary_id,canary_status,canary_json"""


def run_columns(include_answers=False):
    fields = RUN_FIELDS.replace("\n", "").split(",")
    if include_answers:
        fields.append("quality_json")
    return ",".join(f"tr.{f}" for f in fields)


def result_columns(include_answers=False):
    if include_answers:
        return "*"
    return """id,run_id,test_id,category,score,detail,evaluator,question_index,
        prompt_tokens,completion_tokens,passed,pass_threshold,difficulty,weight,
        latency_ms,ttft_ms,request_ok,quality_scored,quality_outcome,
        COALESCE(prompt_preview,'') AS prompt"""
