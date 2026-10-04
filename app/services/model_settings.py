"""Validated model decoding settings shared by forms and run snapshots."""

from fastapi import HTTPException

from app.benchmarking.llm_client import ClientConfig


def decoding_settings(temperature=0, reasoning_effort=None):
    try:
        if isinstance(temperature, bool):
            raise ValueError("Temperature must be a finite number between 0 and 2")
        if reasoning_effort is not None and not isinstance(reasoning_effort, str):
            raise ValueError("Unsupported reasoning effort")
        value = float(temperature)
        effort = reasoning_effort.strip() if isinstance(reasoning_effort, str) else reasoning_effort
        config = ClientConfig("", "", "", temperature=value, reasoning_effort=effort or None)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"temperature": config.temperature, "reasoning_effort": config.reasoning_effort}
