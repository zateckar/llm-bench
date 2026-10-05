"""A small, exact JSON Schema validator for the subset the conformance suite uses.

Supported keywords: type (string or list, including "null"), properties,
required, additionalProperties (boolean), enum, items, minimum, maximum,
minItems, maxItems, minLength, maxLength, description and title. Suite
validation rejects any other keyword, so the grader never silently ignores a
constraint shown to the model.
"""

import math

SUPPORTED = {
    "type", "properties", "required", "additionalProperties", "enum", "items",
    "minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength",
    "description", "title",
}
TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def unsupported_keywords(schema, path="$"):
    """Return ``[(path, keyword)]`` for constructs outside the subset."""
    problems = []
    if not isinstance(schema, dict):
        return [(path, "<non-object schema>")]
    for key in schema:
        if key not in SUPPORTED:
            problems.append((path, key))
    kinds = schema.get("type")
    for kind in kinds if isinstance(kinds, list) else [kinds]:
        if kind is not None and kind not in TYPES:
            problems.append((path, f"type:{kind}"))
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        problems.append((path, "additionalProperties:<schema>"))
    for name, child in (schema.get("properties") or {}).items():
        problems.extend(unsupported_keywords(child, f"{path}.{name}"))
    if "items" in schema:
        problems.extend(unsupported_keywords(schema["items"], f"{path}[]"))
    return problems


def _type_matches(value, kind):
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return type(value) is int
    if kind == "number":
        return type(value) in (int, float) and math.isfinite(value)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    return False


def validate(value, schema, path="$"):
    """Return a list of ``{"path", "error"}`` violations (empty when valid)."""
    errors = []
    kinds = schema.get("type")
    if kinds is not None:
        allowed = kinds if isinstance(kinds, list) else [kinds]
        if not any(_type_matches(value, kind) for kind in allowed):
            return [{"path": path, "error": f"expected {'/'.join(allowed)}"}]
    if "enum" in schema and not any(
        type(value) is type(option) and value == option for option in schema["enum"]
    ):
        errors.append({"path": path, "error": "not an allowed value"})
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append({"path": path, "error": f"below minimum {schema['minimum']}"})
        if "maximum" in schema and value > schema["maximum"]:
            errors.append({"path": path, "error": f"above maximum {schema['maximum']}"})
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append({"path": path, "error": "string too short"})
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append({"path": path, "error": "string too long"})
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append({"path": path, "error": "too few items"})
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append({"path": path, "error": "too many items"})
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(validate(item, schema["items"], f"{path}[{index}]"))
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                errors.append({"path": f"{path}.{name}", "error": "required property missing"})
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append({"path": f"{path}.{name}", "error": "unexpected property"})
        for name, child in properties.items():
            if name in value:
                errors.extend(validate(value[name], child, f"{path}.{name}"))
    return errors
