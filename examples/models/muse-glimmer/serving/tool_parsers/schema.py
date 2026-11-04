# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local JSON Schema validation and lossless ATEM scalar conversion."""

import json
import math
import re
from itertools import chain
from urllib.parse import unquote

from executorch.examples.llm_server.python.errors import APIError
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry
from referencing.jsonschema import DRAFT202012


class ToolOutputError(APIError):
    def __init__(self, message):
        super().__init__(502, message, "server_error", "invalid_tool_output")


class ToolSchema:
    def __init__(self, schema):
        self.schema = schema
        try:
            Draft202012Validator.check_schema(schema)
            resources = [DRAFT202012.create_resource(schema)]
            while resources:
                resource = resources.pop()
                node = resource.contents
                if isinstance(node, dict):
                    if (
                        node.get(
                            "$schema", "https://json-schema.org/draft/2020-12/schema"
                        ).rstrip("#")
                        != "https://json-schema.org/draft/2020-12/schema"
                    ):
                        raise ValueError("only JSON Schema Draft 2020-12 is supported")
                    if "$dynamicRef" in node or "$id" in node:
                        raise ValueError("$dynamicRef and $id are unsupported")
                    if "$ref" in node:
                        self._resolve(node["$ref"])
                resources.extend(resource.subresources())
        except (SchemaError, ValueError, KeyError, IndexError, TypeError) as error:
            raise APIError(
                400,
                "Invalid or unsupported tool schema: " + str(error).splitlines()[0],
                "invalid_request_error",
                "invalid_tool_schema",
            ) from error
        # Registry has no retrieval hook: validation never fetches remote schemas.
        self.validator = Draft202012Validator(schema, registry=Registry())

    def _resolve(self, ref):
        if ref == "#":
            return self.schema
        if not ref.startswith("#/"):
            raise ValueError("tool schema references must be local JSON pointers")
        node = self.schema
        for part in unquote(ref[2:]).split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            node = node[int(part)] if isinstance(node, list) else node[part]
        if not isinstance(node, (dict, bool)):
            raise ValueError("tool schema reference must point to a schema")
        return node

    def _property(self, schema, key, seen=()):
        if not isinstance(schema, dict) or id(schema) in seen:
            return {}
        seen = (*seen, id(schema))
        pieces = []
        if "$ref" in schema:
            pieces.append(self._property(self._resolve(schema["$ref"]), key, seen))
        if key in schema.get("properties", {}):
            pieces.append(schema["properties"][key])
        matched = key in schema.get("properties", {})
        for pattern, child in schema.get("patternProperties", {}).items():
            if re.search(pattern, key):
                pieces.append(child)
                matched = True
        if not matched and "additionalProperties" in schema:
            pieces.append(schema["additionalProperties"])
        for keyword in ("anyOf", "oneOf", "allOf"):
            if keyword in schema:
                # A property can occur in more than one oneOf branch even when
                # complete objects match exactly one branch.
                pieces.append(
                    {
                        "anyOf" if keyword == "oneOf" else keyword: [
                            self._property(child, key, seen)
                            for child in schema[keyword]
                        ]
                    }
                )
        return {"allOf": pieces} if pieces else {}

    def _declares(self, schema, kind, seen=()):
        if not isinstance(schema, dict) or id(schema) in seen:
            return False
        seen = (*seen, id(schema))
        declared = schema.get("type", [])
        if kind == declared or isinstance(declared, list) and kind in declared:
            return True
        if kind == "string" and (
            isinstance(schema.get("const"), str)
            or any(isinstance(value, str) for value in schema.get("enum", []))
        ):
            return True
        if "$ref" in schema and self._declares(
            self._resolve(schema["$ref"]), kind, seen
        ):
            return True
        return any(
            self._declares(child, kind, seen)
            for keyword in ("anyOf", "oneOf", "allOf")
            for child in schema.get(keyword, [])
        )

    def coerce(self, key, value, *, schema=None):
        schema = self._property(self.schema if schema is None else schema, key)
        validator = self.validator.evolve(schema=schema)
        stripped = value.strip()
        if (
            stripped == "null"
            and self._declares(schema, "null")
            and validator.is_valid(None)
        ):
            return None
        if self._declares(schema, "string") and validator.is_valid(value):
            return value
        if self._declares(schema, "boolean") and stripped.lower() in ("true", "false"):
            return stripped.lower() == "true"
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return value
        if isinstance(parsed, str):
            return value
        if isinstance(parsed, float) and not math.isfinite(parsed):
            return value
        return parsed

    def _object_variants(self, schema, keys, seen=()):
        if not isinstance(schema, dict) or id(schema) in seen:
            yield schema
            return
        seen = (*seen, id(schema))
        groups = []
        if "$ref" in schema:
            groups.append([self._resolve(schema["$ref"])])
        groups.extend([child] for child in schema.get("allOf", []))
        for keyword in ("anyOf", "oneOf"):
            if keyword in schema:
                groups.append(schema[keyword])
        if "if" in schema:
            groups.append(
                [
                    {"allOf": [schema["if"], schema.get("then", {})]},
                    {"allOf": [{"not": schema["if"]}, schema.get("else", {})]},
                ]
            )
        groups.extend(
            [child]
            for key, child in schema.get("dependentSchemas", {}).items()
            if key in keys
        )
        base = {
            key: value
            for key, value in schema.items()
            if key
            not in (
                "$ref",
                "allOf",
                "anyOf",
                "oneOf",
                "if",
                "then",
                "else",
                "dependentSchemas",
            )
        }
        yield from self._combine_variants([base], groups, keys, seen)

    def _combine_variants(self, parts, groups, keys, seen):
        if not groups:
            yield {"allOf": parts}
            return
        for choice in groups[0]:
            for variant in self._object_variants(choice, keys, seen):
                yield from self._combine_variants(
                    parts + [variant], groups[1:], keys, seen
                )

    def coerce_arguments(self, name, raw):
        variants = chain([self.schema], self._object_variants(self.schema, raw.keys()))
        last_error = None
        for index, schema in enumerate(variants):
            if index >= 1024:
                raise ToolOutputError(
                    "Tool argument coercion exceeded 1024 schema alternatives"
                )
            arguments = {
                key: self.coerce(key, value, schema=schema)
                for key, value in raw.items()
            }
            try:
                self.validate(name, arguments)
                return arguments
            except ToolOutputError as error:
                last_error = error
        raise last_error or ToolOutputError(
            f"Tool {name!r} has no valid argument schema"
        )

    def validate(self, name, arguments):
        try:
            json.dumps(arguments, allow_nan=False)
            error = next(self.validator.iter_errors(arguments), None)
        except (ValueError, RecursionError) as error:
            raise ToolOutputError(
                f"Tool {name!r} returned invalid arguments"
            ) from error
        if error is not None:
            path = "/" + "/".join(str(part) for part in error.absolute_path)
            raise ToolOutputError(
                f"Tool {name!r} arguments violate {error.validator!r} at {path}"
            )
