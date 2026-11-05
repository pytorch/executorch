# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local JSON Schema validation and lossless ATEM scalar conversion."""

import json
import math
import re
from itertools import chain

from executorch.examples.llm_server.python.errors import APIError
from jsonschema import Draft201909Validator, Draft202012Validator, Draft7Validator
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT201909, DRAFT202012, DRAFT7


class ToolOutputError(APIError):
    def __init__(self, message):
        super().__init__(502, message, "server_error", "invalid_tool_output")


class _ToolWorkLimit(ToolOutputError):
    pass


class ToolOutputBudget:
    def __init__(self):
        self.remaining = 65536

    def spend(self, visits=1):
        self.remaining -= visits
        if self.remaining < 0:
            raise _ToolWorkLimit(
                "Tool argument processing exceeded its traversal budget"
            )

    def validation(self, schema_nodes, value):
        pending = [value]
        while pending:
            value = pending.pop()
            self.spend(schema_nodes)
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)


class ToolSchema:
    def __init__(self, schema):
        self.schema = schema
        try:
            cls = validator_for(
                schema,
                default=Draft202012Validator if "$schema" not in schema else None,
            )
            specs = {
                Draft7Validator: DRAFT7,
                Draft201909Validator: DRAFT201909,
                Draft202012Validator: DRAFT202012,
            }
            if cls not in specs:
                raise ValueError(
                    "supported JSON Schema drafts are 7, 2019-09 and 2020-12"
                )
            self._legacy_ref = cls is Draft7Validator
            cls.check_schema(schema)
            root = Resource.from_contents(schema, default_specification=specs[cls])
            base = root.id() or ""
            registry = Registry().with_resource(base, root).crawl()
            self._resolver = registry.resolver(base)
            resources = [root]
            while resources:
                resource = resources.pop()
                node = resource.contents
                if isinstance(node, dict):
                    if node is not schema and ("$id" in node or "$schema" in node):
                        raise ValueError(
                            "nested schema identifiers and dialect changes are unsupported"
                        )
                    if "$dynamicRef" in node or "$recursiveRef" in node:
                        raise ValueError(
                            "dynamic and recursive references are unsupported"
                        )
                    if "$ref" in node:
                        self._resolve(node["$ref"])
                resources.extend(resource.subresources())
            self._schema_nodes, keys = self._check_reference_work(root, specs[cls])
            if self._variant_count(schema) > 1023:
                raise ValueError("tool schema exceeds 1023 coercion alternatives")
            try:
                self._check_coercion_work(keys)
            except _ToolWorkLimit as error:
                raise ValueError(
                    "tool schema exceeds 65536 estimated coercion work visits"
                ) from error
        except (
            SchemaError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            RecursionError,
            Unresolvable,
        ) as error:
            raise APIError(
                400,
                "Invalid or unsupported tool schema: " + str(error).splitlines()[0],
                "invalid_request_error",
                "invalid_tool_schema",
            ) from error
        self.validator = cls(schema, registry=registry)

    def _resolve(self, ref):
        if not ref.startswith("#"):
            raise ValueError("tool schema references must be local fragments")
        node = self._resolver.lookup(ref).contents
        if not isinstance(node, (dict, bool)):
            raise ValueError("tool schema reference must point to a schema")
        return node

    def _check_reference_work(self, root, specification):
        pending = [(root, ())]
        visits = 0
        keys = {None}
        while pending:
            resource, ancestors = pending.pop()
            visits += 1
            if visits > 4096:
                raise ValueError("tool schema exceeds 4096 expanded schema nodes")
            node = resource.contents
            if not isinstance(node, dict) or id(node) in ancestors:
                continue
            ancestors = (*ancestors, id(node))
            for keyword in ("properties", "dependentSchemas", "dependencies"):
                keys.update(node.get(keyword, {}))
            if "$ref" in node:
                referenced = Resource.from_contents(
                    self._resolve(node["$ref"]), default_specification=specification
                )
                pending.append((referenced, ancestors))
            pending.extend((child, ancestors) for child in resource.subresources())
        return visits, keys

    def _check_coercion_work(self, keys):
        # Walk all alternatives and declared keys before generation, including a
        # wildcard projection for map properties. Charge worst-case scalar
        # validation too; nested values and extra keys consume the output budget.
        budget = ToolOutputBudget()
        for variant in chain(
            [self.schema], self._object_variants(self.schema, keys, budget=budget)
        ):
            for key in keys:
                projected = self._property(
                    variant, key, budget=budget, all_patterns=True
                )
                for kind in ("null", "string", "boolean", "integer", "number"):
                    self._declares(projected, kind, budget=budget)
                budget.spend(2 * 4 * self._schema_nodes)
            budget.spend(self._schema_nodes * (len(keys) + 1))

    def _variant_count(self, schema, seen=()):
        if not isinstance(schema, dict) or id(schema) in seen:
            return 1
        seen = (*seen, id(schema))
        count = 1
        if "$ref" in schema:
            count = self._variant_count(self._resolve(schema["$ref"]), seen)
            if self._legacy_ref:
                return count
        children = list(schema.get("allOf", []))
        children.extend(schema.get("dependentSchemas", {}).values())
        children.extend(
            v for v in schema.get("dependencies", {}).values() if isinstance(v, dict)
        )
        for child in children:
            count *= self._variant_count(child, seen)
            if count > 1023:
                return count
        for keyword in ("anyOf", "oneOf"):
            if keyword in schema:
                count *= sum(
                    self._variant_count(child, seen) for child in schema[keyword]
                )
        if "if" in schema:
            count *= self._variant_count(schema["if"], seen) * self._variant_count(
                schema.get("then", {}), seen
            ) + self._variant_count(schema.get("else", {}), seen)
        return count

    def _property(self, schema, key, seen=(), *, budget, all_patterns=False):
        budget.spend()
        if not isinstance(schema, dict) or id(schema) in seen:
            return {}
        seen = (*seen, id(schema))
        pieces = []
        if "$ref" in schema and self._legacy_ref:
            return self._property(
                self._resolve(schema["$ref"]),
                key,
                seen,
                budget=budget,
                all_patterns=all_patterns,
            )
        if "$ref" in schema:
            pieces.append(
                self._property(
                    self._resolve(schema["$ref"]),
                    key,
                    seen,
                    budget=budget,
                    all_patterns=all_patterns,
                )
            )
        if key in schema.get("properties", {}):
            pieces.append(schema["properties"][key])
        matched = key in schema.get("properties", {})
        for pattern, child in schema.get("patternProperties", {}).items():
            if all_patterns or re.search(pattern, key):
                pieces.append(child)
                matched = True
        if (all_patterns or not matched) and "additionalProperties" in schema:
            pieces.append(schema["additionalProperties"])
        for keyword in ("anyOf", "oneOf", "allOf"):
            if keyword in schema:
                # A property can occur in more than one oneOf branch even when
                # complete objects match exactly one branch.
                pieces.append(
                    {
                        "anyOf" if keyword == "oneOf" else keyword: [
                            self._property(
                                child,
                                key,
                                seen,
                                budget=budget,
                                all_patterns=all_patterns,
                            )
                            for child in schema[keyword]
                        ]
                    }
                )
        return {"allOf": pieces} if pieces else {}

    def _declares(self, schema, kind, seen=(), *, budget):
        budget.spend()
        if not isinstance(schema, dict) or id(schema) in seen:
            return False
        seen = (*seen, id(schema))
        if "$ref" in schema and self._legacy_ref:
            return self._declares(
                self._resolve(schema["$ref"]), kind, seen, budget=budget
            )
        declared = schema.get("type", [])
        if kind == declared or isinstance(declared, list) and kind in declared:
            return True
        if kind == "string" and (
            isinstance(schema.get("const"), str)
            or any(isinstance(value, str) for value in schema.get("enum", []))
        ):
            return True
        if "$ref" in schema and self._declares(
            self._resolve(schema["$ref"]), kind, seen, budget=budget
        ):
            return True
        return any(
            self._declares(child, kind, seen, budget=budget)
            for keyword in ("anyOf", "oneOf", "allOf")
            for child in schema.get(keyword, [])
        )

    def _valid_scalar(self, schema, value, budget):
        # Property projection introduces composition wrappers. Charge a
        # conservative estimate of their validation work before calling jsonschema.
        budget.spend(4 * self._schema_nodes)
        return self.validator.evolve(schema=schema).is_valid(value)

    def coerce(self, key, value, *, schema=None, budget=None):
        budget = budget if budget is not None else ToolOutputBudget()
        schema = self._property(
            self.schema if schema is None else schema, key, budget=budget
        )
        stripped = value.strip()
        if (
            stripped == "null"
            and self._declares(schema, "null", budget=budget)
            and self._valid_scalar(schema, None, budget)
        ):
            return None
        if self._declares(schema, "string", budget=budget) and self._valid_scalar(
            schema, value, budget
        ):
            return value
        if self._declares(schema, "boolean", budget=budget) and stripped.lower() in (
            "true",
            "false",
        ):
            return stripped.lower() == "true"
        try:
            parsed = json.loads(stripped)
        except RecursionError as error:
            raise ToolOutputError("Tool parameter JSON is nested too deeply") from error
        except ValueError:
            numeric = self._declares(
                schema, "integer", budget=budget
            ) or self._declares(schema, "number", budget=budget)
            if not numeric or not re.fullmatch(
                r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", stripped
            ):
                return value
            try:
                parsed = (
                    float(stripped)
                    if any(c in stripped for c in ".eE")
                    else int(stripped)
                )
            except ValueError:
                return value
        if isinstance(parsed, str):
            return value
        if isinstance(parsed, float) and not math.isfinite(parsed):
            return value
        return parsed

    def _object_variants(self, schema, keys, seen=(), *, budget):
        budget.spend()
        if not isinstance(schema, dict) or id(schema) in seen:
            yield schema
            return
        seen = (*seen, id(schema))
        groups = []
        if "$ref" in schema and self._legacy_ref:
            yield from self._object_variants(
                self._resolve(schema["$ref"]), keys, seen, budget=budget
            )
            return
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
        groups.extend(
            [child]
            for key, child in schema.get("dependencies", {}).items()
            if key in keys and isinstance(child, dict)
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
                "dependencies",
            )
        }
        yield from self._combine_variants([base], groups, keys, seen, budget=budget)

    def _combine_variants(self, parts, groups, keys, seen, *, budget):
        budget.spend()
        if not groups:
            yield {"allOf": parts}
            return
        for choice in groups[0]:
            for variant in self._object_variants(choice, keys, seen, budget=budget):
                yield from self._combine_variants(
                    parts + [variant], groups[1:], keys, seen, budget=budget
                )

    def coerce_arguments(self, name, raw, *, budget=None):
        budget = budget if budget is not None else ToolOutputBudget()
        variants = chain(
            [self.schema], self._object_variants(self.schema, raw.keys(), budget=budget)
        )
        last_error = None
        for index, schema in enumerate(variants):
            if index >= 1024:
                raise ToolOutputError(
                    "Tool argument coercion exceeded 1024 schema alternatives"
                )
            arguments = {
                key: self.coerce(key, value, schema=schema, budget=budget)
                for key, value in raw.items()
            }
            try:
                self.validate(name, arguments, budget=budget)
                return arguments
            except _ToolWorkLimit:
                raise
            except ToolOutputError as error:
                last_error = error
        raise last_error or ToolOutputError(
            f"Tool {name!r} has no valid argument schema"
        )

    def validate(self, name, arguments, *, budget=None):
        budget = budget if budget is not None else ToolOutputBudget()
        budget.validation(self._schema_nodes, arguments)
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
