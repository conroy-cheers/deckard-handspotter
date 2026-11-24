"""Data-only settings shared by the Studio form and batch command."""
import argparse
import json
from pathlib import Path

# These switches select executable code or control the invocation itself.
PRIVATE = {"help", "out", "node", "chromium", "toolkit", "three", "assets", "soma_worker", "job", "studio_schema", "studio_resolve_job", "viewport", "capture_sample"}
DIRECTORIES = {"body_bank", "foundation_scenes", "garment_library", "soma_assets"}


def job_fields(parser):
    fields = {}
    for action in parser._actions:
        key = action.dest
        if key in PRIVATE:
            continue
        kind = "text"
        if action.type is Path:
            kind = "directory" if key in DIRECTORIES else "file"
        elif isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
            kind = "flag"
        elif action.type is int:
            kind = "integer"
        fields[key] = dict(kind=kind, multiple=action.nargs == "+", choices=action.choices,
                           default=action.default, help=action.help)
        if isinstance(action, argparse.BooleanOptionalAction):
            fields[key]["negatable"] = True
        if hasattr(action, "suggestions"):
            fields[key]["suggestions"] = action.suggestions
    return fields


def job_arguments(job, fields, directory):
    if not isinstance(job, dict):
        raise ValueError("Expected a render job object")
    arguments = []
    for name, value in job.items():
        if name not in fields:
            raise ValueError(f"Unknown render job setting: {name}")
        field = fields[name]
        option = "--" + name.replace("_", "-")
        if field["kind"] == "flag":
            if type(value) is not bool:
                raise ValueError(f"{name} requires true or false")
            if value:
                arguments.append(option)
            elif field.get("negatable"):
                arguments.append("--no-" + option[2:])
            continue
        values = value if field["multiple"] else [value]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} requires a nonempty list")
        arguments.append(option)
        for item in values:
            if field["kind"] == "integer":
                if type(item) is not int:
                    raise ValueError(f"{name} requires an integer")
            elif not isinstance(item, str) or not item.strip() or item.startswith("--"):
                raise ValueError(f"Invalid value for {name}")
            if field["kind"] in ("file", "directory"):
                item = (directory / item).resolve()
            arguments.append(str(item))
    return arguments


def parse_render_args(parser, argv):
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--job", type=Path)
    selected, rest = selector.parse_known_args(argv)
    fields = job_fields(parser)
    if argv == ["--studio-schema"]:
        print(json.dumps(fields, default=str))
        parser.exit()
    if selected.job:
        try:
            job = json.loads(selected.job.read_text())
            rest = job_arguments(job, fields, selected.job.resolve().parent) + rest
        except (OSError, ValueError) as error:
            parser.error(str(error))
    return parser.parse_args(rest)
