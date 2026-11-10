"""Statically check that the snapshot's first-party imports resolve inside it.

Usage: python tools/check_closure.py [SOURCE_WORKSPACE]

Walks every Python file in the snapshot, including imports nested in functions,
``importlib.import_module("...")`` literals and dotted ``"pkg.mod.Name"`` string
literals. A first-party module that is missing from the snapshot is reported with
its location; if SOURCE_WORKSPACE is given, the report says whether the module
exists there (i.e. was trimmed) or nowhere (pre-existing dangling import).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS = ("RealWorldInference", "ss-soma", "ss-falcon", "ss-morph", "ss-modelzoo")
FIRST_PARTY = {"policy", "hardware", "slave", "utils", "soma", "falcon", "morph", "modelzoo"}
DOTTED = re.compile(r"^([a-z_][a-z0-9_]*)(\.[A-Za-z_][A-Za-z0-9_]*)+$")


def module_exists(roots: list[Path], name: str) -> bool:
    rel = Path(*name.split("."))
    for root in roots:
        if (root / rel).with_suffix(".py").is_file() or (root / rel).is_dir():
            return True
    return False


def resolve_relative(module: str, level: int, path: Path, roots: list[Path]) -> str:
    for root in roots:
        try:
            parts = list(path.relative_to(root).with_suffix("").parts)
        except ValueError:
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        else:
            parts = parts[:-1]
        base = parts[: len(parts) - (level - 1)] if level > 1 else parts
        return ".".join(base + ([module] if module else []))
    raise ValueError(path)


def collect(path: Path, roots: list[Path]):
    tree = ast.parse(path.read_text(), str(path))
    nested = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is not node:
                    nested.add(id(child))
    for node in ast.walk(tree):
        lazy = id(node) in nested
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, lazy, ()
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = resolve_relative(name, node.level, path, roots)
            yield node.lineno, name, lazy, tuple(a.name for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if DOTTED.match(node.value) and node.value.split(".")[0] in FIRST_PARTY:
                yield node.lineno, node.value, True, "string"


def main() -> int:
    snap_roots = [ROOT / r for r in REPOS]
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    src_roots = [src / r for r in REPOS] if src else []
    missing: dict[str, list[str]] = {}
    for repo in snap_roots:
        for path in sorted(repo.rglob("*.py")):
            for line, name, lazy, names in collect(path, snap_roots):
                if not name or name.split(".")[0] not in FIRST_PARTY:
                    continue
                candidates = [name]
                if names == "string":
                    # "pkg.mod.Attr" — the module is some prefix.
                    parts = name.split(".")
                    candidates = [".".join(parts[:i]) for i in range(len(parts), 1, -1)]
                    if any(module_exists(snap_roots, c) for c in candidates):
                        continue
                    if not any(module_exists(src_roots, c) for c in candidates) and src:
                        continue  # not a module path at all
                    candidates = candidates[:1]
                elif names:
                    # from X import y: y may be a submodule.
                    candidates += [
                        f"{name}.{n}" for n in names
                        if src and module_exists(src_roots, f"{name}.{n}")
                    ]
                for mod in candidates:
                    if module_exists(snap_roots, mod):
                        continue
                    where = "trimmed" if src and module_exists(src_roots, mod) else "absent-in-source"
                    tag = "lazy" if lazy else "top-level"
                    missing.setdefault(mod, []).append(
                        f"{path.relative_to(ROOT)}:{line} [{tag}, {where}]"
                    )
    for mod in sorted(missing):
        print(mod)
        for loc in missing[mod]:
            print("   ", loc)
    print(f"\n{len(missing)} unresolved first-party modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
