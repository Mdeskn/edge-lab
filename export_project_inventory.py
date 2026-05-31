"""
Export a recursive project inventory to a Markdown file.

The report contains:
1. A folder/file tree for the whole project.
2. A content section for every text-like file, grouped by path.

Binary files are listed in the tree and content section with a short note instead
of raw bytes. Common generated folders such as .git, __pycache__, and virtualenvs
are skipped by default so the report stays useful.

Usage:
    python export_project_inventory.py
    python export_project_inventory.py --output inventory.md
    python export_project_inventory.py --include-hidden --max-bytes 500000
"""

from __future__ import annotations

import argparse
import fnmatch
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXCLUDE_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}

DEFAULT_EXCLUDE_PATTERNS = {
    "*.egg-info",
    "*.pyc",
    "*.pyo",
    "*.swp",
    ".DS_Store",
}

TEXT_EXTENSIONS = {
    ".cfg",
    ".conf",
    ".css",
    ".csv",
    ".dockerfile",
    ".env",
    ".example",
    ".gitignore",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".log",
    ".md",
    ".py",
    ".rb",
    ".rst",
    ".sh",
    ".sql",
    ".svg",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class InventoryItem:
    path: Path
    is_dir: bool
    size: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a recursive Markdown inventory of project files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root to scan. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("project_inventory.md"),
        help="Markdown file to write. Defaults to project_inventory.md.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help=(
            "Include hidden files and folders except names explicitly excluded. "
            "Hidden files are included by default when they are common project "
            "files such as .env.example."
        ),
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Do not skip common generated folders/files.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional file or folder glob to skip. Can be used multiple times.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=1_000_000,
        help="Maximum bytes to read from one text file. Defaults to 1,000,000.",
    )
    return parser.parse_args()


def add_project_name_to_output(output_path: Path, project_name: str) -> Path:
    if project_name in output_path.stem:
        return output_path

    return output_path.with_name(f"{project_name}_{output_path.name}")


def should_skip(
    path: Path,
    root: Path,
    output_path: Path,
    include_hidden: bool,
    use_default_excludes: bool,
    extra_excludes: list[str],
) -> bool:
    if path == output_path:
        return True

    relative = path.relative_to(root)
    name = path.name
    parts = relative.parts

    if (
        not include_hidden
        and any(part.startswith(".") for part in parts)
        and name not in {".env", ".env.example", ".gitignore"}
    ):
        return True

    patterns = list(extra_excludes)
    if use_default_excludes:
        if name in DEFAULT_EXCLUDE_NAMES:
            return True
        patterns.extend(DEFAULT_EXCLUDE_PATTERNS)

    return any(
        fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(str(relative), pattern)
        for pattern in patterns
    )


def collect_items(
    root: Path,
    output_path: Path,
    include_hidden: bool,
    use_default_excludes: bool,
    extra_excludes: list[str],
) -> list[InventoryItem]:
    items: list[InventoryItem] = []

    def visit(directory: Path) -> None:
        children = sorted(directory.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower()))

        for child in children:
            if should_skip(
                child,
                root,
                output_path,
                include_hidden,
                use_default_excludes,
                extra_excludes,
            ):
                continue

            relative = child.relative_to(root)
            if child.is_dir():
                items.append(InventoryItem(relative, True))
                visit(child)
            else:
                items.append(InventoryItem(relative, False, child.stat().st_size))

    visit(root)
    return items


def tree_lines(items: list[InventoryItem]) -> list[str]:
    lines = ["."]

    for item in items:
        depth = len(item.path.parts) - 1
        indent = "    " * depth
        marker = "/" if item.is_dir else ""
        lines.append(f"{indent}- {item.path.name}{marker}")

    return lines


def is_text_file(path: Path, sample: bytes) -> bool:
    if b"\x00" in sample:
        return False

    suffixes = {suffix.lower() for suffix in path.suffixes}
    if suffixes & TEXT_EXTENSIONS:
        return True

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type and (
        mime_type.startswith("text/")
        or mime_type
        in {
            "application/json",
            "application/xml",
            "application/x-sh",
            "application/yaml",
        }
    ):
        return True

    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False

    return True


def read_text(path: Path, max_bytes: int) -> tuple[bool, str, bool]:
    with path.open("rb") as file:
        sample = file.read(min(max_bytes, 8192))

    if not is_text_file(path, sample):
        return False, "", False

    with path.open("rb") as file:
        data = file.read(max_bytes + 1)

    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]

    text = data.decode("utf-8", errors="replace")
    return True, text, truncated


def code_fence_language(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".css": "css",
        ".csv": "csv",
        ".html": "html",
        ".ini": "ini",
        ".js": "javascript",
        ".json": "json",
        ".jsx": "jsx",
        ".md": "markdown",
        ".pbtxt": "protobuf",
        ".py": "python",
        ".sh": "bash",
        ".svg": "xml",
        ".toml": "toml",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",     
    }
    return mapping.get(suffix, "")


def write_report(root: Path, output_path: Path, items: list[InventoryItem], max_bytes: int) -> None:
    with output_path.open("w", encoding="utf-8") as report:
        report.write(f"# Project Inventory: {root.name}\n\n")
        report.write(f"Root: `{root}`\n\n")
        report.write("## Folder and File Tree\n\n")
        report.write("```text\n")
        report.write("\n".join(tree_lines(items)))
        report.write("\n```\n\n")
        report.write("## File Contents\n\n")

        for item in items:
            if item.is_dir:
                continue

            absolute_path = root / item.path
            report.write(f"### `{item.path}`\n\n")
            report.write(f"- Path: `{absolute_path}`\n")
            report.write(f"- Size: {item.size} bytes\n\n")

            is_text, text, truncated = read_text(absolute_path, max_bytes)
            if not is_text:
                report.write("_Binary or non-text file; content not included._\n\n")
                continue

            if truncated:
                report.write(f"_Content truncated after {max_bytes} bytes._\n\n")

            language = code_fence_language(item.path)
            report.write(f"```{language}\n")
            report.write(text)
            if text and not text.endswith("\n"):
                report.write("\n")
            report.write("```\n\n")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = (root / output_path).resolve()
    else:
        output_path = output_path.resolve()
    output_path = add_project_name_to_output(output_path, root.name)

    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")

    items = collect_items(
        root=root,
        output_path=output_path,
        include_hidden=args.include_hidden,
        use_default_excludes=not args.no_default_excludes,
        extra_excludes=args.exclude,
    )
    write_report(root, output_path, items, args.max_bytes)
    print(f"Wrote project inventory to: {output_path}")
    print(f"Included {sum(item.is_dir for item in items)} folders and {sum(not item.is_dir for item in items)} files.")


if __name__ == "__main__":
    main()
