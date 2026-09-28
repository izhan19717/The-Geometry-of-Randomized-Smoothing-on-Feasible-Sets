"""Write SHA-256 hashes for the distributable repository files."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", "reproduced", "reproduced_figures"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path.name not in {"MANIFEST.sha256", ".DS_Store"}
        and not any(part in SKIP_DIRS or part.startswith(".venv-") for part in relative.parts)
        and not any(part.endswith(".egg-info") for part in relative.parts)
        and path.suffix not in {".pyc", ".pyo"}
    )


def main() -> None:
    lines = []
    for path in sorted(ROOT.rglob("*")):
        if included(path):
            lines.append(f"{sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}\n")
    (ROOT / "MANIFEST.sha256").write_text("".join(lines), encoding="utf-8")
    print(f"Hashed {len(lines)} files")


if __name__ == "__main__":
    main()
