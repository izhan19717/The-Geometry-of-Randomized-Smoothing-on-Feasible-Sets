"""Create local aliases for paths embedded in recorded study protocols."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    ROOT / "_ICLR_2027__Feasibility_Breaks_Smoothing" / "research": ROOT / "research",
    ROOT / "output" / "iclr2027_safety_gym_safe_controllers_v2": ROOT / "controllers",
}


def main() -> None:
    for alias, target in ALIASES.items():
        if not target.is_dir():
            raise FileNotFoundError(f"Recorded study directory is missing: {target}")
        if alias.is_symlink():
            if alias.resolve(strict=True) != target.resolve(strict=True):
                raise FileExistsError(f"Alias points elsewhere: {alias}")
            continue
        if alias.exists():
            raise FileExistsError(f"Refusing to replace existing path: {alias}")
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=True)
        print(f"Linked {alias.relative_to(ROOT)} to {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
