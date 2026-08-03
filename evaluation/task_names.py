from __future__ import annotations

import re
from typing import Optional, Set

DIFFICULTY_SUFFIXES = ("_easy", "_medium", "_hard")


def resolve_registry_task_key(task_name: str, registry_keys: Set[str]) -> str:
    if task_name in registry_keys:
        return task_name
    for suf in DIFFICULTY_SUFFIXES:
        if task_name.endswith(suf):
            base = task_name[: -len(suf)]
            if base in registry_keys:
                return base
    return task_name


def locale_from_task_name(task: str) -> Optional[bool]:
    if re.search(r"_ko(?:_|$)", task):
        return True
    if re.search(r"_en(?:_|$)", task):
        return False
    return None
