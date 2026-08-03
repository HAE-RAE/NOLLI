"""Shared helpers for writing the release JSONL layout under data/.

Generators write only:
  data/{task}_{easy|medium|hard}.jsonl

No staging dirs (data/csv, data/jsonl), no combined dumps, no filename aliases.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

TIERS = ("easy", "medium", "hard")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def normalize_difficulty(value: Any) -> str:
    s = getattr(value, "value", value)
    return str(s).strip().lower()


def write_release_jsonl(
    task: str,
    records: Sequence[Mapping[str, Any]],
    *,
    data_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Split records by ``difficulty`` and write ``data/{task}_{tier}.jsonl``.

    Empty tiers are skipped. Returns ``{tier: path}`` for files written.
    """
    out = Path(data_dir) if data_dir is not None else DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    by_tier: Dict[str, List[Mapping[str, Any]]] = {t: [] for t in TIERS}
    for rec in records:
        tier = normalize_difficulty(rec.get("difficulty", ""))
        if tier not in by_tier:
            raise ValueError(f"{task}: unexpected difficulty {tier!r} in record id={rec.get('id')!r}")
        by_tier[tier].append(rec)

    written: Dict[str, Path] = {}
    for tier in TIERS:
        items = by_tier[tier]
        if not items:
            continue
        path = out / f"{task}_{tier}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
        print(f"JSONL: {path} ({len(items)} rows)")
        written[tier] = path
    return written
