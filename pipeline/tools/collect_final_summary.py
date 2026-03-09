#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path('/home/congpx/fingerprint')
RUN_DIR = BASE_DIR / 'runs' / 'topic3'
OUT_DIR = RUN_DIR / 'summary'
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary = {}

seg_eval = RUN_DIR / 'seg_shape_eval' / 'metrics_summary.json'
if seg_eval.exists():
    summary['seg_shape_eval'] = json.loads(seg_eval.read_text())

(OUT_DIR / 'metrics_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
