"""Generate a Condor DAG that trains every sweep cell, then runs the
analysis suite on each finished checkpoint.

Reads experiments/configs/_generated/manifest.json (created by
generate_sweeps.py) and writes a single DAG file.

Usage:
    python experiments/condor/build_dag.py \
        --manifest experiments/configs/_generated/manifest.json \
        --out      experiments/condor/sweeps.dag
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ANALYSES = [
    # (script_name, common-extra-args)
    ("forecast_curves",
     "--csv $DATA_ROOT/FileList.csv --video-root $DATA_ROOT/EchoNet-Dynamic/Videos"),
    ("phase_decoding",
     "--filelist-csv $DATA_ROOT/FileList.csv --vt-csv $DATA_ROOT/VolumeTracings.csv "
     "--video-root $DATA_ROOT/EchoNet-Dynamic/Videos"),
    ("heart_rate",
     "--filelist-csv $DATA_ROOT/FileList.csv --video-root $DATA_ROOT/EchoNet-Dynamic/Videos"),
    ("geometry",
     "--csv $DATA_ROOT/FileList.csv --video-root $DATA_ROOT/EchoNet-Dynamic/Videos"),
    ("ef_probe",
     "--csv $DATA_ROOT/FileList.csv --video-root $DATA_ROOT/EchoNet-Dynamic/Videos --target EF"),
    ("cycle_consistency",
     "--csv $DATA_ROOT/FileList.csv --video-root $DATA_ROOT/EchoNet-Dynamic/Videos"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data-root",
                   default="/scratch/bxu/project/echonet",
                   help="Root containing FileList.csv, VolumeTracings.csv, EchoNet-Dynamic/Videos.")
    args = p.parse_args()

    with open(args.manifest) as f:
        cells = json.load(f)

    lines: list[str] = []
    lines.append(f"# Auto-generated DAG. Cells: {len(cells)}, analyses per cell: {len(ANALYSES)}")
    lines.append("")
    lines.append("CONFIG_FILE_DAGMAN_USE_DIRECT_SUBMIT = True")
    lines.append("")
    for cell in cells:
        train_node = f"train_{cell['sweep']}_{cell['cell']}"
        lines.append(f'JOB {train_node} experiments/condor/sonostate_train.sub')
        lines.append(f'VARS {train_node} CONFIG="{cell["config"]}"')
        for script, extra in ANALYSES:
            ck = f"{cell['folder']}/latest.pt"
            ana_out = f"{cell['folder']}/analysis_{script}"
            ana_node = f"ana_{script}_{cell['sweep']}_{cell['cell']}"
            lines.append(f'JOB {ana_node} experiments/condor/sonostate_analysis.sub')
            full_extra = extra.replace("$DATA_ROOT", args.data_root)
            lines.append(
                f'VARS {ana_node} '
                f'SCRIPT="{script}" '
                f'CHECKPOINT="{ck}" '
                f'OUT="{ana_out}" '
                f'EXTRA="{full_extra}"'
            )
            lines.append(f'PARENT {train_node} CHILD {ana_node}')
        lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print(f"wrote DAG with {len(cells)} train + {len(cells) * len(ANALYSES)} analysis nodes -> {args.out}")


if __name__ == "__main__":
    main()
