#!/usr/bin/env python3
"""Collect per-class AP from YOLO-G/val_GRL checkpoints.

Run this from the repository root or from report/ via --repo-root.
It validates each checkpoint with val_GRL.py --verbose, parses the per-class
stdout table, and writes CSV + Markdown tables for the final report.

Example:
  python report/scripts/collect_per_class_ap.py \
    --repo-root . \
    --data domain/city_foggycity.yaml \
    --img 640 \
    --batch-size 16 \
    --device 0
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RUNS = [
    # ID, display method, run directory under runs/train, notes
    ("A1", "Source-only YOLO-G detector", "sanity_pr1_baseline", "source-only baseline"),
    # Despite the run name, opt.yaml for this checkpoint has advgrl: false,
    # da_img: true, and da_img_faithful: true. It is the fixed image-level DA baseline.
    ("A2", "YOLO-G image-level domain adaptation", "city_foggycity_advgrl_faithful", "fixed image-level DA baseline"),
    ("A3", "Adaptive gradient reversal", "city_foggycity_advgrl_faithful_alpha080", "deepest-feature AdvGRL, alpha 0.80"),
    ("A4", "RainMix-coupled triplet adaptation", "city_foggycity_advgrl_faithful_triplet_50ep", "triplet-only / aux extension if present"),
    ("A5", "Adaptive gradient reversal with RainMix-coupled triplet adaptation", "city_foggycity_advgrl_faithful_full_12", "full aux/triplet combination"),
    ("A6", "Adaptive gradient reversal with multi-scale neck adaptation", "city_foggycity_advgrl_faithful_neck_all_alpha075_50ep", "best multi-scale neck run"),
    ("A7", "Adaptive gradient reversal with foreground-gated multi-scale neck adaptation", "city_foggycity_advgrl_faithful_neck_all_alpha075_objgate_50ep", "foreground/objectness-gated neck run"),
]

# Parse lines like:
#                  bus        500         98      0.868      0.388      0.478      0.388
ROW_RE = re.compile(
    r"^\s*(?P<class>\S+)\s+"
    r"(?P<images>\d+)\s+"
    r"(?P<labels>\d+)\s+"
    r"(?P<p>[0-9.]+)\s+"
    r"(?P<r>[0-9.]+)\s+"
    r"(?P<map50>[0-9.]+)\s+"
    r"(?P<map5095>[0-9.]+)\s*$"
)


@dataclass
class RunSpec:
    id: str
    method: str
    run_name: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root containing val_GRL.py and runs/train")
    parser.add_argument("--data", default="domain/city_foggycity.yaml", help="Dataset YAML passed to val_GRL.py")
    parser.add_argument("--img", type=int, default=640, help="Validation image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--device", default="", help="CUDA device string passed to val_GRL.py, e.g. 0; empty lets YOLO choose")
    parser.add_argument("--weights-name", default="best.pt", choices=["best.pt", "last.pt"], help="Checkpoint filename under weights/")
    parser.add_argument("--project", type=Path, default=Path("runs/per_class_val"), help="val_GRL output project path, relative to repo root unless absolute")
    parser.add_argument("--out-dir", type=Path, default=Path("report/tables"), help="Output directory for CSV/Markdown, relative to repo root unless absolute")
    parser.add_argument("--include-all-neck-sweep", action="store_true", help="Also validate all neck alpha/grl sweep checkpoints found under runs/train")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running validation")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_run_specs(root: Path, include_all_neck_sweep: bool) -> list[RunSpec]:
    specs = [RunSpec(*r) for r in DEFAULT_RUNS]
    if include_all_neck_sweep:
        existing = {s.run_name for s in specs}
        train_dir = root / "runs" / "train"
        for weights in sorted(train_dir.glob("city_foggycity_advgrl_faithful_neck_*_50ep/weights/best.pt")):
            run_name = weights.parents[1].name
            if run_name not in existing:
                specs.append(RunSpec("SWEEP", run_name, run_name, "additional neck sweep run"))
                existing.add(run_name)
    return specs


def run_validation(root: Path, spec: RunSpec, args: argparse.Namespace, log_dir: Path) -> tuple[str, str]:
    weights = root / "runs" / "train" / spec.run_name / "weights" / args.weights_name
    if not weights.exists():
        return "missing", f"Missing checkpoint: {weights}"

    project = resolve(root, args.project)
    cmd = [
        sys.executable,
        "val_GRL.py",
        "--weights", str(weights),
        "--data", args.data,
        "--img", str(args.img),
        "--batch-size", str(args.batch_size),
        "--verbose",
        "--project", str(project),
        "--name", spec.run_name,
        "--exist-ok",
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    if args.dry_run:
        return "dry-run", " ".join(cmd)

    log_path = log_dir / f"{spec.run_name}.log"
    print(f"[validate] {spec.id} {spec.run_name}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        return "failed", f"Validation failed with code {proc.returncode}; see {log_path}"
    return "ok", proc.stdout


def parse_metrics(text: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        # Strip common ANSI escape codes from val output.
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = ROW_RE.match(line)
        if not m:
            continue
        d = m.groupdict()
        rows[d["class"]] = d
    return rows


def write_outputs(out_dir: Path, records: list[dict[str, str]], statuses: list[dict[str, str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    long_csv = out_dir / "per_class_ap_long.csv"
    fieldnames = ["id", "method", "run_name", "class", "images", "labels", "precision", "recall", "ap50", "ap5095", "notes"]
    with long_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    status_csv = out_dir / "per_class_ap_status.csv"
    with status_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "method", "run_name", "status", "message"])
        writer.writeheader()
        writer.writerows(statuses)

    classes = ["bus", "bicycle", "car", "motorcycle", "person", "rider", "train", "truck"]
    by_run: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for r in records:
        key = (r["id"], r["method"], r["run_name"])
        by_run.setdefault(key, {})[r["class"]] = r

    md = out_dir / "per_class_ap.md"
    lines = []
    lines.append("# Per-Class AP@0.5 on Foggy Cityscapes")
    lines.append("")
    lines.append("This table is generated by `report/scripts/collect_per_class_ap.py` using `val_GRL.py --verbose` on the listed checkpoints. Values are AP@0.5 unless otherwise stated. Missing checkpoints are recorded in `per_class_ap_status.csv` and are not filled by estimation.")
    lines.append("")
    lines.append("| ID | Method | " + " | ".join(classes) + " | mean AP@0.5 |")
    lines.append("|---|---|" + "---:|" * (len(classes) + 1))
    for key in sorted(by_run.keys()):
        rid, method, run_name = key
        class_rows = by_run[key]
        vals = []
        for cls in classes:
            vals.append(class_rows.get(cls, {}).get("ap50", "n/a"))
        mean = class_rows.get("all", {}).get("ap50", "n/a")
        lines.append(f"| {rid} | {method} | " + " | ".join(vals) + f" | {mean} |")

    lines.append("")
    lines.append("## Per-Class AP@0.5:0.95")
    lines.append("")
    lines.append("| ID | Method | " + " | ".join(classes) + " | mean mAP@0.5:0.95 |")
    lines.append("|---|---|" + "---:|" * (len(classes) + 1))
    for key in sorted(by_run.keys()):
        rid, method, run_name = key
        class_rows = by_run[key]
        vals = []
        for cls in classes:
            vals.append(class_rows.get(cls, {}).get("ap5095", "n/a"))
        mean = class_rows.get("all", {}).get("ap5095", "n/a")
        lines.append(f"| {rid} | {method} | " + " | ".join(vals) + f" | {mean} |")

    lines.append("")
    lines.append("## Validation Status")
    lines.append("")
    lines.append("| ID | Method | Run | Status | Message |")
    lines.append("|---|---|---|---|---|")
    for s in statuses:
        msg = s["message"].replace("|", "\\|")
        lines.append(f"| {s['id']} | {s['method']} | `{s['run_name']}` | {s['status']} | {msg} |")

    md.write_text("\n".join(lines) + "\n")

    print(f"[write] {long_csv}")
    print(f"[write] {status_csv}")
    print(f"[write] {md}")


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    if not (root / "val_GRL.py").exists():
        raise SystemExit(f"Could not find val_GRL.py under repo root: {root}")

    out_dir = resolve(root, args.out_dir)
    log_dir = out_dir / "per_class_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    statuses: list[dict[str, str]] = []

    for spec in build_run_specs(root, args.include_all_neck_sweep):
        status, payload = run_validation(root, spec, args, log_dir)
        statuses.append({
            "id": spec.id,
            "method": spec.method,
            "run_name": spec.run_name,
            "status": status,
            "message": payload.splitlines()[-1] if payload else "",
        })
        if status != "ok":
            print(f"[{status}] {spec.run_name}: {statuses[-1]['message']}")
            continue

        parsed = parse_metrics(payload)
        if "all" not in parsed:
            statuses[-1]["status"] = "parse-failed"
            statuses[-1]["message"] = f"No val_GRL metric rows parsed; see {log_dir / (spec.run_name + '.log')}"
            print(f"[parse-failed] {spec.run_name}")
            continue

        for cls, row in parsed.items():
            records.append({
                "id": spec.id,
                "method": spec.method,
                "run_name": spec.run_name,
                "class": cls,
                "images": row["images"],
                "labels": row["labels"],
                "precision": row["p"],
                "recall": row["r"],
                "ap50": row["map50"],
                "ap5095": row["map5095"],
                "notes": spec.notes,
            })

    write_outputs(out_dir, records, statuses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
