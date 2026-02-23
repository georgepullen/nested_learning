#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_tps(path: Path) -> float:
    payload = json.loads(path.read_text())
    if "tokens_per_second" not in payload:
        raise ValueError(f"{path} missing tokens_per_second")
    return float(payload["tokens_per_second"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate tensorized throughput gates from benchmark JSON files."
    )
    parser.add_argument("--list-json", required=True, help="per_sample_list benchmark JSON")
    parser.add_argument("--tensorized-json", required=True, help="tensorized_cms benchmark JSON")
    parser.add_argument("--b1-json", required=True, help="shared B=1 benchmark JSON")
    parser.add_argument("--min-list-ratio", type=float, default=1.5)
    parser.add_argument("--min-b1-ratio", type=float, default=1.8)
    parser.add_argument("--label", default="throughput")
    args = parser.parse_args()

    list_tps = _load_tps(Path(args.list_json))
    tensor_tps = _load_tps(Path(args.tensorized_json))
    b1_tps = _load_tps(Path(args.b1_json))

    ratio_list = tensor_tps / max(list_tps, 1e-12)
    ratio_b1 = tensor_tps / max(b1_tps, 1e-12)

    print(
        f"[{args.label}] list_tps={list_tps:.6f} tensorized_tps={tensor_tps:.6f} "
        f"b1_tps={b1_tps:.6f}"
    )
    print(
        f"[{args.label}] ratio_tensorized_vs_list={ratio_list:.6f} "
        f"(min={args.min_list_ratio:.6f})"
    )
    print(
        f"[{args.label}] ratio_tensorized_vs_b1={ratio_b1:.6f} "
        f"(min={args.min_b1_ratio:.6f})"
    )

    failures: list[str] = []
    if ratio_list < float(args.min_list_ratio):
        failures.append(
            "tensorized/list ratio below threshold "
            f"({ratio_list:.6f} < {float(args.min_list_ratio):.6f})"
        )
    if ratio_b1 < float(args.min_b1_ratio):
        failures.append(
            "tensorized/B1 ratio below threshold "
            f"({ratio_b1:.6f} < {float(args.min_b1_ratio):.6f})"
        )
    if failures:
        raise SystemExit("[throughput-gate] " + "; ".join(failures))

    print("[throughput-gate] PASS")


if __name__ == "__main__":
    main()
