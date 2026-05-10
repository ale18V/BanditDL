#!/usr/bin/env python3
"""Regenerate sweep plots from a completed Hydra sweep directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from banditdl.utils.plot_sweep_base import OPTUNA_DB_NAME, plot_sweep_from_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a completed banditdl sweep.")
    parser.add_argument("sweep_dir", type=Path, help="Hydra sweep directory containing .hydra/config.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output directory. Defaults to <sweep_dir>/sweep_artifacts.",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    cfg_path = sweep_dir / ".hydra" / "config.yaml"
    db_path = sweep_dir / OPTUNA_DB_NAME
    if not cfg_path.exists():
        raise SystemExit(f"Missing Hydra config: {cfg_path}")
    if not db_path.exists():
        raise SystemExit(f"Missing Optuna study database: {db_path}")

    cfg = OmegaConf.load(cfg_path)
    plot_sweep_from_cfg(sweep_dir, cfg, output_dir=args.output_dir)
    print(f"[plot_sweep] plots written to: {args.output_dir or sweep_dir / 'sweep_artifacts'}")


if __name__ == "__main__":
    main()
