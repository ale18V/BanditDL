#!/usr/bin/env python3
"""Regenerate sweep plots from a completed Hydra sweep directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from banditdl.utils.plot_sweep_base import (
    OPTUNA_DB_NAME,
    plot_config_from_cfg,
    plot_sweep_from_cfg,
)


def load_config(sweep_dir: Path, override_path: Path | None):
    cfg_path = sweep_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Missing Hydra config: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    if override_path is None:
        return cfg
    if not override_path.exists():
        raise SystemExit(f"Missing plotting config: {override_path}")

    override = OmegaConf.load(override_path)
    if "plot" not in override:
        raise SystemExit(f"Plotting config must contain a top-level 'plot' key: {override_path}")

    cfg.plot = OmegaConf.merge(plot_config_from_cfg(cfg), override.plot)
    if "optuna" in cfg:
        cfg.optuna.plot = None
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a completed banditdl sweep.")
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Hydra sweep directory containing .hydra/config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output directory. Defaults to <sweep_dir>/sweep_artifacts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Plotting YAML merged over the sweep's stored Hydra configuration.",
    )
    parser.add_argument(
        "--single-runs",
        action="store_true",
        help="Also generate standard plots for every completed trial.",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    db_path = sweep_dir / OPTUNA_DB_NAME
    if not db_path.exists():
        raise SystemExit(f"Missing Optuna study database: {db_path}")

    cfg = load_config(sweep_dir, args.config)
    if args.single_runs:
        OmegaConf.update(cfg, "plot.single_runs.enabled", True, force_add=True)
    plot_sweep_from_cfg(sweep_dir, cfg, output_dir=args.output_dir)
    print(f"[plot_sweep] plots written to: {args.output_dir or sweep_dir / 'sweep_artifacts'}")


if __name__ == "__main__":
    main()
