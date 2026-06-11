from __future__ import annotations

import pathlib

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from banditdl.experiments.config_adapter import build_engine_config, resolve_device
from banditdl.experiments.engine import run_experiment
from banditdl.utils.plotting import plot_all


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_cfg = build_engine_config(cfg)
    config = run_cfg.config
    print("\n" + OmegaConf.to_yaml(cfg, resolve=True) + "\n")

    device = resolve_device(cfg)

    output_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    result_dir = output_dir / "results"
    run_experiment(config, result_dir, config.seed, device)

    plot_all(
        run_dir=result_dir,
        plots_dir=output_dir / "plots",
        run_label=run_cfg.run_name,
    )


if __name__ == "__main__":
    main()
