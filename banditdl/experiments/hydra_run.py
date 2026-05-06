from __future__ import annotations

import pathlib

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from banditdl.experiments.config_adapter import build_engine_config, resolve_device
from banditdl.experiments.engine import run_dynamic, run_fixed
from banditdl.utils.plotting import plot_all


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("\n" + OmegaConf.to_yaml(cfg, resolve=True) + "\n")
    run_cfg = build_engine_config(cfg)
    device = resolve_device(cfg)

    output_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    result_dir = output_dir / "results"
    if run_cfg.run_mode == "dynamic":
        run_dynamic(
            params=run_cfg.params,
            result_dir=result_dir,
            seed=int(cfg.seed),
            device=device,
        )
    else:
        run_fixed(
            params=run_cfg.params,
            result_dir=result_dir,
            seed=int(cfg.seed),
            device=device,
        )

    plot_all(
        run_dir=result_dir,
        plots_dir=output_dir / "plots",
        run_label=run_cfg.run_name,
    )


if __name__ == "__main__":
    main()
