from pathlib import Path

from omegaconf import OmegaConf

from scripts.plot_sweep import load_config


def test_external_plot_config_overrides_profile_plot(tmp_path: Path):
    sweep_dir = tmp_path / "sweep"
    hydra_dir = sweep_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create(
            {
                "plot": {"directions": ["avg"]},
                "optuna": {
                    "plot": {"heatmaps": [{"x": "old.x", "y": "old.y", "exclude_metrics": []}]}
                },
            }
        ),
        hydra_dir / "config.yaml",
    )
    override_path = tmp_path / "plot.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "plot": {
                    "directions": ["final"],
                    "heatmaps": [
                        {
                            "x": "new.x",
                            "y": "new.y",
                            "exclude_metrics": ["train_loss"],
                        }
                    ],
                }
            }
        ),
        override_path,
    )

    cfg = load_config(sweep_dir, override_path)

    assert list(cfg.plot.directions) == ["final"]
    assert cfg.plot.heatmaps[0].x == "new.x"
    assert list(cfg.plot.heatmaps[0].exclude_metrics) == ["train_loss"]
    assert cfg.optuna.plot is None
