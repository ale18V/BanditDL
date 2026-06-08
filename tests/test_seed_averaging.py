import numpy as np
import pytest
import json

from banditdl.utils.seed_averaging import aggregate_seed_results, seed_result_dir


def _write_seed_result(result_dir, seed, accuracy_array, selected_neighbors):
    seed_dir = seed_result_dir(result_dir, seed)
    seed_dir.mkdir(parents=True)
    np.save(seed_dir / "evaluation_steps.npy", np.array([0, 2]))
    np.save(seed_dir / "validation_accuracy.npy", accuracy_array)
    np.save(seed_dir / "selected_neighbors.npy", selected_neighbors)
    (seed_dir / "sampler_states.jsonl").write_text(
        json.dumps({"step": 0, "workers": [], "seed_local": seed}) + "\n"
    )


def test_aggregate_seed_results_writes_public_mean_and_by_seed_arrays(tmp_path):
    result_dir = tmp_path / "results"
    _write_seed_result(
        result_dir,
        10,
        np.array([[0.1, 0.3], [0.5, 0.7]]),
        np.array([[[1], [0]]]),
    )
    _write_seed_result(
        result_dir,
        11,
        np.array([[0.3, 0.5], [0.7, 0.9]]),
        np.array([[[0], [1]]]),
    )

    aggregate_seed_results(result_dir, [10, 11])

    np.testing.assert_allclose(np.load(result_dir / "evaluation_steps.npy"), np.array([0, 2]))
    np.testing.assert_allclose(
        np.load(result_dir / "validation_accuracy.npy"),
        np.array([[0.2, 0.4], [0.6, 0.8]]),
    )
    assert np.load(result_dir / "validation_accuracy_by_seed.npy").shape == (2, 2, 2)
    assert np.load(result_dir / "selected_neighbors_by_seed.npy").shape == (2, 1, 2, 1)
    assert not (result_dir / "selected_neighbors.npy").exists()
    rows = [
        json.loads(line)
        for line in (result_dir / "sampler_states_by_seed.jsonl").read_text().splitlines()
    ]
    assert [row["seed"] for row in rows] == [10, 11]


def test_aggregate_seed_results_copies_single_seed_sampler_states(tmp_path):
    result_dir = tmp_path / "results"
    _write_seed_result(
        result_dir,
        10,
        np.array([[0.1, 0.3], [0.5, 0.7]]),
        np.array([[[1], [0]]]),
    )

    aggregate_seed_results(result_dir, [10])

    assert (result_dir / "sampler_states.jsonl").read_text().strip()


def test_aggregate_seed_results_rejects_partial_metric_files(tmp_path):
    result_dir = tmp_path / "results"
    _write_seed_result(
        result_dir,
        10,
        np.array([[0.1, 0.3], [0.5, 0.7]]),
        np.array([[[1], [0]]]),
    )
    seed_result_dir(result_dir, 11).mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        aggregate_seed_results(result_dir, [10, 11])
