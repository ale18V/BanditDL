from banditdl.core.worker.dynamic import DynamicWorker


class RecordingSampler:
    def __init__(self):
        self.observation = None

    def update(self, neighbors, rewards):
        self.observation = (neighbors, rewards)


def test_observe_neighbors_forwards_precomputed_rewards():
    worker = DynamicWorker.__new__(DynamicWorker)
    worker.neighbor_sampler = RecordingSampler()

    worker.observe_neighbors([2, 4], [0.25, 0.75])

    assert worker.neighbor_sampler.observation == ([2, 4], [0.25, 0.75])
