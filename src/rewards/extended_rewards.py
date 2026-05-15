"""Extended Reward Functions for Bandit Algorithms."""
import numpy as np

def gaussian_drift_reward(initial_mean=1.0, drift_rate=0.01, std=0.5):
    state = {"step": 0}
    def _reward():
        mean = initial_mean + drift_rate * state["step"]
        state["step"] += 1
        return np.random.normal(mean, std)
    return _reward

def exponential_decay_reward(initial_value=10.0, decay_rate=0.99):
    state = {"step": 0}
    def _reward():
        value = initial_value * (decay_rate ** state["step"])
        state["step"] += 1
        return value
    return _reward

def contextual_reward(feature_weights, noise_std=0.1):
    def _reward(features):
        return float(np.dot(feature_weights, features) + np.random.normal(0, noise_std))
    return _reward

def composite_reward(reward_fns, weights=None):
    if weights is None: weights = [1.0/len(reward_fns)] * len(reward_fns)
    def _reward():
        return sum(w * fn() for w, fn in zip(weights, reward_fns))
    return _reward
