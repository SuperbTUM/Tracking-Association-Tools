# Experimental
from robust_kalman.robust_kalman import RobustKalman
from dataclasses import dataclass
import numpy as np


@dataclass
class TrackingRawData:
    F: np.ndarray
    H: np.ndarray
    x0: np.ndarray
    P0: np.ndarray
    Q0: np.ndarray
    R0: np.ndarray


def tracking_robust(state_transition_matrix,
                    observation_matrix,
                    init_state_vector,
                    init_state_cov_matrix,
                    state_noise_matrix,
                    state_noise_cov_matrix,
                    measurement):
    tracking_raw_data = TrackingRawData(state_transition_matrix,
                                        observation_matrix,
                                        init_state_vector,
                                        init_state_cov_matrix,
                                        state_noise_matrix,
                                        state_noise_cov_matrix)
    tracker = RobustKalman(**tracking_raw_data.__dict__,
                           B=None,
                           use_robust_estimation=True,
                           use_adaptive_statistics=True)
    tracker.time_update()
    tracker.measurement_update(measurement)
    return tracker.current_estimate


def demo():
    ndim = 4
    measurement = np.asarray([[0.01 for _ in range(ndim)]], dtype=float).T
    std_weight_position = 1 / 30
    std_weight_velocity = 1 / 120
    dt = 1 / 30
    state_transition_matrix = np.eye(2 * ndim, 2 * ndim)
    for i in range(ndim):
        state_transition_matrix[i, ndim + i] = dt
    observation_matrix = np.eye(ndim, 2 * ndim)
    init_state_vector = np.asarray([[0. for _ in range(2 * ndim)]], dtype=float).T
    std = [
        2 * std_weight_position * measurement[3, 0],
        2 * std_weight_position * measurement[3, 0],
        1e-2,
        2 * std_weight_position * measurement[3, 0],
        10 * std_weight_velocity * measurement[3, 0],
        10 * std_weight_velocity * measurement[3, 0],
        1e-5,
        10 * std_weight_velocity * measurement[3, 0]]
    init_state_cov_matrix = np.diag(np.square(std))
    G = np.asarray([[0.5 * dt ** 2, dt, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0.5 * dt ** 2, dt, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.5 * dt ** 2, dt, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0.5 * dt ** 2, dt]], dtype=float).T
    sigma_process = 0.1
    sigma_measure = 0.001
    state_noise_matrix = np.matmul(G, G.T) * sigma_process ** 2
    state_noise_cov_matrix = np.eye(ndim, dtype=float) * sigma_measure ** 2
    next_estimate = tracking_robust(state_transition_matrix,
                                    observation_matrix,
                                    init_state_vector,
                                    init_state_cov_matrix,
                                    state_noise_matrix,
                                    state_noise_cov_matrix,
                                    measurement)
    return next_estimate


if __name__ == "__main__":
    # One observation only
    demo()
