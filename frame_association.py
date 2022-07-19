"""
Data association/tracking in a streaming scenario
Appearance matching only
Author: Ming
Date: Jul 16, 2022
"""
import numpy as np
from sklearn import preprocessing
from collections import defaultdict

try:
    from cuml import DBSCAN
except ImportError:
    from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist

PRECISION = np.float16


def cal_dist(left_matrix: np.ndarray or list, right_matrix=None, weight=0.9):
    if right_matrix is None:
        return preprocessing.normalize(pdist(left_matrix, "euclidean")) * weight + \
               pdist(left_matrix, "cosine") * (1 - weight)
    return preprocessing.normalize(cdist(left_matrix, right_matrix, "euclidean")) * weight + \
           cdist(left_matrix, right_matrix, "cosine") * (1 - weight)


def get_camera_bias(frames):
    """Get camera bias for diminishing bias across sensors"""
    embeddings_under_sensor = defaultdict(list)
    for frame in frames:
        sensor_id = frame.sensorId
        for objct in frame.objects:
            embeddings_under_sensor[sensor_id].append(objct.embedding)
    del frames
    bias = dict()
    for sensor in embeddings_under_sensor:
        embeddings = np.asarray(embeddings_under_sensor[sensor], dtype=PRECISION)
        dist = cal_dist(embeddings)
        # Exclude noise
        clustering_labels = DBSCAN(eps=0.5, min_samples=10, metric="precomputed").fit(dist).labels_
        valid_index = []
        for i, label in enumerate(clustering_labels):
            if label != -1:
                valid_index.append(i)
        valid_index = np.asarray(valid_index)
        valid_embeddings = embeddings[valid_index]
        # Get mean of embeddings
        mean_embeddings = valid_embeddings.mean(axis=0)
        bias[sensor] = mean_embeddings
    return bias


def hungarian_matching(prev,
                       embeddings,
                       distance_threshold,
                       behavior_embeddings,
                       behavior_length,
                       momentum):
    max_distance = 0.
    the_embedding = None
    dist = cal_dist(prev, np.asarray(embeddings))
    row_ind, col_ind = linear_sum_assignment(dist)
    del dist
    sorted_indices = np.argsort(col_ind)
    assignment = row_ind[sorted_indices]
    # Previous embeddings pending for update
    for i, assign in enumerate(assignment):
        # Notation i means detection of the current frame;
        # Notation assign means corresponding detection of the previous frame
        if behavior_length[assign]:
            prev_embedding = (1 - momentum) * prev[assign] + \
                             momentum * behavior_embeddings[assign] / behavior_length[assign]
        else:
            prev_embedding = prev[assign]
        cur_embedding = embeddings[i]
        pair_dist = cal_dist([prev_embedding], [cur_embedding])[0][0]
        if pair_dist > distance_threshold:
            # Find the one with maximum distance
            if max_distance < pair_dist:
                max_distance = pair_dist
                the_embedding = cur_embedding
    # Only extend one new ID
    prev = np.append(prev, np.asarray([the_embedding], dtype=PRECISION), axis=0)
    return prev


def check_match(prev,
                embeddings,
                dist_threshold):
    cost = cal_dist(prev, np.asarray(embeddings, dtype=PRECISION))
    row_ind, col_ind = linear_sum_assignment(cost)
    selected_costs = cost[row_ind, col_ind]
    del cost
    for c in selected_costs:
        if c > dist_threshold:
            return False, None, None
    return True, row_ind, col_ind


def frame_level_matching(frames,
                         bias,
                         confidence_threshold=0.4,
                         distance_threshold=1.,
                         momentum=0.,
                         embedding_dim=256):
    """Distance Comparison with Moving Average"""
    # embeddings from last frame
    prev = None
    behavior_embeddings = defaultdict(lambda: np.zeros((embedding_dim,), dtype=PRECISION))
    behavior_length = defaultdict(int)
    labels = defaultdict(dict)
    for frame in frames:
        frame_id = frame.id
        sensor_id = frame.sensorId
        detections = frame.objects
        embeddings = list()
        local_ids = list()
        for det in detections:
            confidence = det.confidence
            if confidence > confidence_threshold:
                embeddings.append(det.embedding - bias[sensor_id])
                local_ids.append(det.id)
            else:
                labels[frame_id + "@" + sensor_id][det.id] = -1
        # Assign labels on these detections
        if prev is None:
            prev = np.asarray(embeddings, dtype=PRECISION)
            for i, Id in enumerate(local_ids):
                labels[frame_id + "@" + sensor_id][Id] = i
        else:
            while len(prev) < len(embeddings):
                prev = np.append(prev, np.asarray([[3. for _ in range(embedding_dim)]], dtype=PRECISION), axis=0)
            if embeddings:
                check_res, row_ind, col_ind = check_match(prev,
                                                          embeddings,
                                                          distance_threshold)
                while not check_res:
                    # Only add one new ID
                    prev_num_identities = len(prev)
                    prev = hungarian_matching(prev,
                                              embeddings,
                                              distance_threshold,
                                              behavior_embeddings,
                                              behavior_length,
                                              momentum)
                    assert len(prev) == prev_num_identities + 1
                    check_res, row_ind, col_ind = check_match(prev,
                                                              embeddings,
                                                              distance_threshold)
                # Re-associate
                sorted_indices = np.argsort(col_ind)
                assignment = row_ind[sorted_indices]
                for i, assign in enumerate(assignment):
                    labels[frame_id + "@" + sensor_id][local_ids[i]] = assign
                    behavior_embeddings[assign] += np.asarray(embeddings[i], dtype=PRECISION)
                    behavior_length[assign] += 1
                    prev[assign] = embeddings[i]
    return labels
