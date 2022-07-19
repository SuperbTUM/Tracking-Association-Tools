"""
Data association/tracking in a streaming scenario
Appearance matching only
Author: Ming
Date: Jul 16, 2022
"""
import json
import numpy as np
from collections import defaultdict

try:
    from cuml import DBSCAN
except ImportError:
    from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
import bisect
from pydantic import BaseModel

PRECISION = np.float16


class Bbox(BaseModel):
    leftX: float
    topY: float
    rightX: float
    bottomY: float


def cal_dist(left_matrix: np.ndarray or list, right_matrix=None, weight=0.9):
    if right_matrix is None:
        return pdist(left_matrix, "euclidean") * weight + pdist(left_matrix, "cosine") * (1 - weight)
    return cdist(left_matrix, right_matrix, "euclidean") * weight + cdist(left_matrix, right_matrix, "cosine") * \
           (1 - weight)


def cal_iou(prev_bboxes, cur_bboxes):
    def iou(left_bbox, right_bbox):
        left_bbox_leftX = left_bbox.leftX
        left_bbox_topY = left_bbox.topY
        left_bbox_rightX = left_bbox.rightX
        left_bbox_bottomY = left_bbox.bottomY
        area_left = (left_bbox_rightX - left_bbox_leftX + 1.) * (left_bbox_bottomY - left_bbox_topY + 1.)

        right_bbox_leftX = right_bbox.leftX
        right_bbox_topY = right_bbox.topY
        right_bbox_rightX = right_bbox.rightX
        right_bbox_bottomY = right_bbox.bottomY
        area_right = (right_bbox_rightX - right_bbox_leftX + 1.) * (right_bbox_bottomY - right_bbox_topY + 1.)

        iou_leftX = max(left_bbox_leftX, right_bbox_leftX)
        iou_rightX = min(left_bbox_rightX, right_bbox_rightX)
        iou_topY = max(left_bbox_topY, right_bbox_topY)
        iou_bottomY = min(left_bbox_bottomY, right_bbox_bottomY)
        area_iou = (iou_rightX - iou_leftX + 1.) * (iou_bottomY - iou_topY + 1.)
        return area_iou / (area_left + area_right - area_iou)

    iou_dists = np.ones((len(prev_bboxes), len(cur_bboxes)), PRECISION)
    for i in range(len(prev_bboxes)):
        for j in range(len(cur_bboxes)):
            iou_dists[i, j] = iou(prev_bboxes[i], cur_bboxes[j])
    return iou_dists


def cal_dist_spatio_temporal(left_embeddings,
                             right_embeddings,
                             left_bboxes,
                             right_bboxes,
                             weight=0.8):
    return cal_dist(left_embeddings, right_embeddings) * weight + (1 - cal_iou(left_bboxes, right_bboxes)) * (1 - weight)


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
                       prev_bboxes,
                       bboxes,
                       distance_threshold,
                       behavior_embeddings,
                       behavior_length,
                       momentum,
                       spatio=False):
    max_distance = 0.
    the_embedding = None
    the_bbox = None
    ignored_detection = None
    if spatio:
        dist = cal_dist_spatio_temporal(prev, embeddings, prev_bboxes, bboxes)
    else:
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
        prev_bbox = prev_bboxes[assign]
        cur_bbox = bboxes[i]
        if spatio:
            pair_dist = cal_dist_spatio_temporal([prev_embedding], [cur_embedding], [prev_bbox], [cur_bbox])[0][0]
        else:
            pair_dist = cal_dist([prev_embedding], [cur_embedding])[0][0]
        if pair_dist > distance_threshold:
            # Find the one with maximum distance
            if max_distance < pair_dist:
                max_distance = pair_dist
                the_embedding = cur_embedding
                the_bbox = cur_bbox
                ignored_detection = i
    # Only extend one new ID
    prev = np.append(prev, np.asarray([the_embedding], dtype=PRECISION), axis=0)
    prev_bboxes = np.append(prev_bboxes, np.asarray([the_bbox]), axis=0)
    ignored_embedding = embeddings[ignored_detection]
    embeddings.pop(ignored_detection)
    embeddings.append(ignored_embedding)
    ignored_bbox = bboxes[ignored_detection]
    bboxes.pop(ignored_detection)
    bboxes.append(ignored_bbox)
    return prev, embeddings, prev_bboxes, bboxes


def check_match(prev,
                embeddings,
                prev_bboxes,
                bboxes,
                dist_threshold,
                ignored_embedding=0,
                spatio=False):
    if ignored_embedding < 0:
        prev_ = prev[:ignored_embedding]
        embeddings_ = embeddings[:ignored_embedding]
        prev_bboxes_ = prev_bboxes[:ignored_embedding]
        bboxes_ = bboxes[:ignored_embedding]
    else:
        prev_ = prev.copy()
        embeddings_ = embeddings.copy()
        prev_bboxes_ = prev_bboxes.copy()
        bboxes_ = bboxes.copy()
    if spatio:
        cost = cal_dist_spatio_temporal(prev_, np.asarray(embeddings_, dtype=PRECISION), prev_bboxes_, bboxes_)
    else:
        cost = cal_dist(prev_, np.asarray(embeddings_, dtype=PRECISION))
    row_ind, col_ind = linear_sum_assignment(cost)
    selected_costs = cost[row_ind, col_ind]
    del cost
    for c in selected_costs:
        if c > dist_threshold:
            return False, None, None
    while ignored_embedding < 0:
        row_ind = np.append(row_ind, len(prev) + ignored_embedding)
        col_ind = np.append(col_ind, len(embeddings) + ignored_embedding)
        ignored_embedding += 1
    return True, row_ind, col_ind


def frame_level_matching(frames,
                         bias,
                         confidence_threshold=0.4,
                         distance_threshold=1.,
                         momentum=0.,
                         embedding_dim=256,
                         spatio=False):
    """Distance Comparison with Moving Average"""
    # embeddings from last frame
    prev_embeddings = None
    # bounding boxes from last frame
    prev_bboxes = None
    behavior_embeddings = defaultdict(lambda: np.zeros((embedding_dim,), dtype=PRECISION))
    behavior_length = defaultdict(int)
    labels = defaultdict(dict)
    seen_local_ids = set()
    for frame in frames:
        frame_id = frame.id
        sensor_id = frame.sensorId
        detections = frame.objects
        embeddings = list()
        local_ids = list()
        local_bboxes = list()
        for det in detections:
            confidence = det.confidence
            bbox = det.bbox
            if confidence > confidence_threshold:
                embeddings.append(det.embedding - bias[sensor_id])
                local_ids.append(det.id)
                local_bboxes.append(bbox)
            # else:
            #     labels[frame_id + "@" + sensor_id][det.id] = -1
        # Assign labels on these detections
        if prev_embeddings is None:
            prev_embeddings = np.asarray(embeddings, dtype=PRECISION)
            seen_local_ids.update(local_ids)
            for i, Id in enumerate(local_ids):
                labels[frame_id + "@" + sensor_id][Id] = int(i + 1)
            prev_bboxes = np.asarray(local_bboxes, dtype=object)
        else:
            length_prev = len(prev_embeddings)
            # for i, local_id in enumerate(local_ids):
            #     if local_id not in seen_local_ids:
            #         # Possibly a new identity appears
            #         prev_embeddings = np.append(prev_embeddings, np.asarray([embeddings[i]], dtype=PRECISION), axis=0)
            #         prev_bboxes = np.append(prev_bboxes, np.asarray(local_bboxes[i]))
            while len(prev_embeddings) < len(embeddings):
                prev_embeddings = np.append(prev_embeddings, [np.asarray([3 for _ in range(embedding_dim)], PRECISION)], 0)
                prev_bboxes = np.append(prev_bboxes, np.asarray(Bbox(leftX=0.,
                                                                     rightX=0.,
                                                                     topY=0.,
                                                                     bottomY=0.)))
            if embeddings:
                ignored_embedding = 0
                check_res, row_ind, col_ind = check_match(prev_embeddings,
                                                          embeddings,
                                                          prev_bboxes,
                                                          local_bboxes,
                                                          distance_threshold,
                                                          ignored_embedding,
                                                          spatio)
                while not check_res:
                    # Only add one new ID
                    ignored_embedding -= 1
                    prev_embeddings, embeddings, prev_bboxes, local_bboxes = hungarian_matching(prev_embeddings,
                                                                                                embeddings,
                                                                                                prev_bboxes,
                                                                                                local_bboxes,
                                                                                                distance_threshold,
                                                                                                behavior_embeddings,
                                                                                                behavior_length,
                                                                                                momentum,
                                                                                                spatio)
                    check_res, row_ind, col_ind = check_match(prev_embeddings,
                                                              embeddings,
                                                              prev_bboxes,
                                                              local_bboxes,
                                                              distance_threshold,
                                                              ignored_embedding,
                                                              spatio)
                # Re-associate
                sorted_indices = np.argsort(col_ind)
                assignment = row_ind[sorted_indices]
                # If there is no assignment to the new identity, delete this identity
                deleted_pending = []
                if len(prev_embeddings) > length_prev:
                    for i in range(len(prev_embeddings) - 1, length_prev - 1, -1):
                        if i not in assignment:
                            deleted_pending.append(i)
                # Shift assignment
                for i, assign in enumerate(assignment):
                    assignment[i] -= bisect.bisect_left(deleted_pending, assign)
                # Delete unassigned identity in the possible new identities
                prev_embeddings = np.delete(prev_embeddings, deleted_pending, 0)
                prev_bboxes = np.delete(prev_bboxes, deleted_pending)
                for i, assign in enumerate(assignment):
                    labels[frame_id + "@" + sensor_id][local_ids[i]] = int(assign + 1)
                    behavior_embeddings[assign] += np.asarray(embeddings[i], dtype=PRECISION)
                    behavior_length[assign] += 1
                    prev_embeddings[assign] = embeddings[i]
                    prev_bboxes[assign] = local_bboxes[i]
            seen_local_ids.update(local_ids)
    with open("results.json", "w") as f:
        json.dump(labels, f)
    return labels
