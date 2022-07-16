## Introduction

This repository is designed for data association in a streaming scenario with Hungarian matching. Different from Hungarian matching in DeepSort, where the scenario is a closed world, we apply the algorithm in an open world. In the most popular implementation of DeepSort, if the re-identification model is trained upon Market-1501 training set, the total identities would be 751. If there are more than 751 identities in the real scenario, we are bound to make mistakes. 

## Methodology

We set up two thresholds: detection confidence, and distance thresholds. Only the bounding boxes with confidence more than the threshold will be regarded as valid detections. During the matching, if one representation is far away from any representations from the previous frame, this should be assigned with a new ID. After expanding identities, we should re-associate the representations between previous frame and current frame. 

## Input format

Consider a cross-sensor tracking/association scenario, the input should be in the format of `{id: frame index, sensorId: sensor representation, objects: List[detection]}` where detections consists of confidence, bounding box coordinates (if IOU matching applied) and embeddings.