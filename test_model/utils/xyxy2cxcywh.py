import numpy as np
def xyxy2cxcywh(bboxes):
    bbox = bboxes.copy()
    bbox[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bbox[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bbox[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bbox[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bbox