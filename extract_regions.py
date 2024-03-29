import selective_search
import numpy as np


def extract_candidates(img):

    """
    Extract the regions that possibly contain objects in it
    """

    h,w,_ = img.shape
    # Extract regions with selective search method
    boxes = selective_search.selective_search(img, mode='fast')

    regions = selective_search.box_filter(boxes,
                                          min_size=20,
                                          topN=80)

    # Return the product of array elements over a given axis
    img_area = np.prod(img.shape[:2])

    candidates = []

    for x1,y1,x2,y2 in regions:

        if [x1,y1,x2,y2] in candidates: continue
        if x2 > w or x1 > w or y1 > h or y2 > h: continue
        if (abs(x1-x2) * abs(y1-y2)) <= (0.05 * img_area): continue
        if (abs(x1-x2) * abs(y1-y2)) >= (img_area): continue

        candidates.append([x1,y1,x2,y2])

    return candidates


def extract_iou(boxA, boxB, epsilon=1e-5):

    """
    Calculates the intersection over union of two anchor boxes
    """

    x1 = min(boxA[0], boxB[0])
    y1 = min(boxA[2], boxB[2])

    x2 = max(boxA[1], boxB[1])
    y2 = max(boxA[3], boxB[3])

    width = abs(x1 - x2)
    height = abs(y1 - y2)

    area_overlap = width * height

    area_a = abs(boxA[1] - boxA[0]) * abs(boxA[3] - boxA[2])
    area_b = abs(boxB[1] - boxB[0]) * abs(boxB[3] - boxB[2])

    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined + epsilon)

    return iou