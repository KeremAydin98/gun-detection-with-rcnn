import selective_search
import numpy as np


def extract_candidates(img):

    """
    Extract the regions that possibly contain objects in it
    """

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
        if (np.abs(x1-x2) * np.abs(y1-y2)) < (0.05 * img_area): continue
        if (np.abs(x1-x2) * np.abs(y1-y2)) > (img_area): continue

        candidates.append([x1,y1,x2,y2])

    return candidates


def extract_iou(boxA, boxB, epsilon=1e-5):

    """
    Calculates the intersection over union of two anchor boxes
    """

    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])

    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    width = x2 - x1
    height = y2 - y1

    if (width < 0) or (height < 0):

        return 0

    area_overlap = width * height

    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined + epsilon)

    return iou