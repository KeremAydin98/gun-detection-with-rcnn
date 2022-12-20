import os
import pandas as pd
import warnings
from preprocessing import *
from extract_regions import *
warnings.filterwarnings("ignore")


df = pd.DataFrame(columns=["image_name", "label", "xmax", "xmin", "ymax", "ymin"])

for i, label in enumerate(os.listdir(config.label_root)):

    if label.endswith(".txt"):

        with open(config.label_root + label,"r") as f:

            lines = f.readlines()

            for j, line in enumerate(lines):

                if j == 0:

                    label = line.replace("\n","")

                else:

                    b_boxs = [int(loc) for loc in line.replace("\n","").split()]

                    df = df.append({"image_name":f"{i+1}.jpeg", "label":label,
                                    "xmax":b_boxs[0], "xmin":b_boxs[1],
                                    "ymax":b_boxs[2], "ymin":b_boxs[3]}, ignore_index=True)


FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [], [], [], [], [], []

dataset = OpenImages(df=df)

N = 500
for ix, (img, bbs, labels, fpath) in enumerate(dataset):

    if (ix == N):
        break

    # Extract candidates from each image
    h, w, _ = img.shape
    candidates = extract_candidates(img)
    candidates = np.array([(x, y, x+w, y+h) for x,y,w,h in candidates])

    ious, rois, clss, deltas = [], [], [], []

    # Store the IOU of all candidates with respect to all ground truths
    ious = np.array([extract_iou(candidate, bb) for candidate in candidates for bb in bbs]).T

    # Loop through each candidate and store the xmin, ymin, xmax, ymax
    for jx, candidate in enumerate(candidates):

        cx, cy, cX, cY = candidate

        candidate_ious = ious[jx]

        # Find the index of a candidate(best_iou_at) that has the highest IOU
        # And the ground truth(best_bb)
        best_iou_at = np.argmax(candidate_ious)
        best_iou = candidate_ious[best_iou_at]
        best_bb = x,y,X,Y = bbs[best_iou_at]

        if best_iou > 0.3: clss.append(labels[best_iou_at])
        else: clss.append('background')

        delta = np.array([x-cx])
