import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

label_root = "dataset/Labels/"
df = pd.DataFrame(columns=["image_name", "label", "xmax", "xmin", "ymax", "ymin"])

for i, label in enumerate(os.listdir(label_root)):

    if label.endswith(".txt"):

        with open(label_root + label,"r") as f:

            lines = f.readlines()

            for j, line in enumerate(lines):

                if j == 0:

                    label = line.replace("\n","")

                else:

                    b_boxs = [int(loc) for loc in line.replace("\n","").split()]

                    df = df.append({"image_name":f"{i+1}.jpeg", "label":label,
                                    "xmax":b_boxs[0], "xmin":b_boxs[1],
                                    "ymax":b_boxs[2], "ymin":b_boxs[3]}, ignore_index=True)

