import os
import pandas as pd
import warnings

import config
from preprocessing import *
from extract_regions import *
from models import *
warnings.filterwarnings("ignore")


df = pd.DataFrame(columns=["image_name", "label", "xmax", "xmin", "ymax", "ymin"])

for img_name in os.listdir(config.image_root):

    if img_name.endswith(".jpeg"):

        img_path = config.image_root + img_name
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(config.label_root + img_name.split(".")[0] + ".txt","r") as f:

            lines = f.readlines()
            n_boxes = int(lines[0].replace("\n","")) + 1

            for n in range(1,n_boxes):

                b_boxs = [int(loc) for loc in lines[n].replace("\n","").split()]

                df = df.append({"image_name":img_name, "label":"Gun",
                                "xmax":b_boxs[0] / img.shape[1], "xmin":b_boxs[1] / img.shape[1],
                                "ymax":b_boxs[2] / img.shape[0], "ymin":b_boxs[3] / img.shape[0]}, ignore_index=True)


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

        # Find the index of a candidate(best_iou_at) that has the highest IOU
        # And the ground truth(best_bb)
        best_iou_at = np.argmax(ious)
        best_iou = ious[best_iou_at]
        best_bb = x,y,X,Y = bbs[0]

        if best_iou > 0.3: clss.append("Gun")
        else: clss.append('background')

        delta = np.array([x-cx, y-cy, X-cX, Y-cY]) / np.array([w,h,w,h])
        deltas.append(delta)
        rois.append(candidate / np.array([w,h,w,h]))

    FPATHS.append(fpath)
    IOUS.append(ious)
    ROIS.append(rois)
    CLSS.append(clss)
    DELTAS.append(deltas)
    GTBBS.append(bbs)

FPATHS = [f'{str(f)}' for f in FPATHS]

targets = pd.DataFrame(np.array(CLSS).reshape(-1,1), columns=['label'])
label2target = {i:t for t,i in enumerate(["Gun","background"])}
target2label = {t:i for t,i in label2target.items()}
background_class = label2target['background']


"""
Split dataset into train and test 
"""
split_size = 9 * len(FPATHS) // 10

train_ds = RCNNDataset(FPATHS[:split_size], ROIS[:split_size], CLSS[:split_size], DELTAS[:split_size], GTBBS[:split_size], label2target)
test_ds = RCNNDataset(FPATHS[split_size:], ROIS[split_size:], CLSS[split_size:], DELTAS[split_size:], GTBBS[split_size:], label2target)

train_dl = DataLoader(train_ds, batch_size=32, collate_fn=train_ds.collate_fn, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=32, collate_fn=test_ds.collate_fn, drop_last=True)

"""
Load the pretrained and then RCNN model
"""
vgg_base = torchvision.models.vgg16(pretrained=True)
vgg_base.classifier = nn.Sequential()
for param in vgg_base.parameters():
    param.requires_grad = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RCNN(vgg_base, label2target).to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = model.calc_loss

N_EPOCHS = 5

for epoch in range(N_EPOCHS):

    losses = []
    accuracies = []
    for data in train_dl:

        loss, ce_loss, l1_loss, accuracy = train_batch(data, model, optimizer, criterion)
        losses.append(loss)
        accuracies.append(accuracy)

    val_losses = []
    val_accuracies = []
    for data in test_dl:

        val_loss, val_ce_loss, val_l1_loss, val_accuracy = validate_batch(data, model, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    print(f"Epoch: {epoch+1}/{N_EPOCHS}, Loss: {sum(losses) / len(train_dl)}, Val_Loss: {sum(val_losses) / len(test_dl)}, "
          f"Accuracy: {sum(accuracies) / len(train_dl)}, Val_Accuracy: {sum(val_accuracies) / len(test_dl)}")
