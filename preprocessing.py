from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch
import torchvision


class RCNNDataset(Dataset):

    def __init__(self, fpaths, rois, labels, deltas, gtbbs, device):

        """
        fpaths: file paths
        rois: region proposal locations
        labels: classes of the object
        deltas: offset of ground truth to proposed bounding box
        gtbbs: ground truth bounding boxes
        """

        self.fpaths = fpaths
        self.rois = rois
        self.labels = labels
        self.deltas = deltas
        self.gtbbs = gtbbs
        self.device = device

    def __len__(self):

        return len(self.fpaths)

    def __getitem__(self, ix):

        fpath = str(self.fpaths[ix])

        image = cv2.imread(fpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h,w,_ = image.shape

        sh = np.array([w,h,w,h])

        gtbbs = self.gtbbs[ix]
        rois = self.rois[ix]
        bbs = (np.array(rois) * sh).astype(np.uint16)

        labels = self.labels[ix]
        deltas = self.deltas[ix]

        crops = [image[y:Y, x:X] for (x,X,y,Y) in bbs]

        return image, crops, bbs, labels, deltas, gtbbs, fpath

    def collate_fn(self, batch):

        inputs, rois, rixs, labels, deltas = [], [], [], [], []

        for ix in range(len(batch)):

            image, crops, image_bbs, image_labels, image_deltas, image_gtbbs, image_fpath = batch[ix]

            crops = [cv2.resize(crop, (224,224)) for crop in crops]
            crops = [self.preprocess_image(crop/255) for crop in crops]

            inputs.extend(crops)
            labels.extend([c for c in image_labels])
            deltas.extend(image_deltas)

        inputs = torch.cat(inputs).to(self.device)
        labels = torch.Tensor(labels).long().to(self.device)
        deltas = torch.Tensor(deltas).float().to(self.device)

        return inputs, labels, deltas

    def preprocess_image(self, image):

        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])

        image = torch.Tensor(image).permute(2,0,1)
        image = normalize(image)

        return image.to(self.device).float()





