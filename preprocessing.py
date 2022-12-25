from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch
import torchvision
import config


class RCNNDataset(Dataset):

    def __init__(self, fpaths, rois, labels, deltas, gtbbs, label2target, device=torch.device('cpu')):

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
        self.label2target = label2target
        self.device = device

    def __len__(self):

        return len(self.fpaths)

    def __getitem__(self, ix):

        """
        Extracting information of a certain image from the dataset
        :param ix:
        :return: image, cropped image, bounding box corners, labels, offsets,
        ground truth bounding box corners, file path
        """

        # Extracting the image path
        fpath = str(self.fpaths[ix])

        # Reading the image from path
        image = cv2.imread(fpath)
        # Converting its color scale from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h,w,_ = image.shape

        sh = np.array([w,h,w,h])

        # Ground truth bounding box
        gtbbs = self.gtbbs[ix]
        # Region of interest
        rois = self.rois[ix]
        # Bounding box with correct scale for the image so that it can work with any image size
        bbs = (np.array(rois) * sh).astype(np.uint16)

        # Label of the certain image
        labels = self.labels[ix]
        # Offsets of its bounding box corners
        deltas = self.deltas[ix]

        # Cropping the region proposal by the region of interest locations
        crops = [image[y:Y, x:X] for (x,X,y,Y) in bbs]

        return image, crops, bbs, labels, deltas, gtbbs, fpath

    def collate_fn(self, batch):

        """
        inputs: cropped images
        labels: labels
        deltas: offsets of bounding box corner locations
        """
        inputs, labels, deltas = [], [], []

        for ix in range(len(batch)):

            image, crops, image_bbs, image_labels, image_deltas, image_gtbbs, image_fpath = batch[ix]

            # resizing the cropped region to the pretrained model input image size
            crops = [cv2.resize(crop, (224,224)) for crop in crops]
            crops = [self.preprocess_image(crop/255) for crop in crops]

            inputs.extend(crops)
            labels.extend([self.label2target(c) for c in image_labels])
            deltas.extend(image_deltas)

        # Concatenates the given sequence of seq tensors in the given dimension
        inputs = torch.cat(inputs).to(self.device)
        # Casting labels to long data type
        labels = torch.Tensor(labels).long().to(self.device)
        # Casting offsets to float data type
        deltas = torch.Tensor(deltas).float().to(self.device)

        return inputs, labels, deltas

    def preprocess_image(self, image):

        """
        Preprocess image to prepare for extracting its feature maps with a pretrained CNN model
        """

        # Normalization process
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])

        # Transforming image shape to (Channels, Width, Height)
        image = torch.Tensor(image).permute(2,0,1)
        # Normalizing image
        image = normalize(image)

        return image.to(self.device).float()


class OpenImages(Dataset):

    def  __init__(self, df, image_folder=config.image_root):

        self.root = image_folder
        self.df = df
        self.unique_images = df["image_name"].unique()

    def __len__(self):

        return len(self.unique_images)

    def __getitem__(self, ix):

        # Name of the image
        image_id = self.unique_images[ix]

        # Image path
        image_path = f'{self.root}{image_id}'

        # Reading the image from path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h,w,_ = image.shape

        df = self.df.copy()
        df = df[df["image_name"] == image_id]

        boxes = df[["xmin","ymin","xmax","ymax"]].values
        boxes = (boxes * np.array([w,h,w,h])).astype(np.uint16).tolist()

        classes = df["label"].values.tolist()

        return image, boxes, classes, image_path


