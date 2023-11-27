import os
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import numpy as np

class COCOSegmentationDataset(Dataset):
    def __init__(self, root, annotation, transform=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        image = np.array(Image.open(os.path.join(self.root, path)).convert('RGB'))

        mask = Image.new('L', (image.shape[1], image.shape[0]))
        draw = ImageDraw.Draw(mask)
        for ann in annotations:
            if 'segmentation' in ann:
                for polygon in ann['segmentation']:
                    draw.polygon(polygon, outline=1, fill=ann['category_id'])
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask

    def __len__(self):
        return len(self.ids)


