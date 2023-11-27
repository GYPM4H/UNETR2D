import numpy as np

import torch
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader

from os.path import join

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

from datetime import datetime
import time

import wandb

from model import UNETR2D
from utils import FocalLoss, TverskyScore, IoU
from dataloader import COCOSegmentationDataset

import segmentation_models_pytorch as smp

wandb.init(
    project=f"UNETR2D_TEST",
    
    config={
        "epochs": 500,
        "train batch size": 4,
        "val batch size": 4,
        "learning rate": 0.0001,
        "weight decay": 0,
        "dataset": "EM",
        "model": "UNETR2D",
    })

if __name__ == "__main__":
    transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])
    root = '/gpfs/space/home/tsiporen/EM/images'

    train_ds = COCOSegmentationDataset(root,
                                        "/gpfs/space/home/tsiporen/OneFormer/datasets_prepaired/EM/original_ann/EM_train.json",
                                        transform)
    test_ds = COCOSegmentationDataset(root,
                                       "/gpfs/space/home/tsiporen/OneFormer/datasets_prepaired/EM/original_ann/EM_test.json",
                                       transform)

    train_dl = DataLoader(dataset=train_ds,
                          batch_size= 4,
                          num_workers=4,
                          pin_memory=True)
    
    test_dl = DataLoader(dataset=test_ds,
                          batch_size= 4,
                          num_workers=4,
                          pin_memory=True)

    model = UNETR2D(image_size=512,
                    patch_size=16,
                    in_channels=3,
                    num_classes=1)
    model.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    criterion_dice = smp.losses.DiceLoss(mode='binary', from_logits= True)
    criterion_focal = FocalLoss()

    f1 = TverskyScore()
    iou = IoU()

    START_TIME = time.time()
    best_val_loss = float('inf')

    for epoch in range(1000):
        print(f"EPOCH #{epoch} \n")

        loss_train = []
        loss_val = []

        f1_train = []
        f1_val = []

        iou_train = []
        iou_val = []

        model.train()

        pbar_train = tqdm(train_dl, total=len(train_dl), desc="TRAIN")
        for step, data in enumerate(pbar_train):
            #load data
            image, mask = data
            image, mask = image.to("cuda"), mask.to("cuda")
            optimizer.zero_grad()
            
            out = model(image)

            #calculate loss
            loss = criterion_dice(out, mask) + criterion_focal(out, mask)
            loss.backward()
            optimizer.step()

            #raw logits ==> probability map
            out = torch.sigmoid(out)

            #threshold probability map to obtain predicted mask
            out[out >= 0.5] = 1
            out[out < 0.5] = 0

            #calculate f1 and iou
            f1_score = f1(out, mask.unsqueeze(1))
            iou_score = iou(out, mask.unsqueeze(1))

            loss_train.append(loss.item())
            f1_train.append(f1_score.item())
            iou_train.append(iou_score.item())
            pbar_train.set_postfix({"Loss": loss.item()})

        print(f"TRAIN LOSS: {np.mean(loss_train)}")
        print(f"TRAIN F1: {np.mean(f1_train)}")
        print(f"TRAIN IOU: {np.mean(iou_train)}")

        wandb.log({
            'TRAIN LOSS': np.mean(loss_train),
            'TRAIN F1': np.mean(f1_train),
            'TRAIN IOU': np.mean(iou_train),
        })

        model.eval()
        pbar_val = tqdm(test_dl, total=len(test_dl), desc="VALID")
        with torch.no_grad():
            for step, data in enumerate(pbar_val):
                image, mask = data
                image, mask = image.to("cuda"), mask.to("cuda")

                out = model(image)

                loss = criterion_dice(out, mask) + criterion_focal(out, mask)

                out = torch.sigmoid(out)

                out[out >= 0.5] = 1
                out[out < 0.5] = 0

                f1_score = f1(out, mask.unsqueeze(1))
                iou_score = iou(out,  mask.unsqueeze(1))

                loss_val.append(loss.item())
                f1_val.append(f1_score.item())
                iou_val.append(iou_score.item())
                pbar_val.set_postfix({"Loss": loss.item()})

            print(f"VAL LOSS: {np.mean(loss_val)}")
            print(f"VAL F1: {np.mean(f1_val)}")
            print(f"VAL IOU: {np.mean(iou_val)}")

            wandb.log({
                'VAL LOSS': np.mean(loss_val),
                'VAL F1': np.mean(f1_val),
                'VAL IOU': np.mean(iou_val),
            })

            if np.mean(loss_val) < best_val_loss:
                print(f"VAL LOSS IMPROVED {best_val_loss} --> {np.mean(loss_val)}")
                best_val_loss = np.mean(loss_val)
                torch.save(model.state_dict(), join("/gpfs/space/home/tsiporen/UNETR2D/em_test_results", "best.pth"))
                print("NEW MODEL SAVED")
            print()
    wandb.finish()

    print(f"FINISHED IN {(time.time() - START_TIME) / 3600} HOURS")
    print(f"CURRENT TIME {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")