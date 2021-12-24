import os
import numpy as np
import random
import argparse
from importlib import import_module
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder, VisionDataset

from utils.utils import seed_everything
from utils.Dataset import CustomAugmentation, CustomDataset, custom_augment_train
from utils.seg_train import train, validation


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transform
    val_transform = CustomAugmentation('val')
    # dataset
    val_dataset = CustomDataset(data_path=args.dataset_path, mode='test', app_mode='semantic', transform=val_transform)
    # DataLoader
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            drop_last=True,
                                            pin_memory=(torch.cuda.is_available()),
                                            collate_fn=None)

    # model 정의
    model_module = getattr(import_module("utils.models"), args.model)
    model = model_module()
    # Loss function 정의
    criterion_module = getattr(import_module("utils.loss"), args.criterion)
    criterion = criterion_module()

    # Create trainer
    avrg_loss = validation(1, model, val_loader, criterion, device, 6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--dataset_path', type=str, default='../data/QCdataset', help='dataset_path')
    parser.add_argument('--saved_dir', type=str, default='./saved', help='saved_dir')
    parser.add_argument('--save_file_name', type=str, default='semantic_seg1.pt', help='save_file_name')
    parser.add_argument('--num_epochs', type=int, default=40, help='num_epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--model', type=str, default='UnetResnet18', help='model')
    parser.add_argument('--criterion', type=str, default='DiceWCELoss', help='criterion')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')
    parser.add_argument('--val_every', type=int, default=1, help='val_every (default: 1)')
    args = parser.parse_args()

    # hyper parameter
    
    if not os.path.isdir('./saved'):
        os.mkdir('./saved')

    main(args)