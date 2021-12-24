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
from utils.Trainer import TorchTrainer


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transform
    train_transform = CustomAugmentation('train')
    val_transform = CustomAugmentation('val')

    # dataset
    train_dataset = CustomDataset(data_path=args.dataset_path, mode='train', transform=train_transform)
    val_dataset = CustomDataset(data_path=args.dataset_path, mode='val', transform=val_transform)

    # train_dataset = ImageFolder(root=args.dataset_path+'/images/train', transform=train_transform)
    # val_dataset = ImageFolder(root=args.dataset_path+'/images/val', transform=val_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            drop_last=True,
                                            pin_memory=(torch.cuda.is_available()),
                                            collate_fn=None)

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
    model = model.to(device)
    # Loss function 정의
    criterion_module = getattr(import_module("utils.loss"), args.criterion)
    criterion = criterion_module()
    # Optimizer 정의
    optimizer_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = optimizer_module(params = model.parameters(), lr = args.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, verbose=True)

    # Create trainer
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_path=args.saved_dir,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_loader,
        n_epoch=args.num_epochs,
        val_dataloader=val_loader,
    )

    # evaluate model with test set
    model.load_state_dict(torch.load(args.saved_dir))
    test_loss, test_f1, test_acc = trainer.test(
        model=model, test_dataloader=val_loader
    )
    return test_loss, test_f1, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--dataset_path', type=str, default='../data/QCdataset', help='dataset_path')
    parser.add_argument('--saved_dir', type=str, default='./saved/class1.pt', help='saved_dir')
    parser.add_argument('--save_file_name', type=str, default='resnet50.pt', help='save_file_name')
    parser.add_argument('--num_epochs', type=int, default=20, help='num_epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--model', type=str, default='ResNet50', help='model')
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='criterion')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')
    parser.add_argument('--val_every', type=int, default=1, help='val_every (default: 1)')
    args = parser.parse_args()

    # hyper parameter
    
    if not os.path.isdir('./saved'):
        os.mkdir('./saved')

    main(args)