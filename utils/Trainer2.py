"""PyTorch trainer module.

- Author: Jongkuk Lim, Junghoon Kim
- Contact: lim.jeikei@gmail.com, placidus36@gmail.com
"""

import os
import shutil
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import f1_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from tqdm import tqdm

import time
import multiprocessing as mp
from utils.utils import dense_crf_wrapper

def convert_model_to_torchscript(
    model: nn.Module, path: Optional[str] = None
) -> torch.jit.ScriptModule:
    """Convert PyTorch Module to TorchScript.

    Args:
        model: PyTorch Module.

    Return:
        TorchScript module.
    """
    model.eval()
    jit_model = torch.jit.script(model)

    if path:
        jit_model.save(path)

    return jit_model

    
def save_model(model, path, data, device):
    """save model to torch script, onnx."""
    try:
        torch.save(model.state_dict(), f=path)
        ts_path = os.path.splitext(path)[:-1][0] + ".ts"
        convert_model_to_torchscript(model, ts_path)
    except Exception:
        print("Failed to save torch")



def _get_n_data_from_dataloader(dataloader: DataLoader) -> int:
    """Get a number of data in dataloader.

    Args:
        dataloader: torch dataloader

    Returns:
        A number of data in dataloader
    """
    if isinstance(dataloader.sampler, SubsetRandomSampler):
        n_data = len(dataloader.sampler.indices)
    elif isinstance(dataloader.sampler, SequentialSampler):
        n_data = len(dataloader.sampler.data_source)
    else:
        n_data = len(dataloader) * dataloader.batch_size if dataloader.batch_size else 1

    return n_data


def _get_n_batch_from_dataloader(dataloader: DataLoader) -> int:
    """Get a batch number in dataloader.

    Args:
        dataloader: torch dataloader

    Returns:
        A batch number in dataloader
    """
    n_data = _get_n_data_from_dataloader(dataloader)
    n_batch = dataloader.batch_size if dataloader.batch_size else 1

    return n_data // n_batch


def _get_len_label_from_dataset(dataset: Dataset) -> int:
    """Get length of label from dataset.

    Args:
        dataset: torch dataset

    Returns:
        A number of label in set.
    """
    if isinstance(dataset, torchvision.datasets.ImageFolder) or isinstance(
        dataset, torchvision.datasets.vision.VisionDataset
    ):
        return len(dataset.classes)
    elif isinstance(dataset, torch.utils.data.Subset):
        return _get_len_label_from_dataset(dataset.dataset)
    else:
        raise NotImplementedError


class TorchTrainer:
    """Pytorch Trainer."""

    def __init__(
        self,
        model: nn.Module,
        seg_model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        model_path: str,
        scaler=None,
        device: torch.device = "cpu",
        verbose: int = 1,
    ) -> None:
        """Initialize TorchTrainer class.

        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            verbose: verbosity level.
        """

        self.model = model
        self.seg_model = seg_model
        self.model_path = model_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.verbose = verbose
        self.device = device
    
    def train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[float, float]:
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            loss and accuracy
        """
        best_test_acc = -1.0
        best_test_f1 = -1.0
        # num_classes = _get_len_label_from_dataset(train_dataloader.dataset)
        num_classes = 5
        label_list = [i for i in range(num_classes)]

        start = time.time()
        normalize = torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        for epoch in range(n_epoch):
            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            self.model.train()
            for batch, (data, labels, mask) in pbar:
                data, labels = data.to(self.device), labels.to(self.device)


                # segmentation part
                output_segmentation = self.seg_model(data)
                # crf
                image_unnorm = 255*torch.div(torch.add(data, -data.min()),torch.add(data.max(), -data.min()))
                probs_seg = torch.nn.functional.softmax(output_segmentation, dim=1).detach().cpu().numpy()

                pool = mp.Pool(mp.cpu_count())
                images_rgb = image_unnorm.detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
                probs_crf = np.array(pool.map(dense_crf_wrapper, zip(images_rgb, probs_seg)))
                pool.close()
                probs_crf = torch.tensor(probs_crf).to(self.device)

                masked_crf_images = torch.mul(image_unnorm, torch.stack([probs_crf[:,1,:,:]]*3, dim=1)).detach().cpu()
                result = []
                for masked_crf_image in masked_crf_images:
                    result.append(normalize(masked_crf_image))
                masked_crf_images = torch.stack(result,dim=0).to(self.device)

                # outputs = self.model(data)
                outputs = self.model(masked_crf_images)
                outputs = torch.squeeze(outputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                f1 = f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0)

                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                    f"F1(macro): {f1:.2f}"
                )
            pbar.close()


            _, test_f1, test_acc = self.test(
                model=self.model, test_dataloader=val_dataloader
            )

            self.scheduler.step()
            # if self.scheduler.name =='ReduceLROnPlateau':
            #     # self.scheduler.step(f1)
            # else:
            #     self.scheduler.step()

            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_f1 = test_f1
            print(f"Model saved. Current best test f1: {best_test_f1:.3f}")
            save_model(
                model=self.model,
                path=self.model_path,
                data=data,
                device=self.device,
            )
            print(f'time: {time.time() - start}')

        return best_test_acc, best_test_f1


    @torch.no_grad()
    def test(
        self, model: nn.Module, test_dataloader: DataLoader
    ) -> Tuple[float, float, float]:
        """Test model.

        Args:
            test_dataloader: test data loader module which is a iterator that returns (data, labels)

        Returns:
            loss, f1, accuracy
        """

        n_batch = _get_n_batch_from_dataloader(test_dataloader)

        running_loss = 0.0
        preds = []
        gt = []
        correct = 0
        total = 0

        num_classes = 5
        label_list = [i for i in range(num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        model.to(self.device)
        model.eval()
        normalize = torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        for batch, (data, labels, mask) in pbar:
            data, labels = data.to(self.device), labels.to(self.device)

            # segmentation part
            output_segmentation = self.seg_model(data)
            
            # crf
            image_unnorm = 255*torch.div(torch.add(data, -data.min()),torch.add(data.max(), -data.min()))
            probs_seg = torch.nn.functional.softmax(output_segmentation, dim=1).detach().cpu().numpy()

            pool = mp.Pool(mp.cpu_count())
            images_rgb = image_unnorm.detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
            probs_crf = np.array(pool.map(dense_crf_wrapper, zip(images_rgb, probs_seg)))
            pool.close()
            probs_crf = torch.tensor(probs_crf).to(self.device)

            masked_crf_images = torch.mul(image_unnorm, torch.stack([probs_crf[:,1,:,:]]*3, dim=1)).detach().cpu()
            result = []
            for masked_crf_image in masked_crf_images:
                result.append(normalize(masked_crf_image))
            masked_crf_images = torch.stack(result,dim=0).to(self.device)

            # outputs = self.model(data)
            outputs = self.model(masked_crf_images)
            
            # if self.scaler:
            #     with torch.cuda.amp.autocast():
            #         outputs = model(data)
            # else:
            #     outputs = model(data)

            # outputs = torch.squeeze(outputs) 
            # running_loss += self.criterion(outputs, labels).item()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()
            pbar.update()
            pbar.set_description(
                f" Val: {'':5} Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
        loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        f1 = f1_score(
            y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0
        )
        return loss, f1, accuracy


def count_model_params(
    model: torch.nn.Module,
) -> int:
    """Count model's parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
