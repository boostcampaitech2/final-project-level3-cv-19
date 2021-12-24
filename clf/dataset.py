import numpy as np
import torch
from torch.utils.data import Dataset
import os
from glob import glob
from PIL import Image
import cv2
import random
import matplotlib.pyplot as plt
import json
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from torchvision.datasets import ImageFolder

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class CustomDataset(Dataset):
    
    def __init__(self, root, train, image_transform, background_transform):
        
        self.root = root
        self.image_transform = image_transform
        self.background_transform = background_transform
        self.normalize = A.Normalize()
        self.toTensor = ToTensorV2()
        self.split = 'train' if train else 'valid'
        
        # images and labels
        self.classes = sorted(os.listdir(os.path.join(root, 'image', self.split)))
        self.jsons = []
        self.labels = []
           
        for i, _class in enumerate(self.classes):
            _jsons = [x for x in listdir_fullpath(os.path.join(root, 'label', self.split, _class)) if not x.startswith('.')]
            self.jsons += _jsons
            self.labels += [i] * len(_jsons)
            
        # background images
        self.backgrounds = [cv2.imread(img, cv2.IMREAD_COLOR) for img in listdir_fullpath(os.path.join(root, 'background'))]
        self.backgrounds = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.backgrounds]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        try:
            json_path = self.jsons[idx]
            label = self.labels[idx]

            with open(json_path, 'r') as f:
                label_info = json.load(f)['label_info']

            background = self.backgrounds[np.random.choice(len(self.backgrounds))]
            background = self.background_transform(image=background)['image'].copy()

            image = Image.open(os.path.join(self.root, 'image', self.split, self.classes[label], label_info['image']['file_name']))
            image = np.array(image)
            mask = np.array(label_info['shapes'][0]['points'], dtype=np.int32) // 2

            left, top = np.min(mask, axis=0)
            right, bottom = np.max(mask, axis=0)
            image = image[top:bottom, left:right]

            mask -= np.min(mask, axis=0)
            empty_array = np.zeros((bottom-top, right-left), np.uint8)
            mask = cv2.fillPoly(empty_array, [mask], 1)

            transformed = self.image_transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

            h, w = image.shape[:2]
            x = np.random.randint(5, 220-5-h)
            y = np.random.randint(5, 220-5-w)

            background[x:x+h, y:y+w] = cv2.bitwise_and(background[x:x+h, y:y+w], background[x:x+h, y:y+w], mask=(1-mask)) \
                                     + cv2.bitwise_and(image, image, mask=mask)

            image = self.normalize(image=background)['image']
            image = self.toTensor(image=image)['image']

            return image, label
        except:
            return self.__getitem__(idx-1)

def get_train_dataset():

    image_transform = A.Compose([
        A.LongestMaxSize(max_size=224),
        A.RandomScale(scale_limit=(-0.4, -0.1), p=1),
        A.HorizontalFlip(p=0.5),
        A.RGBShift(always_apply=False, p=1.0, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20))
    ])

    background_transform = A.Compose([
        A.RandomResizedCrop(224, 224, ratio=(1,1)),
        A.HorizontalFlip(p=0.5),
    ])
    
    return CustomDataset('data', True, image_transform, background_transform)

def get_valid_dataset():
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    val_dataset = ImageFolder(os.path.join('data/image/valid'),
                        transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop((224,224)),
                            transforms.ToTensor(),
                            normalize,
                    ]))

    return val_dataset

def get_test_dataset():
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    val_dataset = ImageFolder(os.path.join('test_imgs'),
                        transforms.Compose([
                            transforms.Resize(240),
                            transforms.CenterCrop((224,224)),
                            transforms.ToTensor(),
                            normalize,
                    ]))

    return val_dataset


class CustomCropDataset(Dataset):
    
    def __init__(self, root, train, image_transform):#, background_transform):
        
        self.root = root
        self.image_transform = image_transform
        #self.background_transform = background_transform
        self.normalize = A.Normalize()
        self.toTensor = ToTensorV2()
        self.split = 'train' if train else 'valid'
        
        # images and labels
        self.classes = os.listdir(os.path.join(root, 'image', self.split))
        self.jsons = []
        self.labels = []
           
        for i, _class in enumerate(self.classes):
            _jsons = listdir_fullpath(os.path.join(root, 'label', self.split, _class))
            self.jsons += _jsons
            self.labels += [i] * len(_jsons)
            
        # background images
        #self.backgrounds = [cv2.imread(img, cv2.IMREAD_COLOR) for img in listdir_fullpath(os.path.join(root, 'background'))]
        #self.backgrounds = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.backgrounds]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        try:
            json_path = self.jsons[idx]
            label = self.labels[idx]

            with open(json_path, 'r') as f:
                label_info = json.load(f)['label_info']

            #background = self.backgrounds[np.random.choice(len(self.backgrounds))]
            #background = self.background_transform(image=background)['image'].copy()

            image = Image.open(os.path.join(self.root, 'image', self.split, self.classes[label], label_info['image']['file_name']))
            image = np.array(image)
            mask = np.array(label_info['shapes'][0]['points'], dtype=np.int32) // 2

            left, top = np.min(mask, axis=0)
            right, bottom = np.max(mask, axis=0)
            image = image[top:bottom, left:right]

            #mask -= np.min(mask, axis=0)
            #empty_array = np.zeros((bottom-top, right-left, 3), np.uint8)
            #mask = cv2.fillPoly(empty_array, [mask], (1,1,1))

            transformed = self.image_transform(image=image)['image']#, mask=mask)
            #image, mask = transformed['image'], transformed['mask']

            #image = mask * image
    #         h, w = image.shape[:2]
    #         x = np.random.randint(5, 220-5-h)
    #         y = np.random.randint(5, 220-5-w)

    #         background[x:x+h, y:y+w] = cv2.bitwise_and(background[x:x+h, y:y+w], background[x:x+h, y:y+w], mask=(1-mask)) \
    #                                  + cv2.bitwise_and(image, image, mask=mask)

            image = self.normalize(image=transformed)['image']
            image = self.toTensor(image=image)['image']

            return image, label
        
        except:
            return self.__getitem__(idx-1)

def get_crop_train_dataset():

    image_transform = A.Compose([
        A.CenterCrop(224, 224),
        A.RandomRotate90(),
        A.MotionBlur(p=1.0),
        # A.RandomScale(scale_limit=(-0.4, -0.1), p=1),
        A.HorizontalFlip(p=0.5),
    ])

    return CustomCropDataset('data', True, image_transform)

def get_crop_valid_dataset():
    
    image_transform = A.Compose([
        A.CenterCrop(224, 224),
        #A.MotionBlur(p=1.0),
        # A.RandomScale(scale_limit=(-0.4, -0.1), p=1),
        #A.HorizontalFlip(p=0.5),
    ])

    return CustomCropDataset('data', False, image_transform)

    return val_dataset

def get_crop_test_dataset():
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    val_dataset = ImageFolder(os.path.join('test_imgs'),
                        transforms.Compose([
                            transforms.Resize(448),
                            transforms.CenterCrop((224,224)),
                            transforms.ToTensor(),
                            normalize,
                    ]))

    return val_dataset


class CustomExtDataset(Dataset):
    
    def __init__(self, root, train, image_transform, background_transform):
        
        self.root = root
        self.image_transform = image_transform
        self.background_transform = background_transform
        self.normalize = A.Normalize()
        self.toTensor = ToTensorV2()
        self.split = 'train' if train else 'valid'
        
        # images and labels
        self.classes = os.listdir(os.path.join(root, 'image', self.split))
        self.jsons = []
        self.labels = []
           
        for i, _class in enumerate(self.classes):
            _jsons = listdir_fullpath(os.path.join(root, 'label', self.split, _class))
            self.jsons += _jsons
            self.labels += [i] * len(_jsons)
            
        # background images
        self.backgrounds = [cv2.imread(img, cv2.IMREAD_COLOR) for img in listdir_fullpath(os.path.join(root, 'background'))]
        self.backgrounds = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.backgrounds]
        
        self.classes += ['back']
        
    def __len__(self):
        return len(self.labels) + int(0.25*len(self.labels))
    
    def __getitem__(self, idx):
        
        try:
            
            if 0 <= idx < len(self.labels):

                json_path = self.jsons[idx]
                label = self.labels[idx]

                with open(json_path, 'r') as f:
                    label_info = json.load(f)['label_info']

                #background = self.backgrounds[np.random.choice(len(self.backgrounds))]
                #background = self.background_transform(image=background)['image'].copy()

                image = Image.open(os.path.join(self.root, 'image', self.split, self.classes[label], label_info['image']['file_name']))
                image = np.array(image)
                mask = np.array(label_info['shapes'][0]['points'], dtype=np.int32) // 2

                left, top = np.min(mask, axis=0)
                right, bottom = np.max(mask, axis=0)
                image = image[top:bottom, left:right]

                #mask -= np.min(mask, axis=0)
                #empty_array = np.zeros((bottom-top, right-left, 3), np.uint8)
                #mask = cv2.fillPoly(empty_array, [mask], (1,1,1))

                transformed = self.image_transform(image=image)['image']#, mask=mask)
                #image, mask = transformed['image'], transformed['mask']

                #image = mask * image
        #         h, w = image.shape[:2]
        #         x = np.random.randint(5, 220-5-h)
        #         y = np.random.randint(5, 220-5-w)

        #         background[x:x+h, y:y+w] = cv2.bitwise_and(background[x:x+h, y:y+w], background[x:x+h, y:y+w], mask=(1-mask)) \
        #                                  + cv2.bitwise_and(image, image, mask=mask)

                image = self.normalize(image=transformed)['image']
                image = self.toTensor(image=image)['image']

                return image, label
            
            else:
                
                image = self.backgrounds[idx % len(self.backgrounds)]
                image = self.background_transform(image=image)['image'].copy()
                image = self.normalize(image=image)['image']
                image = self.toTensor(image=image)['image']
                
                return image, len(self.classes) - 1
        
        except:
            return self.__getitem__(idx-1)
        
def get_ext_train_dataset():

    image_transform = A.Compose([
        A.CenterCrop(224, 224),
        A.MotionBlur(p=1.0),
        # A.RandomScale(scale_limit=(-0.4, -0.1), p=1),
        A.HorizontalFlip(p=0.5),
    ])

    background_transform = A.Compose([
        A.RandomResizedCrop(224, 224, ratio=(1,1)),
        A.HueSaturationValue(always_apply=False, p=1.0, hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20)),
        A.HorizontalFlip(p=0.5),
    ])

    return CustomExtDataset('data', True, image_transform, background_transform)