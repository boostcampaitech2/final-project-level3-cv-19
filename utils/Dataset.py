import os, random, cv2, json
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms

grade_to_eng = {'1':'one', '1+':'onep', '1++':'onepp', '3':'three', '2':'two'}
eng_to_label = {'one':0, 'onep':1, 'onepp':2, 'three':3, 'two':4}

class CustomDataset(Dataset):
    def __init__(self, data_path='../data/QCdataset', mode = 'train', app_mode='segmentation',transform = None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.app_mode = app_mode
        self.transform = transform
        if self.mode in ['train', 'val']:
            self.json_path_list = self.get_json_list_from_folder()
        else:
            self.json_path_list = self.get_img_list_from_folder()
        self.grade_to_class = {'1':0, '1+':1, '1++':2, '3':3, "2":4}

    def get_json_list_from_folder(self):
        json_root = os.path.join(self.data_path, "labels", self.mode)
        json_dir_list = os.listdir(json_root)
        json_path_list = []
        for dir in json_dir_list:
            json_dir = os.path.join(json_root, dir)
            class_json_list = os.listdir(json_dir)
            class_json_list = [os.path.join(json_dir, class_label) for class_label in class_json_list if class_label[0]!='.']
            json_path_list += class_json_list
        return json_path_list

    def get_img_list_from_folder(self):
        json_root = os.path.join(self.data_path, "images", self.mode)
        json_dir_list = os.listdir(json_root)
        json_path_list = []
        for dir in json_dir_list:
            json_dir = os.path.join(json_root, dir)
            class_json_list = os.listdir(json_dir)
            class_json_list = [os.path.join(json_dir, class_label) for class_label in class_json_list if class_label[0]!='.']
            json_path_list += class_json_list
        return json_path_list
        

    def get_img_path_from_json_path(self, json_path, img_name):
        splited_json_path = json_path.split('/')
        splited_json_path[3] = 'images'
        splited_json_path[6] = img_name
        return '/'.join(splited_json_path)

    def __getitem__(self, index):
        json_path = self.json_path_list[index]
        if self.mode in ['train', 'val']:
            with open(json_path) as f:
                json_info = json.load(f)
            img_name = json_info['label_info']['image']['file_name']
            img_path = self.get_img_path_from_json_path(json_path, img_name)
        else:
            img_path = json_path

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image /= 255.0

        if self.mode in ['train', 'val']:
            label = self.grade_to_class[json_info['label_info']['shapes'][0]['grade']]

            width, height = json_info['label_info']['image']['width'], json_info['label_info']['image']['height']
            mask = np.zeros((int(0.5*height), int(0.5*width), 3),np.uint8)
            mask_contour_points = np.array(json_info['label_info']['shapes'][0]['points'])/2
            mask_contour_points = np.array(mask_contour_points,dtype=np.int32)
            if self.app_mode=="semantic":
                value = label+1
            else:
                value = 1
            mask = cv2.fillPoly(mask, [mask_contour_points], (value, value, value))
            # mask = mask/255.0

            # image = image*mask
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.mode in ['test']:
            splited_path = img_path.split('/')
            label = splited_path[-2]
            label = eng_to_label[label]
            # image = image*mask
            mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            image, mask  = self.transform(image=image, mask=mask)
        return image, label, mask

    def __len__(self):
        return(len(self.json_path_list))

def custom_augment_train():
    return transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )


class CustomAugmentation:
    def __init__(self, mode = 'train'):
        if mode == 'train':
            self.transform = A.Compose([
                                    A.Resize(224,224),
                                    A.HorizontalFlip(p=0.5),
                                    A.VerticalFlip(p=0.5),
                                    A.RandomRotate90(p=0.5),
                                    # A.RGBShift(r_shift_limit=[0,30], g_shift_limit=0, b_shift_limit=0, always_apply=False, p=0.5),
                                    A.Normalize(),
                                    ToTensorV2()
                                    ])
        elif mode == 'val':
            self.transform = A.Compose([
                                    A.Resize(224,224),
                                    A.Normalize(),        
                                    ToTensorV2()
                                    ])
        else:
            self.transform = A.Compose([
                                    A.Normalize(),  
                                    ToTensorV2()
                                    ])

    def __call__(self, image, mask=None):
        transformed = self.transform(image=np.array(image), mask=np.array(mask))
        return transformed['image'], transformed['mask']
        # transformed = self.transform(image=np.array(image))
        # return transformed['image']
        # if mask==None:
        #     return self.transform(image=image)
        # else:
        #     return self.transform(image=image, mask=mask)


if __name__=="__main__":
    transform = CustomAugmentation('train')
    dataset = CustomDataset(transform=transform, mode='train', app_mode='semantic')
    print(dataset[0])