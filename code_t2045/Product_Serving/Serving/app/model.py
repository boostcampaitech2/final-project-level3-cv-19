import io
from typing import List, Dict, Any

import albumentations
import albumentations.pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from efficientnet_pytorch import EfficientNet

import torchvision.models as models

class MyEfficientNet(nn.Module):
    """
    EfiicientNet-b4의 출력층만 변경합니다.
    한번에 18개의 Class를 예측하는 형태의 Model입니다.
    """

    def __init__(self, num_classes: int = 18):
        super(MyEfficientNet, self).__init__()
        self.EFF = EfficientNet.from_pretrained("efficientnet-b4", in_channels=3, num_classes=num_classes)

    def forward(self, x) -> torch.Tensor:
        print(type(x))
        x = self.EFF(x)
        x = F.softmax(x, dim=1)
        return x


def get_model(model_path: str = "../../assets/mask_task/model.pth") -> MyEfficientNet:
    """Model을 가져옵니다"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ###model = MyEfficientNet(num_classes=18).to(device)
    ###model.load_state_dict(torch.load(model_path, map_location=device))

    model = models.vgg11().to(device)
    model.load_state_dict(torch.load(get_config()['model_path'], map_location=device))
    
    return model

DATASET_NORMALIZE_INFO = {
    "COW": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
}
def _transform_image(image_bytes: bytes):
    transform = albumentations.Compose(
        [
            #albumentations.Resize(height=512, width=384),
            albumentations.CenterCrop(200, 200),
            albumentations.Normalize(mean=DATASET_NORMALIZE_INFO["COW"]["MEAN"], std=DATASET_NORMALIZE_INFO["COW"]["STD"]),
            #albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            albumentations.pytorch.transforms.ToTensorV2(),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image_array = np.array(image)
    return transform(image=image_array)["image"].unsqueeze(0)


def predict_from_image_byte(model: MyEfficientNet, image_bytes: bytes, config: Dict[str, Any]) -> List[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import time
    start_time = time.time()
    transformed_image = _transform_image(image_bytes)
    #outputs = model.forward(transformed_image)
    outputs = model.forward(transformed_image.to(device))
    _, y_hat = outputs.max(1)
    duration = time.time() - start_time
    print(duration)
    return config["classes"][y_hat.item()]


#def get_config(config_path: str = "../../assets/mask_task/config.yaml"):
def get_config(config_path: str = "./app/config.yaml"):
    import yaml

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
