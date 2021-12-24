# boostcamp_pstage10

# Environment
## 1. Install dependencies
```
pip install -r requirements.txt
```

# Run
## 0. symbolic link
ln -s code_t2045 code
<br>
## 1. train
python train.py  (default)
<br>
or
<br>
python train.py --model_config ${path_to_model_config} --data_config ${path_to_data_config}

## 2. inference(submission.csv)
python inference.py (default)
<br>
or
<br>
python inference.py --model_config configs/model/mobilenetv3.yaml --weight exp/2021-05-13_16-41-57/best.pt --img_root /opt/ml/data/test --data_config configs/data/taco.yaml3
<br>

# Modification
## 1. Augmentation
source: src/augmentation/policies.py <br>
내용: 아래의 augmentation을 train augmentation에  추가 <br> 
            transforms.CenterCrop((200, 200))  <br>
<br>
## 2. Model
source: train.py, inference.py <br>
내용: pretrained vgg11을 선택. <br>
        model = models.vgg11(pretrained=True) <br>
<br>
 

# Reference
Our basic structure is based on [Kindle](https://github.com/JeiKeiLim/kindle)(by [JeiKeiLim](https://github.com/JeiKeiLim))
