# 소고기 단면을 이용한 품질 평가
Final Project in 2nd BoostCamp AI Tech 2기 by **빨간맛(CV 19조)**

## Content
- [Project Overview](#Project-Overview)
- [Dataset](#Dataset)
- [Model](#Model)
- [Product Serving](#Product-Serving)

## Project Overview
-  소고기 단면을 이용한 품질 평가
  - 실생활에서 내가 먹는 고기의 육질을 평가해주는 한우 등심 분류기

-  프로젝트 배경 및 기대효과
  - 소비자로서 실생활에서 내가 먹는 소고기 등급이 맞는 지 검증
  - 부스트캠프를 통해서 배운 AI 지식을 실생활에 활용

## Dataset
-  축산물 품질(QC) 이미지 [https://aihub.or.kr/aidata/30733]
  - Training Dataset: 69434장 
  - Validation Dataset: 8679장
![Image_TrainVal](https://user-images.githubusercontent.com/4301916/147234012-482f65f3-ea5f-411a-a177-b489c632e77c.jpg)
 
-  실제 판매되는 등심 데이터 수집 
  - Test Dataset: 110장 
![Image_Test](https://user-images.githubusercontent.com/4301916/147234479-834a246e-61a0-4cc1-94a0-4da794ad3f44.jpg)


## Model
-  2단계로 구성 (Segmentation -> Classification 2)
-  Segmentation  
  - UNet - Encoder: ResNet50
    - 소고기 단면의 육질을 다른 배경으로 부터 분리 
    - mIoU: 0.8911, val mIoU: 0.9384 
-  Classification   
  - ResNet50
    - Segmentation mask를 사용하여 배경을 제거하여 classification

## Product Serving
-  MObile Application
![Slide17](https://user-images.githubusercontent.com/4301916/147235398-abd09838-8de0-486f-ba47-dc18ace445a1.jpg)
![Slide20](https://user-images.githubusercontent.com/4301916/147235450-21ea19bc-75e1-4e01-aaf6-340911114348.jpg)
   
-  Web Application   
![Slide21](https://user-images.githubusercontent.com/4301916/147235460-e3d58c2e-1df4-413d-8e02-b0080d0e2551.jpg)


final-project-level3-cv-19 created by GitHub Classroom
