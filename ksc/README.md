# final-project-level3-cv-19

**모델별 추론시간 정리**
모델 스크립트화 및 모바일 기기에 최적화 후 100회씩 inference  
[PC에서 돌려서 실제 추론시간과 차이 있음.]  
|모델|추론시간|
|:---:|:---:|
|EfficientNet B0|958ms|
|MobileNetV3|799ms|
|Resnet50|789ms|
  
  
  
https://pytorch.org/docs/1.9.0/mobile_optimizer.html?highlight=optimize%20mobile#torch.utils.mobile_optimizer.optimize_for_mobile  
**optimize 실험**  
%%주의%%  
핸드폰에서 실험했으므로, 시행 횟수와 스로틀링으로 인해 매 시행마다 추론 시간이 큰 폭으로 달라져서 객관적인 실험이 불가능 했습니다.  
최종 적용은 insert hoist conv packed params가 앱 크기가 두배 증가하는데 비해 성능 향상 폭이 크지 않다고 판단되어 제외하고  다른 모든 기능을 적용시켰습니다.

  
|적용 optimize 기법|추론시간|
|:---:|:---:|
|미적용|2100ms|
|모두적용|1330ms|
|ReLU/Hardtanh fusion만 사용(기본)|1800ms|
|ReLU/Hardtanh fusion + Remove Dropout|ReLU fusion 대비 약 100ms 감소|
|ReLU/Hardtanh fusion + Conv BN fusion|ReLU fusion 대비 약 100ms 감소|
|ReLU/Hardtanh fusion + insert fold prepack|ReLU fusion 대비 약 50ms 감소|
|ReLU/Hardtanh fusion + insert hoist conv packed params (모델 용량이 두배로 증가)|큰 차이 없음.|
|insert hoist conv packed params 만 제외하고 모두 적용시|1320ms|
