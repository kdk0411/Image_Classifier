# Image_Classifier
<h3>목차</h3>
<ul>
  <li>데이터셋 설명</li>
  <li>결과 및 보고</li>
  <li>문제점 및 해결</li>
</ul>

<h1>데이터셋 설명</h1>
  데이터셋은 227*227의 크기와 3채널을 가진 이미지 데이터를 사용하였습니다.
  7개의 클래스를 가지고 있으며 각각 329, 205, 235, 134, 151, 245, 399개의 데이터셋을 포함합니다.
  데이터는 아래와 같이 직관적으로 구별하기 쉽지 않은 이미지를 사용하였습니다.
  
  ![horse](https://github.com/kdk0411/Image_Classifier/assets/99461483/79e78dd9-b03d-41b6-95dd-6ae8b8fc4708)

<h1>Albumentations</h1>
  Albumentations은 부족한 데이터셋을 증강하기 위한 라이브러리입니다.
  회전, 반전 등의 방법을 사용하여 증강합니다.
  이 프로젝트에서는 회전, 상하반전, 특정 색상 부여 이렇게 3가지를 적용하였습니다.

## 결과 및 보고

## 문제점 
  1. 데이터셋 불균형 및 부족
  2. 모델의 학습에대한 신빙성
  3. 모델 사용

### 1. 데이터셋 불균형 및 부족
  신생아의 울음소리 음성 데이터를 배포하는 것이 법적으로 문제가 있다는 해외 기업의 조언에 따라서
  모든 음성데이터는 Kaggle에서 가져온 데이터셋입니다.
  그에 따라 풍부하지 못한 데이터셋을 가졌습니다.
  음성 데이터를 Kaggle에서만 가져온 이유는 데이터의 신빙성떄문입니다.
  Kaggle에서 조차도 성인이 아이의 울음소리를 따라한 데이터가 존재하였기 때문에 모두 믿고 사용할 수 없었습니다.
  데이터셋의 절대적 양이 적다는것은 모델이 학습함에 있어서 과적합현상을 야기할 수 있습니다.
  이를 해결하기 위해 CNN에 ResidualBlock을 추가하는 방식을 사용하고 매 학습시에 'shuffle=True'를 사용하여
  학습 데이터의 순서를 바꾸며 학습 시켰습니다. 이에 초기 과적합으로 인해 정확도가 66.66% 등의 잘못된 학습을 고쳤습니다.

### 2. 모델의 학습에 대한 신빙성
  앞서 말했듯이 모델이 데이터를 옳바르게 학습한것이 맞는지에 대한 신빙성이 부족하였습니다.
  프로젝트 종료시에도 88~97% 까지의 넓은 정확도의 분포를 보여주었습니다.(해당 프로젝트는 가장 높은 정확도가 나온 모델을 저장하여 사용하였습니다.)
  때문에 정확도가 높은 모델이 나올때까지 재학습 시켜야 했습니다.
  문제점으로는 'shuffle=True'과 적은 데이터셋으로 예상하고 있습니다.

### 3. 모델 사용
  Pytho 기반으로 제작한 모델을 java기반의 서버에서 사용하는 것에 대한 정보가 없었기 떄문에
  모델 사용에 있어서 문제점을 가지고 있었습니다.
  이는 Flask를 사용하여 Node.js 서버와 통신하는 방법을 사용하였습니다.
