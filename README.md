# Image_Classifier
<h3>목차</h3>
<ul>
  <li>데이터셋 설명</li>
  <li>결과 및 보고</li>
  <li>문제점 및 해결</li>
</ul>

<h1>데이터셋 설명</h1>
<pre>
  데이터셋은 227*227의 크기와 3채널을 가진 이미지 데이터를 사용하였습니다.
  7개의 클래스를 가지고 있으며 각각 329, 205, 235, 134, 151, 245, 399개의 데이터셋을 포함합니다.
  데이터는 아래와 같이 직관적으로 구별하기 쉽지 않은 이미지를 사용하였습니다.
</pre>
  
  ![horse](https://github.com/kdk0411/Image_Classifier/assets/99461483/79e78dd9-b03d-41b6-95dd-6ae8b8fc4708)

## Albumentations
  Albumentations은 부족한 데이터셋을 증강하기 위한 라이브러리입니다.<br>
  회전, 반전 등의 방법을 사용하여 증강합니다.<br>
  이 프로젝트에서는 회전, 상하반전, 특정 색상 부여 이렇게 3가지를 적용하였습니다.
  
## 결과 및 보고

  아래와 같은 결과값이 나왔으며 비교적 성공적입니다.<br>
  가벼운 모델을 사용하였기 떄문에 더 높은 결과를 추출하지 못한것 같습니다.<br>
    
  train loss : 0.7825<br>
  val loss : 0.5374<br>
  val acc : 0.85%<br>
  val f1 score : 0.8238
## 문제점 - 적은 실험 횟수
  config 파일을 사용하여 실험 하였지만 더 많은 모델, 더 많은 데이터 증강을 사용하여 더 높고 정확한 모델을 만들지 못한것이 문제점입니다.
