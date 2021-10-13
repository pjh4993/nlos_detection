<div align="center">
    <img src="docs/adel-logo.svg" width="160" >
</div>

#  NLOS object detection

AdelaiDet 프로젝트(https://github.com/aim-uofa/AdelaiDet) 를 기반으로 작성한 NLOS object detection 코드 입니다. 
AdelaiDet이 Detectron2를 기반으로 하고 있기 때문에 Detectron2도 같이 필요합니다.

## 코드 structure

### Source & Tools

Model과 Dataloader, Trainer를 비롯한 학습과 evaluation관련 소스 코드들은 adet/ 디렉토리 안에 있습니다.
이 코드들을 실행시키기 위한 script파일들은 tools/ 안에 있습니다. 주로 tools/train_net.py 를 통해 돌리게 됩니다. 

tools/train_net.py의 argument들은 detectron2의 argument parser를 참고해주시면 되겠습니다.

configs/ 디렉토리 안에는 tools/ 의 script들을 돌릴시 필요한 configuration file 들을 모아놓은 곳 입니다.
NLOS object detection에 관련된 configuration들은 configs/NLOS 안에 있습니다.

NLOS dataset은 datasets/ 폴더안에 위치해주셔야 합니다. 정확한 경로명은 adet/data/builtin.py 에서 _PREDEFINED_SPLITS_NLOS 를 통해 확인할 수 있습니다.

### Data Loader

NLOS 데이터셋은 COCO와 같은 형식으로 annotation이 구성되어 있습니다. annotation json파일을 읽어서 Dataloader에게 전달해주기 위한 list로 바꿔주는 코드가 adet/data/datasets/nlos.py에 있습니다.

json 파일의 구조는 다음과 같습니다.

```
root
|-- image_groups
|    |
|    |--[id, group_name, gt_image, laser_image, depthimage, position]
|-- annotations
     |
     |--[id, image_group_id, category_id, area, bbox]
```

### Model



### Evaluator

## Installation

## Quick Start
