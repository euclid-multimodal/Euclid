<p align="center">
  <img src="./playground/demo_images/euclid_symbol.png" alt="Euclid" width="150">
  <h1 align="center" style="margin-top: -20px;">Supercharging Multimodal LLMs with Synthetic High-Fidelity Visual Descriptions</h1>
</p>

<div style="font-family: charter;">
    <a href="https://saccharomycetes.github.io/" target="_blank">Jiarui Zhang</a>,
    <a href="https://ollieliu.com/" target="_blank">Ollie Liu</a>,
    <a href="https://github.com/yiranyyu" target="_blank">Tianyu Yu</a>,
    <a href="https://jameshujy.github.io/" target="_blank">Jinyi Hu</a>,
    <a href="https://willieneis.github.io/" target="_blank">Willie Neiswanger</a>
</div>

## Updates
- [12/05/24] 🔥 Euclid paper is released! We also release our dataset generation engine and training scripts.


## Installation

```
conda create -n euclid python=3.10 -y
conda activate euclid
pip3 install --upgrade pip
pip3 install -e .
pip3 install flash-attn --no-build-isolation
```

## Dataset Preparation

### Testing Data for Empirical Study
```
from image_engine.training_data_engine import *

tasks = ['PointLiesOnLine_empirical', 'LineComparison_empirical']
stages = [1, 2, 3]

data_engine = Euclid_DataEngine(tasks=tasks, stages=stages, attenuation_rate=0, image_path='./playground/data/testing_data/image', tol=0.3)

datas = data_engine.generate_datas(6000)

with open('./playground/data/testing_data/data.json', 'w') as f:
    json.dump(datas, f, indent=4)
```

### Testing Data for Euclid Training
```
from image_engine.training_data_engine import *

tasks = ['PointLiesOnLine', 'PointLiesOnCircle', 'AngleClassification', 'LineComparison', 'Parallel', 'Perpendicular', 'Equal']
stages = [1, 2, 3]

data_engine = Euclid_DataEngine(tasks=tasks, stages=stages, attenuation_rate=0, image_path='./playground/data/euclid/image', tol=0.3)

datas = data_engine.generate_datas(10500)

with open('./playground/data/euclid/data.json', 'w') as f:
    json.dump(datas, f, indent=4)
```

## Scripts for Empirical Study and Euclid Training

### Empirical Study Example Script

```
bash scripts/empirical_study/run.sh
```


### Euclid Training

```
bash scripts/euclid_training/run.sh
```



## Acknowledgements
[LLaVA](https://github.com/haotian-liu/LLaVA): The codebase that our training framework is built on.

[AlphaGeometry](https://github.com/google-deepmind/alphageometry): The codebase that our dataset generation engine is built on.

[Openclip](https://github.com/mlfoundations/open_clip): The codebase containing the pre-trained visual encoders that we used for our experiments.
