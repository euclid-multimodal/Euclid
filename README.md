<p align="center">
  <img src="./playground/demo_images/euclid_symbol.png" alt="Euclid" width="150">
  <h4 align="center" style="margin-top: -20px;">Supercharging Multimodal LLMs with Synthetic High-Fidelity Visual Descriptions</h4>
</p>

<div style="font-family: charter;" align="center">
    <a href="https://saccharomycetes.github.io/" target="_blank">Jiarui Zhang</a>,
    <a href="https://ollieliu.com/" target="_blank">Ollie Liu</a>,
    <a href="https://github.com/yiranyyu" target="_blank">Tianyu Yu</a>,
    <a href="https://jameshujy.github.io/" target="_blank">Jinyi Hu</a>,
    <a href="https://willieneis.github.io/" target="_blank">Willie Neiswanger</a>
</div>
<br>
<div align="center">
[<a href="https://arxiv.org/abs/2412.08737">paper</a> 📄]
[<a href="https://euclid-multimodal.github.io/">website</a> 🌐]
[<a href="https://huggingface.co/papers/2412.08737">huggingface</a> 🌐]
<br>
[<a href="https://huggingface.co/datasets/EuclidAI/Geoperception">Geoperception</a> 🤗]
[<a href="https://huggingface.co/EuclidAI/Euclid-convnext-large">Euclid convnext-large</a> 🤗]
[<a href="https://huggingface.co/EuclidAI/Euclid-convnext-xxlarge">Euclid convnext-xxlarge</a> 🤗]
<br>
<hr>
</div>

<p align="center">
  <img src="./playground/demo_images/geoperception_examples.png" alt="Geoperception" width="600">
</p>

This project studies the low-level geometric understanding of multimodal LLMs, including:


- **Geoperception Benchmark**: First benchmark focusing specifically on fine-grained low-level geometric perception in MLLMs.
- **Synthetic Data Engine + Empirical Study**: A comprehensive study of the multimodal LLM design space using our developed synthetic data engine (We found that data curriculum is helpful).
- **Euclid Model**: Two less than 3B models surpassing the best proprietary MLLMs on Geoperception.



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
## Contents

- [Geoperception](#geoperception)
- [Image Generation Engine](#geometry-image-generation-engine)
- [Empirical Study and Euclid Training Scripts](#scripts-for-empirical-study-and-euclid-training)
- [Euclid Models](#euclid-models)

## Geoperception

Download the dataset:

```
from datasets import load_dataset
loaded_dataset = load_dataset("EuclidAI/Geoperception")['train']
```

You can also directly evaluate the Euclid model on the Geoperception dataset by running (this will download the dataset automatically):
```
python euclid/eval/run_euclid_geo.py --model_path $MODEL_PATH --device cuda
```

please refer to [Euclid Models](#euclid-models) for downloading the Euclid model.

To evaluate your own model on the Geoperception dataset, you should append the detailed instruction from [general_eval_prompt.json](./euclid/eval/general_eval_prompt.json) to the original question to make the model follow the format of the answer, then parse the model's prediction and compute the accuracy using the function `compute_accuracy` in [answer_parser.py](./euclid/eval/answer_parser.py).

## Geometry Image Generation Engine

The implementation of our geometry image generation engine which is able to produce numerical instances of the logical geometric shapes as many as you want.

[training_data_engine.py](./image_engine/training_data_engine.py) produce geometry images, questions, and answers for model training, with the following components:
- [produce_shape.py](./image_engine/produce_shape.py) contains our carefully designed geometry shapes for empirical study and Euclid training.
- [question_engine.py](./image_engine/question_engine.py) takes the geometry shapes as input and generate the questions and corresponding answers for training and testing.
- [alphageometry](./image_engine/alphageometry) convert the logical geometry shapes into pixel-level images, which is bulit based on [AlphaGeometry](https://github.com/google-deepmind/alphageometry).

Prepare Testing Data for Empirical Study
```
from image_engine.training_data_engine import *

tasks = ['PointLiesOnLine_empirical', 'LineComparison_empirical']
stages = [1, 2, 3]

data_engine = Euclid_DataEngine(tasks=tasks, stages=stages, attenuation_rate=0, image_path='./playground/data/testing_data/image', tol=0.3)

datas = data_engine.generate_datas(6000)

with open('./playground/data/testing_data/data.json', 'w') as f:
    json.dump(datas, f, indent=4)
```

Prepare Testing Data for Euclid Training
```
from image_engine.training_data_engine import *

tasks = ['PointLiesOnLine', 'PointLiesOnCircle', 'AngleClassification', 'LineComparison', 'Parallel', 'Perpendicular', 'Equal']
stages = [1, 2, 3]

data_engine = Euclid_DataEngine(tasks=tasks, stages=stages, attenuation_rate=0, image_path='./playground/data/euclid/image', tol=0.3)

datas = data_engine.generate_datas(10500)

with open('./playground/data/euclid/data.json', 'w') as f:
    json.dump(datas, f, indent=4)
```

To construct a new geometry logical shape, you can refer to [produce_shape.py](./image_engine/produce_shape.py) and prepare a new dictionary, fields in the dictionary are:
- `orcle_text`: The text description of the geometry logical shape.
- `connection_list`: The list of connections between the points.
- `points_set`: The logical relationships between the geometry components, which is used in the question generation.
- `remove_set`: The set of points you used in constructing the shape, but you don't want to show in the image.
- `highlights`: Highlights of perpendicular and parallel.
- `equals`: Equal length segments and equal angles.

A test example:

```
from image_engine.training_data_engine import *

shape_data = {
    'orcle_text': 'A B C = triangle A B C; O = circle A B C; D = on_circle D O C, angle_bisector C A B; E = midpoint A B',
    'connection_list': 'AB, AC, BC, CD, DB, AD',
    'remove_set': 'O',
    'points_set': ['angles_value;BCA=actual'],
    'equals': ['angles_value;BCA=actual', 'angles;DAC=DAB', 'segments;AE=EB']
}

g, letter_map, highlights, equals, shape_data = get_graph_from_shape(shape_data)

draw_g(g, highlights=highlights, equals=equals)
```

One image instance generated by the above code:
<p align="center">

  <div style="text-align: center;">
    <img src="./playground/demo_images/engine_test.png" alt="Geoperception" width="300">
  </div>
</p>

## Scripts for Empirical Study and Euclid Training

With the dataset generated by our geometry engine, we investigated four key aspects of MLLM model design:

- **Language Model Size**: Scaling LLM sizes (0.5B, 1.5B, and 3B parameters) did not substantially improve low-level geometric perception tasks.

- **Vision Tower Choices**: Compared with Vision Transformer-based encoders, ConvNeXt family encoders produced faster and more stable learning of geometric concepts.

- **Tuning Vision Encoders**: Tuning the visual encoder did not yield significant benefits in our experimental setting.

- **Curriculum Learning**: Incorporating easier examples before introducing more complex shapes enhanced model convergence and efficiency. Sequential or mixed training strategies that leverage curriculum-based approaches led to robust improvements over directly training on the hardest problems.

### Empirical Study Example Script

```
bash scripts/empirical_study/run.sh
```

### Euclid Training Example Script 
Training Euclid-ConvNeXt-Large takes around 16 hours on a single A100-80GB GPU.

Training Euclid-ConvNeXt-XXLarge takes around 24 hours on a single A100-80GB GPU. 

```
bash scripts/euclid_training/run.sh
```

Important Training Arguments:
- `--tune_vision_tower`: Whether to finetune the vision tower.
- `--language_model`: The language model to be used. Currently, we support [Qwen-2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) and [Qwen-2](https://huggingface.co/collections/Qwen/qwen2-7b-instruct-8192) series. Use the original huggingface model name for input, such as `Qwen/Qwen2.5-1.5B-Instruct`.
- `--vision_tower`: The vision tower to be used. Currently, we support following models:
  - [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)  
  - [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)  
  - [facebook/dinov2-giant](https://huggingface.co/facebook/dinov2-giant)  
  - [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large)  
  - [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)  
  - [google/siglip-so400m-patch14-224](https://huggingface.co/google/siglip-so400m-patch14-224)  
  - [laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup)  
  - [laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup)  
  - [laion/CLIP-ViT-g-14-laion2B-s34B-b88K](https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s34B-b88K)  
  - [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
  Use the original huggingface model name for input.
- `--tasks`: The tasks of the empirical study, containing:
  - **PointLiesOnLine**
  - **LineComparison**
  - **PointLiesOnCircle**
  - **AngleClassification**
  - **Parallel**
  - **Perpendicular**
  - **Equal**
  - **PointLiesOnLine_empirical** (for empirical study only)
  - **LineComparison_empirical** (for empirical study only)
  If there are multiple tasks, use comma to separate them, such as `--tasks PointLiesOnLine,LineComparison`.
- `--stages`: The stages of the empirical study, for all tasks we have 3 stages currently. For example, `--stages 1,2,3` will train the model on all tasks with stage 1, 2, and 3 respectively.
- `--attenuation_rate`: Set to 0 for static (single stage or mixed) training in empirical study, set to 1.5 (default) for curriculum learning.

## Euclid Models

- [Euclid-ConvNeXt-Large](https://huggingface.co/EuclidAI/Euclid-convnext-large)
- [Euclid-ConvNeXt-XXLarge](https://huggingface.co/EuclidAI/Euclid-convnext-xxlarge)

To evaluate the Euclid models on the Geoperception dataset, first download the model by:
```
pip install -U "huggingface_hub[cli]"
huggingface-cli download --cache-dir $MODEL_PATH EuclidAI/Euclid-convnext-large
```

Then run:
```
python euclid/eval/run_euclid_geo.py --model_path $MODEL_PATH --device cuda
```

Other usage: Our current Euclid model can only follow specific instructions, so it is challenging to use it as a general-purpose MLLM. However, the model demonstrates strength in low-level visual perception, this capability makes it potentially valuable for serving as a base model for specialized downstream tasks requiring high-fidelity low-level visual understanding.

## Citation

If you find Euclid useful for your research and applications, please cite using this BibTeX:
```bibtex 
@article{zhang2024euclid,
  title={Euclid: Supercharging Multimodal LLMs with Synthetic High-Fidelity Visual Descriptions},
  author={Zhang, Jiarui and Liu, Ollie and Yu, Tianyu and Hu, Jinyi and Neiswanger, Willie},
  journal={arXiv preprint arXiv:2412.08737},
  year={2024}
}
```

## Acknowledgements
[LLaVA](https://github.com/haotian-liu/LLaVA): The codebase that our training framework is built on.

[AlphaGeometry](https://github.com/google-deepmind/alphageometry): The codebase that our dataset generation engine is built on.

[Openclip](https://github.com/mlfoundations/open_clip): The codebase containing the pre-trained visual encoders that we used for our experiments.