import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
import argparse
from euclid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from euclid.conversation import conv_templates, SeparatorStyle
from euclid.model.builder import load_pretrained_model
from euclid.utils import disable_torch_init
from euclid.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
import json
import random
from tqdm import tqdm

def jaccard_similarity(ground_truth, predicted):
    set1 = set(ground_truth)
    set2 = set(predicted)
    return len(set1.intersection(set2)) / len(set1.union(set2))

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


class qa():
    def __init__(self, model, tokenizer, image_processor):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def answer(self, question, image):
        from euclid.mm_utils import process_images, tokenizer_image_token
        image_tensor = process_images([image], self.image_processor, self.model.config).to(self.model.device, torch.float16)
        input_ids = tokenizer_image_token(question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=512,
            )
        outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return outputs

def extract_answer(answer):
    answer = answer.split('is')[1].strip()
    answer = answer.split('are')[1].strip()
    return answer.split(', ')


def answer_match(task, pred, gt):
    pred = [i.strip() for i in pred.split(',')]
    gt = [i.strip() for i in gt.split(',')]
    pred_set = set(pred)
    gt_set = set(gt)

    if task == 'PointLiesOnLine' or task == 'PointLiesOnCircle':
        if pred_set.issubset(gt_set):
            return len(pred_set)/len(gt_set)
        else:
            return 0
    elif task == 'AngleClassification' or task == 'LineComparison' or task == 'Equals':
        if ',' in gt:
            gts = [i.strip() for i in gt.split(',')]
            unique_gts = [''.join(sorted(line)) for line in set(map(frozenset, gts))]
            for gt in unique_gts:
                if set(gt) == set(pred):
                    return 1
            return 0
        return int(set(pred) == set(gt))
    elif task == 'Parallel' or task == 'Perpendicular':
        unique_gt = [''.join(sorted(line)) for line in set(map(frozenset, gt))]
        covered_gt = []
        correct_pred = []
        for p in pred_set:
            for g in unique_gt:
                if set(p).issubset(set(g)):
                    covered_gt.append(g)
                    correct_pred.append(p)
                    break
        # if there is at least one incorrect prediction, return 0
        if len(correct_pred) < len(pred_set):
            return 0
        else:
            return len(covered_gt)/len(unique_gt)

def main(args):
    model_path = args.model_path
    model_base = None
    device = "cuda:0"
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, device=device)
    conv_mode = "qwen_direct"
    model.model.vision_tower.to(model.device)
    model.model.vision_tower.to(model.dtype)
    qa_model = qa(model, tokenizer, image_processor)

    accs = {}
    with open('playground/downstream/geoperception/geoperception/geoperception_data.json', 'r') as f:
        all_data = json.load(f)
    points_on_line_data = [data for data in all_data if data['predicate'] == 'Equals']

    for data in tqdm(all_data, ncols=100):
        conv = conv_templates[conv_mode].copy()
        image = load_image(f'playground/downstream/geoperception/data/{data["data_point"]}/img_diagram.png')
        input_text = f"<image>\n{data['question']}"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # spped up inference
        if data['predicate'] == 'AngleClassification':
            angle_name = prompt.split('angle')[1].split('acute')[0].strip()
            prompt += f'According to the diagram, angle {angle_name} is'
        elif data['predicate'] == 'PointLiesOnLine':
            line_points = prompt.split("line")[1].split('?')[0].strip()
            prompt += f'The point lying on line {line_points}'
        elif data['predicate'] == 'PointLiesOnCircle':
            circle_center = prompt.split("center")[1].split("?")[0].strip()
            prompt += f'The point lying on circle with center {circle_center} are'
        elif data['predicate'] == 'LineComparison':
            prompt += f'The longer line is'
        elif data['predicate'] == 'Perpendicular':
            prompt += f'According to the diagram, the line perpendicular to'
        elif data['predicate'] == 'Parallel':
            prompt += f'According to the diagram, the line parallel to'
        answer = qa_model.answer(prompt, image).split('is')[-1].split('are')[-1].split('as')[-1].split('segment')[-1].split('angle')[-1].strip()
        ground_truth = data['answer']
        acc = answer_match(data['predicate'], answer, ground_truth)
        if data['predicate'] not in accs:
            accs[data['predicate']] = []
        accs[data['predicate']].append(acc)
    for task, accs in accs.items():
        print(f'{task}: {sum(accs)/len(accs):.4f}')
    # with open('playground/geo3k/version_1_acc.txt', 'a') as f:
    #     f.write(f'{model_path.split("/")[-1]:<30}: {acc:.4f}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="checkpoints/euclid-qwen-euclid_conv_large")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)