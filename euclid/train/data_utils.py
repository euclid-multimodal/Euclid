import os
import json
import copy
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, Sequence
from PIL import Image

from preprocess import preprocess, preprocess_multimodal
from constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from conversation import conversation_lib

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 data_args,
                 training_args):
        super(LazySupervisedDataset, self).__init__()
        if not os.path.exists(data_path):
            data_names = data_path.split('/')[-1].split('.')[0].split('-')
            diff_level = data_names[0].split('_')[-1]
            num_samples = data_names[1].split('_')[-1]
            os.system(f"python data_engine/euclid_format.py --diff_level {diff_level} --num_sample {num_samples}")
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.training_args = training_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0] or 'code' in sources[0]:
            processor = self.data_args.image_processor
            if 'image' in sources[0]:
                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                try:
                    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                except:
                    print(f"Failed to open image: {os.path.join(image_folder, image_file)}")
                    image = Image.new('RGB', (256, 256), (255, 255, 255))
            if 'code' in sources[0]:
                from image_engine import generate_image  # Import here to avoid circular imports
                image = generate_image(sources[0]['code'], sources[0]['letter_map'], resolution=256)
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i] or 'code' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i] or 'code' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        if self.training_args.seperate_log:
            diff_level = self.list_data_dict[i]['image'].split('/')[-2].split('-')[0][-1]
            data_dict['diff_level'] = diff_level
        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        if 'diff_level' in instances[0]:
            diff_levels = [instance['diff_level'] for instance in instances]
            batch['diff_levels'] = diff_levels

        return batch

def make_supervised_data_module(tokenizer: PreTrainedTokenizer,
                                data_args,
                                training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                training_args=training_args)
    if data_args.test_data_path is None:
        test_path = data_args.data_path.split('.json')[0] + '-test.json'
    else:
        test_path = data_args.test_data_path
    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=test_path,
                                data_args=data_args,
                                training_args=training_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
