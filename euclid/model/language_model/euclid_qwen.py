from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# Assuming you have a EuclidMetaModel and EuclidMetaForCausalLM defined elsewhere
from ..euclid_arch import EuclidMetaModel, EuclidMetaForCausalLM

class EuclidQwen2Config(Qwen2Config):
    model_type = "euclid_qwen2"

class EuclidQwen2Model(EuclidMetaModel, Qwen2Model):
    config_class = EuclidQwen2Config

    def __init__(self, config: Qwen2Config):
        super(EuclidQwen2Model, self).__init__(config)

class EuclidQwen2ForCausalLM(Qwen2ForCausalLM, EuclidMetaForCausalLM):
    config_class = EuclidQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = EuclidQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        diff_levels: Optional[str] = None,
        tasks: Optional[str] = None,
        stages: Optional[str] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self.model.dtype)

        if past_key_values is not None:
            if isinstance(past_key_values, list) and len(past_key_values) > 0:
                if not isinstance(past_key_values[0][0], list):
                    past_key_values = [
                        [tensor.to(dtype=self.model.dtype) for tensor in layer]
                        for layer in past_key_values
                    ]
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        if labels is not None and tasks is not None and stages is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())
            non_zero_mask = (loss != 0)
            sample_losses = (loss * non_zero_mask).sum(dim=1) / non_zero_mask.sum(dim=1)

            return outputs, sample_losses
        else:
            return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        ## hardcode: remove the eos token and the last token (which is the answer token in this case), this is only for evaluation during training, for normal evaluation, we will just directly take 'inputs' as input
        # if inputs is None:
        #     # kwargs['input_ids'] = kwargs['input_ids'][:, :-kwargs['input_ids'].squeeze().tolist()[::-1].index(198)]
        #     # kwargs['attention_mask'] = kwargs['attention_mask'][:,:kwargs['input_ids'].shape[1]]
        #     new_input_ids = []
        #     new_attention_mask = []
        #     max_length = 0
        #     pad_token_id = 151643
        #     for input_id, attention_mask in zip(kwargs['input_ids'], kwargs['attention_mask']):
        #         input_id = input_id[:-input_id.squeeze().tolist()[::-1].index(198)]
        #         attention_mask = attention_mask[:len(input_id)]
        #         new_input_ids.append(input_id)
        #         new_attention_mask.append(attention_mask)
        #         max_length = max(max_length, len(input_id))
        #     padded_input_ids = []
        #     padded_attention_mask = []

        #     for input_id, attention_mask in zip(new_input_ids, new_attention_mask):
        #         pad_length = max_length - len(input_id)
                
        #         padded_input_id = torch.full((pad_length,), pad_token_id, dtype=input_id.dtype, device=input_id.device)
        #         padded_input_id = torch.cat([padded_input_id, input_id])
                
        #         padded_mask = torch.zeros(pad_length, dtype=attention_mask.dtype, device=attention_mask.device)
        #         padded_mask = torch.cat([padded_mask, attention_mask])
                
        #         padded_input_ids.append(padded_input_id)
        #         padded_attention_mask.append(padded_mask)
        #     kwargs['input_ids'] = torch.stack(padded_input_ids)
        #     kwargs['attention_mask'] = torch.stack(padded_attention_mask)
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        generation = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        return generation

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("euclid_qwen2", EuclidQwen2Config)
AutoModelForCausalLM.register(EuclidQwen2Config, EuclidQwen2ForCausalLM)