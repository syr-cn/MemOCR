# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Type
import numpy as np
import torch
from codetiming import Timer
from copy import deepcopy
from collections import defaultdict
from PIL import Image
from verl import DataProto
import concurrent.futures

from .interface import RAgent, RConfig
from .utils import (chat_template, create_attention_mask, create_position_ids,
                    graceful_padding, indexing_proto,
                    pad_tensor_list_to_length)
import verl.utils.torch_functional as verl_F
from .qwen_vl_utils import process_rlhf_inputs

def batch_subsample_images(images: List[Image.Image], ratio: float = 0.5, max_workers: int = 10) -> List[Image.Image]:
    def subsample_image(img_index, image: Image.Image, ratio: float = 0.5) -> Image.Image:
        if ratio >= 1.0 or ratio <= 0 or image is None:
            return img_index, image
        orig_w, orig_h = image.size
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)

        small_img = image.resize((new_w, new_h), Image.BILINEAR)
        restored_img = small_img.resize((orig_w, orig_h), Image.NEAREST)
        return img_index, restored_img

    results = [None for _ in range(len(images))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(subsample_image, i, img, ratio) 
            for i, img in enumerate(images)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            idx, img = future.result()
            results[idx] = img
    return results

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of data."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return tensors, non_tensors


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timing_raw.get(name, 0.) + timer.last


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: RConfig,
        agent_cls: Type[RAgent],
        processor = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.world_size = actor_rollout_wg.world_size
        if processor is not None:
            self.agent = agent_cls(tokenizer, config, processor)
        else:
            self.agent = agent_cls(tokenizer, config)
        self.chat_template = chat_template(tokenizer)
        self.PADDING_WORD_TOKENS = tokenizer.encode(self.chat_template.format(message="Hello."), add_special_tokens=False)


    from functools import lru_cache
    @lru_cache(maxsize=3)
    def get_paddings(self, shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return padding_token_ids, padding_attention_masks, padding_position_ids
        """
        pad_shape = shape[1:]
        padding_word_ids = self.PADDING_WORD_TOKENS
        padding_token_ids = torch.full(pad_shape, fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        padding_attention_masks = torch.zeros(pad_shape, dtype=torch.long)
        padding_position_ids = torch.zeros(pad_shape, dtype=torch.long)
        # token_ids <pad> <pad> <pad> <tok> <tok> <tok>
        # attn_mask 0     0     0     1     1     1
        # posit_ids 0     0     0     0     1     2
        padding_token_ids[-len(padding_word_ids):] = torch.tensor(padding_word_ids, dtype=torch.long)
        padding_attention_masks[-len(padding_word_ids):] = 1
        padding_position_ids[-len(padding_word_ids):] = torch.arange(0, len(padding_word_ids))
        return padding_token_ids, padding_attention_masks, padding_position_ids

    # @lru_cache(maxsize=3)
    def get_paddings_vl(self, model_inputs: DataProto) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return padding_token_ids, padding_attention_masks, padding_position_ids
        """
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        position_ids = model_inputs['position_ids']

        padding_token_ids = deepcopy(input_ids[0])
        padding_attention_masks = deepcopy(attention_mask[0])
        padding_position_ids = deepcopy(position_ids[0])
        return padding_token_ids, padding_attention_masks, padding_position_ids
    
    def generate_with_graceful_padding(self, input_ids: torch.Tensor,
                                    attention_masks: torch.Tensor,
                                    position_ids: torch.Tensor,
                                    meta_info: dict):

        """
        batch may not be divisible by wordsize.
        Use "Hello" as padding, insert padding data into batch so that data 
        """
        bsz = input_ids.shape[0]

        group_nums = self.world_size
        remainder = bsz % group_nums
        if remainder:
            # Example pattern for bsz=7, group_nums=3:
            # no_padding_mask: [1, 1, 1, 0, 1, 1, 0, 1, 1]
            # padding_index:   [0, 1, 2, -1, 3, 4, -1, 5, 6]
            padding_index, no_padding_mask = graceful_padding(bsz, group_nums)
            padding_token_ids, padding_attention_masks, padding_position_ids = self.get_paddings(input_ids.shape)
            def padding_by_index(tensor, padding, padding_index):
                if not len(padding.shape) == 2:
                    padding = padding.unsqueeze(0)
                # 2. prepare data for padding, concat padding to the end of batch
                tensor_for_indexing = torch.cat([tensor, padding], dim=0)
                # 3. index, -1 will select padding, else select the corresponding original data 
                return tensor_for_indexing[padding_index]
            
            input_ids = padding_by_index(input_ids, padding_token_ids, padding_index)
            attention_masks = padding_by_index(attention_masks, padding_attention_masks, padding_index)
            position_ids = padding_by_index(position_ids, padding_position_ids, padding_index)

        batch = DataProto.from_dict(tensors={
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_masks
        }, meta_info=meta_info)
        output_batch = self.actor_rollout_wg.generate_sequences(batch)
        if remainder:
            # 4. remove padding
            output_batch = indexing_proto(output_batch, no_padding_mask)
        return output_batch

    def run_llm_loop(self, gen_batch, timing_raw) -> Tuple[DataProto, torch.BoolTensor, torch.LongTensor]:
        """Run main LLM generation loop.
        genbatch: 'context_ids','context_length','prompt_ids'
        timing_raw: timing dict used in ray_trainer, note that we will accumulate the time cost in this loop, instead of override each time as in ray_trainer.
        see `_timer` implementation at the top of this file for more details.
        """
        active_num_list = [] # trace the active number of sample in each turn
        gen_output_list = [] # store I/O batch in each turn, used for policy optimization
        meta_info = gen_batch.meta_info #  do_sample, is_validate, eos/pad are stored in here.
        # pad_token_id = self.tokenizer.pad_token_id
        pad_token_id = -100
        self.agent.start(gen_batch, timing_raw)
        # Main generation loop, agent should indicate whether to stop
        while not self.agent.done():
            with _timer('mt_prepare', timing_raw):
                messages, meta_info_gen = self.agent.action()
                meta_info_gen.update(meta_info)
                # [len(x) for x in messages] == [len(x[x!=pad_token_id]) for x in input_ids]
                # torch.all(attention_masks.sum(-1) == torch.tensor([len(x[x!=pad_token_id]) for x in input_ids]))
                input_ids = pad_tensor_list_to_length(messages, 
                                                pad_token_id=pad_token_id,
                                                max_length=meta_info_gen['input_pad_to'], 
                                                left_pad=True)
                attention_masks = create_attention_mask(input_ids, pad_token_id=pad_token_id)
                position_ids = create_position_ids(attention_masks)
                active_num_list.append(len(messages))
                logger.info(f'padding done')
            with _timer('mt_gen', timing_raw):
                gen_output = self.generate_with_graceful_padding(input_ids, attention_masks, position_ids, meta_info_gen)
                logger.info('generation done')
            with _timer('mt_update', timing_raw):
                gen_output = self.agent.update(gen_output)
                gen_output_list.append(gen_output)
                logger.info('agent update done')
        final_mask, sample_index = self.agent.end()
        
        # OK, now we've got all we need in gen_output_list, and the final_mask indicates which one is final answer.
        assert len(sample_index) == sum(active_num_list)
        assert sum(final_mask) == len(gen_batch)
        logger.info(f"ACTIVE_TRAJ_NUM: {active_num_list}")
        return DataProto.concat(gen_output_list), final_mask, sample_index # pyright: ignore

    def _build_messages(self, prompts, images):
        messages = []
        assert isinstance(prompts[0], str)
        for prompt, image in zip(prompts, images):
            content = []
            if image is not None:
                content.append({
                    "type": "image", 
                    "image": image,
                    "min_pixels": self.agent.min_image_pixels,
                    "max_pixels": self.agent.max_image_pixels
                })
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])
        return messages

    def generate_with_graceful_padding_vl(self, model_inputs, non_tensor_batch, meta_info):
        bsz = model_inputs['input_ids'].shape[0]
        group_nums = self.world_size
        remainder = bsz % group_nums

        if remainder:
            padding_index, no_padding_mask = graceful_padding(bsz, group_nums)
            
            padding_token_ids, padding_attention_masks, padding_position_ids = self.get_paddings_vl(model_inputs)
            
            def padding_by_index(tensor, padding, p_index):
                if not len(padding.shape) == 2:
                    padding = padding.unsqueeze(0) # 1 x LEN_MAX
                if len(padding.shape) == 2 and len(tensor.shape) == 3: # BS x 3 x LEN_MAX position_ids
                    padding = padding.unsqueeze(1) # 1 x 1 x LEN_MAX
                    padding = padding.repeat(1, tensor.shape[1], 1) # BS x 3 x LEN_MAX
                tensor_for_indexing = torch.cat([tensor, padding], dim=0)
                return tensor_for_indexing[p_index]

            model_inputs['input_ids'] = padding_by_index(model_inputs['input_ids'], padding_token_ids, padding_index)
            model_inputs['attention_mask'] = padding_by_index(model_inputs['attention_mask'], padding_attention_masks, padding_index)
            model_inputs['position_ids'] = padding_by_index(model_inputs['position_ids'], padding_position_ids, padding_index)

            p_index_list = padding_index.tolist()

            def padding_list_by_index(data_list: List, padding_value, p_index: List[int]):
                list_for_indexing = list(data_list) + [padding_value]
                return [list_for_indexing[i] for i in p_index]

            non_tensor_batch['raw_prompt_ids'] = padding_list_by_index(non_tensor_batch['raw_prompt_ids'], non_tensor_batch['raw_prompt_ids'][0], p_index_list)
            if 'multi_modal_data' in non_tensor_batch:
                non_tensor_batch['multi_modal_data'] = padding_list_by_index(non_tensor_batch['multi_modal_data'], non_tensor_batch['multi_modal_data'][0], p_index_list)
            if 'multi_modal_inputs' in non_tensor_batch:
                non_tensor_batch['multi_modal_inputs'] = padding_list_by_index(non_tensor_batch['multi_modal_inputs'], non_tensor_batch['multi_modal_inputs'][0], p_index_list)

        batch = DataProto.from_dict(tensors=model_inputs, non_tensors=non_tensor_batch, meta_info=meta_info)
        output_batch = self.actor_rollout_wg.generate_sequences(batch)

        if remainder:
            output_batch = indexing_proto(output_batch, no_padding_mask)
        return output_batch

    def process_messages_vl(self, prompts, images, meta_info_gen):
        assert images is not None, "Images must be provided for VL models."
        if len(images) == 0:
            images_list = [None for _ in prompts]
        else:
            images_list = images
        messages = self._build_messages(prompts, images_list)
        max_token_length = meta_info_gen['input_pad_to']
        data_list = []
        for message, image in zip(messages, images_list):
            out = process_rlhf_inputs(
                messages=message,
                images_data=[image] if image is not None else None,
                tokenizer=self.tokenizer,
                processor=self.processor,
                max_prompt_length=max_token_length,
                truncation="error",
            )
            data_list.append(out)
        model_inputs, non_tensor_batch = collate_fn(data_list) # BS x LEN_MAX
        image_token_id = 151655
        image_token_counts = (model_inputs['input_ids'] == image_token_id).sum(-1) # BS

        if image_token_counts.sum().item() > 0:
            # 转换为 float 以计算精确的平均值
            counts_float = image_token_counts.float()
            
            stats = {
                "avg": counts_float.mean().item(),
                "max": counts_float.max().item(),
                "min": counts_float.min().item(),
                "median": counts_float.median().item()
            }
            
            logger.info(f"[Image Token Stats] Avg: {stats['avg']:.2f} | Max: {int(stats['max'])} | Min: {int(stats['min'])} | Median: {int(stats['median'])}")
        return model_inputs, non_tensor_batch

    def run_llm_loop_vl(self, gen_batch, timing_raw):
        assert self.processor is not None, "Processor required for VL models."
        active_num_list = []
        gen_output_list = []
        meta_info = gen_batch.meta_info
        self.agent.start(gen_batch, timing_raw)

        while not self.agent.done():
            with _timer("mt_prepare", timing_raw):
                text_messages, images, meta_info_gen = self.agent.action()
                meta_info_gen.update(meta_info)
                model_inputs, non_tensor_batch = self.process_messages_vl(text_messages, images, meta_info_gen)
                active_num_list.append(len(text_messages))
                logger.info("VL batch prepared")

            with _timer("mt_gen", timing_raw):
                gen_output = self.generate_with_graceful_padding_vl(model_inputs, non_tensor_batch, meta_info_gen)
                logger.info("VL generation done")

            with _timer("mt_update", timing_raw):
                gen_output = self.agent.update(gen_output)
                gen_output_list.append(gen_output)
                logger.info("Agent update done")

        final_mask, sample_index = self.agent.end()
        assert len(sample_index) == sum(active_num_list)
        assert sum(final_mask) == len(gen_batch)
        logger.info(f"ACTIVE_TRAJ_NUM: {active_num_list}")
        return DataProto.concat(gen_output_list), final_mask, sample_index

    def run_llm_loop_vl_triple_turn(self, gen_batch, timing_raw):
        assert self.processor is not None, "Processor required for VL models."
        active_num_list = []
        gen_output_list = []
        meta_info = gen_batch.meta_info
        self.agent.start(gen_batch, timing_raw)

        while not self.agent.done():
            with _timer("mt_prepare", timing_raw):
                text_messages, images, meta_info_gen = self.agent.action()
                meta_info_gen.update(meta_info)
                model_inputs, non_tensor_batch = self.process_messages_vl(text_messages, images, meta_info_gen)
                active_num_list.append(len(text_messages))
                logger.info("VL batch prepared")

            with _timer("mt_gen", timing_raw):
                gen_output = self.generate_with_graceful_padding_vl(model_inputs, non_tensor_batch, meta_info_gen)
                logger.info("VL generation done")

            with _timer("mt_update", timing_raw):
                gen_output = self.agent.update(gen_output)
                gen_output_list.append(gen_output)
                logger.info("Agent update done")
        
        latest_images = images # the images used in final response generation
        latest_text_messages = text_messages # the text messages used in final response generation
        (final_mask, vanilla_qa_mask, subsampled_qa_mask, gap_fill_mask), sample_index, (all_gap_fill_questions, all_gap_fill_answers) = self.agent.end(triple_answer_turn=True)

        # prepare second training objective: QA with subsampled images
        with _timer("mt_prepare_objective2_subsampled_images", timing_raw):
            subsampled_images = batch_subsample_images(latest_images, ratio=0.25, max_workers=int(os.cpu_count()*0.8))
            subsampled_model_inputs, subsampled_non_tensor_batch = self.process_messages_vl(latest_text_messages, subsampled_images, meta_info_gen)
            logger.info(f"Subsampled images prepared. {sum(1 for img in subsampled_images if img is not None)}/{len(subsampled_images)} subsampled images are not None.")

        with _timer("mt_gen_objective2_subsampled_images", timing_raw):
            vanilla_min_pixels, vanilla_max_pixels = self.agent.min_image_pixels, self.agent.max_image_pixels
            self.agent.min_image_pixels = int(vanilla_min_pixels * 0.25)
            self.agent.max_image_pixels = int(vanilla_max_pixels * 0.25)
            logger.info(f"Subsampled images generation with min_pixels: {self.agent.min_image_pixels} and max_pixels: {self.agent.max_image_pixels}")
            subsampled_gen_output = self.generate_with_graceful_padding_vl(subsampled_model_inputs, subsampled_non_tensor_batch, meta_info_gen)
            logger.info("Subsampled images generation done")
            self.agent.min_image_pixels = vanilla_min_pixels
            self.agent.max_image_pixels = vanilla_max_pixels
            logger.info(f"Reset min_pixels and max_pixels to original values: {vanilla_min_pixels} and {vanilla_max_pixels}")

        # prepare third training objective: gap-fill questions with full-res images
        with _timer("mt_prepare_objective3_gap_fill", timing_raw):
            gap_fill_model_inputs, gap_fill_non_tensor_batch = self.process_messages_vl(all_gap_fill_questions, latest_images, meta_info_gen)
            # gap_fill_non_tensor_batch['reward_model'] = np.array([{'ground_truth': [ans]} for ans in all_gap_fill_answers], dtype=object)
            logger.info(f"Gap fill questions prepared. {sum(1 for question in all_gap_fill_questions if question is not None)}/{len(all_gap_fill_questions)} gap fill questions are not None.")

        with _timer("mt_gen_objective3_gap_fill", timing_raw):
            gap_fill_gen_output = self.generate_with_graceful_padding_vl(gap_fill_model_inputs, gap_fill_non_tensor_batch, meta_info_gen)
            logger.info("Gap fill questions generation done")
        
        gen_output_list.append(subsampled_gen_output)
        gen_output_list.append(gap_fill_gen_output)

        assert len(sample_index) == sum(active_num_list) + 2 * len(gen_batch), f'sample_index should have sum(active_num_list) + 2 * len(gen_batch) values: {len(sample_index)} != {sum(active_num_list) + 2 * len(gen_batch)}'
        assert sum(final_mask.long()) == len(gen_batch) * 3, f'final_mask should have 3*BSZ True values: {sum(final_mask.long())} != 3*len(gen_batch): {3*len(gen_batch)}'
        assert sum(vanilla_qa_mask.long()) == len(gen_batch), f'vanilla_qa_mask should have BSZ True values: {sum(vanilla_qa_mask.long())} != len(gen_batch): {len(gen_batch)}'
        assert sum(subsampled_qa_mask.long()) == len(gen_batch), f'subsampled_qa_mask should have BSZ True values: {sum(subsampled_qa_mask.long())} != len(gen_batch): {len(gen_batch)}'
        assert sum(gap_fill_mask.long()) == len(gen_batch), f'gap_fill_mask should have BSZ True values: {sum(gap_fill_mask.long())} != len(gen_batch): {len(gen_batch)}'

        generated_batch = DataProto.concat(gen_output_list)
        assert len(generated_batch) == len(final_mask), f'len(generated_batch): {len(generated_batch)} != len(final_mask): {len(final_mask)}'
        logger.info(f"ACTIVE_TRAJ_NUM: {active_num_list}")
        return generated_batch, (final_mask, vanilla_qa_mask, subsampled_qa_mask, gap_fill_mask), sample_index, all_gap_fill_answers