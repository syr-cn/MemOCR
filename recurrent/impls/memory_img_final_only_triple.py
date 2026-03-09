import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from uuid import uuid4
import os
import markdown
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import TokenTemplate, chat_template, now, unpad
# from recurrent.impls.multiprocess_image_renderer import MPTextRenderer
# from recurrent.impls.multiprocess_image_renderer_md import MPMarkdownRenderer
from recurrent.impls.call_md_renderer import batch_generate_images
from recurrent.impls.format_utils import make_gap_question_from_html
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel('INFO')


@dataclass
class MemoryConfig(RConfig):
    context_key: str
    max_prompt_length: int  #
    chunk_size: int  # size of each context chunk in number of tokens
    max_memorization_length: int  # max number of tokens to memorize
    # max_input_length = max_prompt_length + chunk_size + max_memorization_length + template_length
    max_chunks: int  # max number of chunks to process
    max_final_response_length: int
    # max_output_length = max_final_response_length if final else max_memorization_length

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + self.chunk_size + self.max_memorization_length

    # use property incase we want to adapt soft punishment to length.
    @property
    def gen_max_tokens_memorization(self):
        return self.max_memorization_length

    @property
    def gen_max_tokens_final_response(self):
        return self.max_final_response_length

    @property
    def gen_pad_to(self):
        return max(self.max_prompt_length, self.max_final_response_length)

class MemoryDataset(RDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(
        self,
        recurrent_config: MemoryConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if data_config.truncation != 'middle':
            pass
            # raise ValueError('MemoryDataset only support center truncation')
        data_config.max_prompt_length=recurrent_config.max_chunks * recurrent_config.chunk_size
        self.context_key = recurrent_config.context_key
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )

    @override
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)
        context = row_dict.pop(self.context_key)

        model_inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)

        context_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        context_ids, attention_mask = verl_F.postprocess_data(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id, # pyright: ignore
            left_pad=False,
            truncation=self.truncation,
        )

        row_dict["context_ids"] = context_ids[0]
        lengths = attention_mask.sum(dim=-1)
        row_dict["context_length"] = lengths[0]
        row_dict["prompt_ids"] = self.tokenizer.encode(
            chat[0]["content"], add_special_tokens=False
        )
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
        return ["context_ids", "context_length"], ["prompt_ids"]

TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<article>
{article}
</article>

<memory>
{memory}
</memory>

You are given a problem, an article, and a previous memory.
You should draft the memory in markdown format with the crucial information in it that helps to answer the problem.
In your markdown draft, you may use different headings to arrange the font sizes and styles of the information.
E.g., more important information should be emphasized and more visible (larger font size, bolder, colored red, etc.), in case the rendered image can be clearly read.
No need to wrap your markdown draft with ```markdown or ```, just write the markdown draft directly.
The draft memory, in markdown format:
"""
TEMPLATE_FINAL_BOXED = """You are presented with a problem, an article, and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

Your answer:
"""


class MemoryAgent(RAgent):
    def __init__(self, tokenizer:PreTrainedTokenizer, config: MemoryConfig, processor: ProcessorMixin):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor # This should now be the Qwen-VL Processor
        
        self.template = TEMPLATE
        self.template_final = TEMPLATE_FINAL_BOXED

        self.token_message_template_length = len(self.tokenizer.encode(self.template, add_special_tokens=False))

        # --- Remove TokenTemplate and max_input_length calculation ---
        self.max_image_pixels = self.processor.image_processor.max_pixels
        self.min_image_pixels = self.processor.image_processor.min_pixels
        max_image_tokens = int(self.max_image_pixels / (28 * 28))
        min_image_tokens = int(self.min_image_pixels / (28 * 28))
        # assert max_image_tokens <= 1024, f'max_image_tokens: {max_image_tokens} is too large'
        if max_image_tokens > 2048:
            logger.warning(f'max_image_tokens: {max_image_tokens} is too large, will be set to 2048')
            max_image_tokens = 2048
        self.max_input_length = self.config.max_raw_input_length + self.token_message_template_length + max_image_tokens * 2
        logger.info(f'\n[RECURRENT] max_input_length: {self.config.max_raw_input_length}(raw) '
              f'+ {self.token_message_template_length}(message_template) + {max_image_tokens * 2}(max_image_tokens * 2) = {self.max_input_length}\n')
        self.NO_MEMORY_TEXT = "No previous memory"
        self.NO_MEMORY_TOKENS = self.tokenizer.encode(self.NO_MEMORY_TEXT, add_special_tokens=False)

        assert 'vl' in self.tokenizer.name_or_path.lower(), "Tokenizer must be a Qwen-VL tokenizer"
        assert self.processor is not None, "Processor is required for Qwen-VL model"

    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.step = 0
        self.final_mask_list = [] 
        self.sample_index_list = [] 
        
        self.ctx_length = gen_batch.batch['context_length'] 
        self.bsz = len(self.ctx_length)
        self.memory = np.empty(self.bsz, dtype=object)
        self.is_final = False
        print(f'MemoryAgent.start() with bsz: {self.bsz}, step: {self.step}')
    
    @override
    def action(self) -> Tuple[List[list], dict]: # <-- Changed return type
        active_mask = self.ctx_length > self.step * self.config.chunk_size
        self.active_mask = active_mask
        gen_batch = self.gen_batch
        
        self.messages = [] # This will be a list of chat dicts
        
        if active_mask.sum().item() == 0:
            self.is_final = True
            
            # --- Create text-only chat for final turn ---
            prompts = gen_batch.non_tensor_batch['prompt_ids']
            all_memory_text = []
            for i in range(self.bsz):
                prompt_text = self.tokenizer.decode(prompts[i], skip_special_tokens=True)
                memory_tokens = self.memory[i] if self.memory[i] is not None else self.NO_MEMORY_TOKENS
                memory_text = self.tokenizer.decode(memory_tokens, skip_special_tokens=True)
                all_memory_text.append(memory_text)

                text_content = self.template_final.format(
                    prompt=prompt_text,
                )
                self.messages.append(text_content)
            
            num_workers = int(os.cpu_count() * 0.8)
            start_time = time.time()
            logger.info(f'start processing {len(all_memory_text)} memory images')
            all_memory_images = batch_generate_images(all_memory_text, max_workers=num_workers)
            logger.info(f'finished processing {len(all_memory_text)} memory images. Time taken: {time.time() - start_time} seconds')
            images = all_memory_images

            sample_index = torch.arange(self.bsz, dtype=torch.int)
            final_mask = torch.full(sample_index.shape, True, dtype=torch.bool)
            self.meta_info = {'input_pad_to': self.max_input_length, # Note: max_input_length is no longer defined
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_final_response, # Use final response length
                          'n': 1 
                        }}
            logger.info(f'FINAL TURN: MemoryAgent.action() done')
        else:
            images = []
            # --- Create multimodal chat for chunk turn ---
            prompt_i = gen_batch.non_tensor_batch['prompt_ids'][active_mask]
            chunk_i = gen_batch.batch['context_ids'][active_mask, self.config.chunk_size * self.step: self.config.chunk_size * (self.step+1)] 
            memory_i = self.memory[active_mask]

            logger.info(f'Skipping memory image generation in FINAL_ONLY mode')
            # all_memory_texts = [self.tokenizer.decode(c[c != self.tokenizer.pad_token_id]) if c is not None else self.NO_MEMORY_TEXT for c in memory_i]
            # unique_memory_texts = list(dict.fromkeys(all_memory_texts))
            # start_time = time.time()
            # logger.info(f'start processing {len(all_memory_texts)} memory images (unique {len(unique_memory_texts)})')
            # all_memory_images = batch_generate_images(all_memory_texts)
            # logger.info(f'finished processing {len(all_memory_texts)} memory images (unique {len(unique_memory_texts)}). Time taken: {time.time() - start_time} seconds')

            # with open('tmp/all_memory_texts_0.txt', 'w') as f:
            #     f.write(all_memory_texts[0])
            # all_memory_images[0].save('tmp/all_memory_images_0.png')
            # assert False

            for prompt, memory_text, chunk_text in zip(prompt_i, memory_i, chunk_i):
                self.messages.append(self.template.format(
                    prompt=self.tokenizer.decode(prompt[:self.config.max_prompt_length], skip_special_tokens=True),
                    article=self.tokenizer.decode(chunk_text, skip_special_tokens=True),
                    memory=self.tokenizer.decode(memory_text if memory_text is not None else self.NO_MEMORY_TOKENS, skip_special_tokens=True)
                ))
                
            sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask] 
            final_mask = torch.full(sample_index.shape, False, dtype=torch.bool) 
            self.meta_info = {'input_pad_to': self.max_input_length, # Note: max_input_length is no longer defined
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 
                        }}
            logger.info(f'MemoryAgent.action() done')
        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        return self.messages, images, self.meta_info

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        if not self.is_final:
            assert len(gen_output.batch['responses']) == self.active_mask.sum().item(), f'len(gen_output.batch["responses"]): {len(gen_output.batch["responses"])} != self.active_mask.sum().item(): {self.active_mask.sum().item()}'
            self.memory[self.active_mask] = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
        self.log_step(gen_output)
        self.step += 1
        return gen_output
    
    @override
    def done(self):
        return self.is_final
    
    @override
    def end(self, triple_answer_turn=False):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.messages
        if triple_answer_turn:

            all_true_mask = torch.full(torch.Size([self.bsz]), True, dtype=torch.bool)
            all_false_mask = torch.full(torch.Size([self.bsz]), False, dtype=torch.bool)
            final_mask = torch.cat(self.final_mask_list + [all_true_mask, all_true_mask]) # repeat the last final mask because we're doing triple answer turn
            assert self.final_mask_list[-1].shape == all_true_mask.shape, f'self.final_mask_list[-1] should be True x BSZ. final_mask_list[-1].shape: {self.final_mask_list[-1].shape} != all_true_mask.shape: {all_true_mask.shape}'
            assert sum(self.final_mask_list[-1].long()) == sum(all_true_mask.long()), f'self.final_mask_list[-1] should be True x BSZ. sum(final_mask_list[-1].long()): {sum(self.final_mask_list[-1].long())} != sum(all_true_mask.long()): {sum(all_true_mask.long())}'
            vanilla_qa_mask = torch.cat(self.final_mask_list + [all_false_mask, all_false_mask]) # .... + True * BSZ + False * BSZ + True * BSZ
            subsampled_qa_mask = torch.cat(self.final_mask_list[:-1] + [all_false_mask, all_true_mask, all_false_mask]) # .... + False * BSZ + True * BSZ + False * BSZ
            gap_fill_mask = torch.cat(self.final_mask_list[:-1] + [all_false_mask, all_false_mask, all_true_mask]) # .... + False * BSZ + False * BSZ + True * BSZ
            sample_index = torch.cat(self.sample_index_list + [self.sample_index_list[-1], self.sample_index_list[-1]])

            all_memory_text = [self.tokenizer.decode(memory_tokens if memory_tokens is not None else self.NO_MEMORY_TOKENS, skip_special_tokens=True) for memory_tokens in self.memory]
            all_gap_fill_qas = [make_gap_question_from_html(memory_md_text, is_md_input=True) for memory_md_text in all_memory_text]
            all_gap_fill_questions = [qa["question"] for qa in all_gap_fill_qas]
            all_gap_fill_answers = [qa["answer"] for qa in all_gap_fill_qas]

            del self.final_mask_list
            del self.sample_index_list
            del self.memory
            return (final_mask, vanilla_qa_mask, subsampled_qa_mask, gap_fill_mask), sample_index, (all_gap_fill_questions, all_gap_fill_answers)
        else:
            final_mask = torch.cat(self.final_mask_list)
            del self.final_mask_list
            sample_index = torch.cat(self.sample_index_list)
            del self.sample_index_list
            del self.memory
            return final_mask, sample_index

    def log_step(self, gen_output):
        """Log multi-turn conversation details in a single consolidated function.
        """
        def clip_long_string(string, max_length=2000):
            """Clip long string to a maximum length."""
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]

        # Header with dynamic step number
        step = self.step if not self.is_final else "FINAL"
        logger.info(f"\n{'='*30}[RECURRENT] STEP {step} (active_mask: {self.active_mask.sum().item()}/{self.bsz}) {'='*30}")

        # Message and Response section
        if self.active_mask[0]:
            decoded_message = self.messages[0]
            rsp0 = gen_output.batch['responses'][0]
            decoded_response = self.tokenizer.decode(rsp0[rsp0!=self.tokenizer.pad_token_id])
            logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
        else:
            logger.info("MESSAGE and RESPONSE is empty since it is not active.")


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryAgent)
