import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import TokenTemplate, chat_template, now, unpad
from recurrent.impls.multiprocess_image_renderer import MPTextRenderer
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel('INFO')


@dataclass
class MemoryConfig(RConfig):
    context_key: str
    max_prompt_length: int  #
    chunk_size: int  # size of each context chunk in number of tokens3
    max_memorization_length: int  # max number of tokens to memorize
    # max_input_length = max_prompt_length + chunk_size + max_memorization_length + template_length
    max_chunks: int  # max number of chunks to process
    max_final_response_length: int
    # max_output_length = max_final_response_length if final else max_memorization_length

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + self.max_memorization_length
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
            raise ValueError('MemoryDataset only support center truncation')
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

<memory>
{memory}
</memory>

The article chunk is denoted by the attached image.
You should first try to extract the content from the image, and then update the memory with the crucial information in it that helps to answer the problem.

Updated memory:
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

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
        max_image_tokens = int(processor.image_processor.max_pixels / (28 * 28))
        self.max_input_length = self.config.max_raw_input_length + self.token_message_template_length + max_image_tokens
        logger.info(f'\n[RECURRENT] max_input_length: {self.config.max_raw_input_length}(raw) '
              f'+ {self.token_message_template_length}(message_template) + {max_image_tokens}(max_image_tokens) = {self.max_input_length}\n')
        self.NO_MEMORY_TOKENS = self.tokenizer.encode("No previous memory", add_special_tokens=False)

        assert 'vl' in self.tokenizer.name_or_path.lower(), "Tokenizer must be a Qwen-VL tokenizer"
        assert self.processor is not None, "Processor is required for Qwen-VL model"

        self.image_renderer = MPTextRenderer(font_size=16, img_width=360)

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
    
    @override
    def action(self) -> Tuple[List[list], dict]: # <-- Changed return type
        active_mask = self.ctx_length > self.step * self.config.chunk_size
        self.active_mask = active_mask
        gen_batch = self.gen_batch
        
        self.messages = [] # This will be a list of chat dicts
        
        if active_mask.sum().item() == 0:
            images = None
            self.is_final = True
            
            # --- Create text-only chat for final turn ---
            prompts = gen_batch.non_tensor_batch['prompt_ids']
            for i in range(self.bsz):
                prompt_text = self.tokenizer.decode(prompts[i], skip_special_tokens=True)
                memory_tokens = self.memory[i] if self.memory[i] is not None else self.NO_MEMORY_TOKENS
                memory_text = self.tokenizer.decode(memory_tokens, skip_special_tokens=True)
                
                text_content = self.template_final.format(
                    prompt=prompt_text,
                    memory=memory_text
                )
                self.messages.append(text_content)
            
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
            all_chunk_texts = [self.tokenizer.decode(c[c != self.tokenizer.pad_token_id]) for c in chunk_i]
            all_chunk_texts = [text.replace('\n\n', '\n').replace('Document ', '\n\nDocument ').strip() for text in all_chunk_texts]
            unique_chunk_texts = list(dict.fromkeys(all_chunk_texts))
            logger.info(f'start processing {len(all_chunk_texts)} chunk images (unique {len(unique_chunk_texts)})')
            all_chunk_images, _ = self.image_renderer.render(all_chunk_texts)
            logger.info(f'finished processing {len(all_chunk_texts)} chunk images (unique {len(unique_chunk_texts)})')

            # with open('tmp/all_chunk_texts_0.txt', 'w') as f:
            #     f.write(all_chunk_texts[0])
            # all_chunk_images[0].save('tmp/all_chunk_images_0.png')
            # assert False

            for prompt, memory, chunk_image in zip(prompt_i, memory_i, all_chunk_images):
                # Decode tokens back to strings for processor
                prompt_text = self.tokenizer.decode(prompt, skip_special_tokens=True)
                memory_tokens = memory if memory is not None else self.NO_MEMORY_TOKENS
                memory_text = self.tokenizer.decode(memory_tokens, skip_special_tokens=True)

                text_content = self.template.format(
                    prompt=prompt_text,
                    memory=memory_text
                )
                
                self.messages.append(text_content)
                images.append(chunk_image)
                
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
            self.memory[self.active_mask] = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
        self.log_step(gen_output)
        self.step += 1
        return gen_output
    
    @override
    def done(self):
        return self.is_final
    
    @override
    def end(self):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.memory
        del self.messages
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
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
