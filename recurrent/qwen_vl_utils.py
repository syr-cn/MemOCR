from typing import List, Optional, Union

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.vision_utils import process_image, process_video
from verl.models.transformers.qwen2_vl import get_rope_index
import torch


def process_rlhf_inputs(
    messages,
    images_data,
    tokenizer,
    processor,
    max_prompt_length=1024,
    truncation="error",
) -> dict:

    output_dict = {}
    assert processor is not None

    raw_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    multi_modal_data = {}
    processed_images = None
    if images_data:
        processed_images = [process_image(image) for image in images_data]
        multi_modal_data["image"] = processed_images

    model_inputs = processor(
        text=raw_prompt, 
        images=processed_images, 
        videos=None, 
        return_tensors="pt"
    )

    input_ids = model_inputs.pop("input_ids")
    attention_mask = model_inputs.pop("attention_mask")

    model_inputs.pop("second_per_grid_ts", None)
    
    output_dict["multi_modal_data"] = multi_modal_data
    output_dict["multi_modal_inputs"] = dict(model_inputs)
    output_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

    input_ids, attention_mask = verl_F.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation=truncation,
    )

    assert processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__

    position_ids = []
    vision_position_ids = get_rope_index(
        processor,
        input_ids=input_ids[0],
        image_grid_thw=model_inputs.get("image_grid_thw"),
        video_grid_thw=model_inputs.get("video_grid_thw"),
        second_per_grid_ts=model_inputs.get("second_per_grid_ts"), # Already popped, but original code uses it
        attention_mask=attention_mask[0],
    )  # (3, seq_length)
    valid_mask = attention_mask[0].bool()
    text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
    position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)

    output_dict["input_ids"] = input_ids[0]       # (seq_len)
    output_dict["attention_mask"] = attention_mask[0] # (seq_len)
    output_dict["position_ids"] = position_ids[0]     # (seq_len) or (1, 4, seq_len)

    raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
    if len(raw_prompt_ids) > max_prompt_length:
        if truncation == "left":
            raw_prompt_ids = raw_prompt_ids[-max_prompt_length :]
        elif truncation == "right":
            raw_prompt_ids = raw_prompt_ids[: max_prompt_length]
        elif truncation == "middle":
            left_half = max_prompt_length // 2
            right_half = max_prompt_length - left_half
            raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
        elif truncation == "error":
            raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {max_prompt_length}.")
    output_dict["raw_prompt_ids"] = raw_prompt_ids

    return output_dict
