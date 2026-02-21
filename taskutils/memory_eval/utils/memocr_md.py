from .aio import get_async_client
from utils import extract_solution
from .envs import URL, API_KEY, RECURRENT_CHUNK_SIZE, RECURRENT_MAX_NEW, RECURRENT_MAX_CONTEXT_LEN
import os
import asyncio
import time
from recurrent.impls.format_utils import make_gap_question_from_html

min_image_pixels = int(os.getenv("MIN_PIXELS_28_28", 8)) * 28 * 28
max_image_pixels = int(os.getenv("MAX_PIXELS_28_28", 512)) * 28 * 28

print(f'Using MAX_PIXELS_28_28: {os.getenv("MAX_PIXELS_28_28", 512)}')
print(f'Using MIN_PIXELS_28_28: {os.getenv("MIN_PIXELS_28_28", 8)}')

if min_image_pixels > max_image_pixels:
    min_image_pixels = int(max_image_pixels/2) # shrink the min_image_pixels to the half of the max_image_pixels
print(f'Actual min_image_pixels: {min_image_pixels}')
print(f'Actual max_image_pixels: {max_image_pixels}')

from recurrent.impls.call_md_renderer import test_single_request
from PIL import Image
import io
import base64

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
E.g., the heading, bold, italic, code block, table, etc.
No need to wrap your markdown draft with ```markdown or ```, just write the markdown draft directly.
The draft memory, in markdown format:
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem, an article, and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

Your answer:
"""

TEMPLATE_FINAL_BOXED_NO_RENDERING = """You are presented with a problem, an article, and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<memory>
{memory}
</memory>

<problem> 
{prompt}
</problem>

Your answer:
"""

NO_MEMORY = "No previous memory"


def _record_trajectory(trajectory, kind, prompt, response):
    trajectory.append({
        "step": len(trajectory),
        "kind": kind,
        "prompt": prompt,
        "response": response,
    })


EMPHASIZE_H1 = os.getenv("EMPHASIZE_H1", "0")
EMPHASIZE_H1_v2 = os.getenv("EMPHASIZE_H1_v2", "0")
DISABLE_EMPHASIZE = os.getenv("DISABLE_EMPHASIZE", "0")
ORACLE_CRUCIAL = os.getenv("ORACLE_CRUCIAL", "0")
ORACLE_DETAILED = os.getenv("ORACLE_DETAILED", "0")
USE_GAP_FILLING = os.getenv("USE_GAP_FILLING", "0")
DISABLE_RENDERING = os.getenv("DISABLE_RENDERING", "0")
BOTH_TEXT_AND_IMAGE_RENDERING = os.getenv("BOTH_TEXT_AND_IMAGE_RENDERING", "0")
print(f"EMPHASIZE_H1: {EMPHASIZE_H1}")
print(f"EMPHASIZE_H1_v2: {EMPHASIZE_H1_v2}")
print(f"DISABLE_EMPHASIZE: {DISABLE_EMPHASIZE}")
print(f"ORACLE_CRUCIAL: {ORACLE_CRUCIAL}")
print(f"ORACLE_DETAILED: {ORACLE_DETAILED}")
print(f"USE_GAP_FILLING: {USE_GAP_FILLING}")
print(f"DISABLE_RENDERING: {DISABLE_RENDERING}")
print(f"BOTH_TEXT_AND_IMAGE_RENDERING: {BOTH_TEXT_AND_IMAGE_RENDERING}")
def markdown_to_image(markdown_content: str) -> Image.Image:
    if EMPHASIZE_H1_v2 == "1":
        # remove the lines that are not h1 headings
        lines = markdown_content.split("\n")
        new_lines = []
        for line in lines:
            if '##' in line or '###' in line:
                continue
            if "# " in line: # only keep lines that contain #
                new_lines.append(line)
        markdown_content = "\n".join(new_lines)
    elif EMPHASIZE_H1 == "1":
        # only emphasize the h1 heading
        markdown_content = markdown_content.replace("## ", "").replace("### ", "")
    elif DISABLE_EMPHASIZE == "1":
        # do not emphasize any headings
        markdown_content = markdown_content.replace("## ", "").replace("### ", "").replace("# ", "")

    image_bytes, _, _, success = test_single_request(markdown_content)
    try:
        assert success, f"Failed to generate image from markdown content: {markdown_content}"
        image = Image.open(io.BytesIO(image_bytes))
        image.load()  # Force load data into memory so file pointer isn't needed
        return image
    except Exception as e:
        raise Exception(f"Failed to load image from bytes: {e}")


def clip_long_string(string, max_length=2000):
    """Clip long string to a maximum length."""
    # assert max_length > 50, "max_length must be greater than 50"
    if not len(string) > max_length:
        return string
    target_len = max_length - len('\n\n...(truncated)\n\n')
    return string[:target_len//2] + '\n\n...(truncated)\n\n' + string[-target_len//2:]

def resize_maintain_aspect(image: Image.Image, max_pixels: int) -> Image.Image:
    w, h = image.size
    current_pixels = w * h
    # 如果当前像素小于设定的最大值，直接返回原图
    if current_pixels <= max_pixels:
        return image
    
    aspect = w / h
    # 计算新的尺寸，保持长宽比
    new_h = int((max_pixels / aspect) ** 0.5)
    new_w = int(new_h * aspect)
    
    # 使用 LANCZOS 算法进行高质量缩放
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


async def async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95, stop=None):
    idx = item["_id"]
    context = item["context"].strip()
    if 'prompt' in item:
        prompt = item['prompt'].strip()
    else:
        prompt = item['input'].strip()
    question = item['input'].strip()
    session = await get_async_client()
    max_len = RECURRENT_MAX_CONTEXT_LEN
    input_ids = tokenizer.encode(context)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
    memory = NO_MEMORY
    start_time = time.time()
    trajectory = []

    # no need to change the original logic for memory update
    for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):
        article = tokenizer.decode(input_ids[i:i+RECURRENT_CHUNK_SIZE])
        msg = TEMPLATE.format(prompt=prompt, article=article, memory=memory)
        if idx == 0:
            print("user:")
            print(clip_long_string(msg))
        
        # Retry logic: attempt up to 3 times
        max_retries = 3
        for retry in range(max_retries):
            try:
                async with session.post(
                    url=URL + "/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json=dict(model=model,
                        messages=[{"role": "user", "content": msg}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=RECURRENT_MAX_NEW,
                    )
                ) as resp:
                    status = resp.status
                    if status!= 200:
                        print(f"{status=}, {model=}")
                        if retry < max_retries - 1:
                            print(f"Retrying... ({retry + 1}/{max_retries})")
                            # Add exponential backoff delay
                            await asyncio.sleep(2 ** retry)
                            continue
                        return '', {}
                    data = await resp.json()
                    memory, _ = extract_solution(data['choices'][0]['message']['content'])
                    if idx == 0:
                        print(f"[Recurrent] Step {i//RECURRENT_CHUNK_SIZE+1}:")
                        print(clip_long_string(memory))
                        print("-" * 100)
                    _record_trajectory(trajectory, "chunk", msg, data['choices'][0]['message']['content'])
                    break  # Success, exit retry loop
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error occurred, retrying... ({retry + 1}/{max_retries})")
                    import traceback
                    traceback.print_exc()
                    # Add exponential backoff delay
                    delay = 2 ** retry
                    print(f"Waiting {delay} seconds before retry...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"Failed after {max_retries} attempts")
                    import traceback
                    traceback.print_exc()
                    return '', {}

    latest_memory_text = memory
    gt_answer = item["answers"][0] if item.get("answers") else None
    if ORACLE_CRUCIAL == "1" and gt_answer:
        # Prepend ground truth prominently for crucial oracle runs
        latest_memory_text = f"# Ground Truth Answer {gt_answer}\n\n{latest_memory_text}"
    elif ORACLE_DETAILED == "1" and gt_answer:
        # Append ground truth for detailed oracle runs
        latest_memory_text = f"{latest_memory_text}\n\nGround Truth Answer: {gt_answer}"
    if USE_GAP_FILLING == "1":
        gap_qa = make_gap_question_from_html(latest_memory_text, is_md_input=True)
        question = gap_qa["question"]
        answer = gap_qa["answer"]
        item['answer'] = answer
        item['gap_fill_question'] = question
    latest_memory_image = markdown_to_image(latest_memory_text)
    if DISABLE_RENDERING == "1" or BOTH_TEXT_AND_IMAGE_RENDERING == "1":
        latest_memory_text = latest_memory_text.replace("## ", "").replace("### ", "").replace("# ", "")
        latest_memory_text = latest_memory_text.replace("```markdown", "").replace("```", "")
        max_memory_tokens = int(os.getenv("MAX_PIXELS_28_28", 512))
        latest_memory_tokens = tokenizer.encode(latest_memory_text)
        truncated_latest_memory_tokens = latest_memory_tokens[:max_memory_tokens]
        truncated_latest_memory_text = tokenizer.decode(truncated_latest_memory_tokens)
        msg = TEMPLATE_FINAL_BOXED_NO_RENDERING.format(prompt=question, memory=truncated_latest_memory_text)
    else:
        msg = TEMPLATE_FINAL_BOXED.format(prompt=question)
    if idx == 0:
        print("user:")
        print(clip_long_string(msg))
    
    # Close the old session before image requests
    await session.close()
    
    retry_times = 5  # Increase retry times for image requests
    last_exception = None
    for attempt in range(retry_times):
        # Create a fresh session for each retry to avoid connection issues
        import aiohttp
        image_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=300,  # 5 minutes total timeout for image requests
                connect=60,  # 1 minute connect timeout
                sock_read=120  # 2 minutes read timeout
            )
        )
        try:
            content = []
            if latest_memory_image is not None and DISABLE_RENDERING == "0":
                processed_image = resize_maintain_aspect(latest_memory_image, max_image_pixels)
                content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": image_to_base64(processed_image),
                        "detail": "auto"
                    },
                })
            else:
                print("[Warning] No memory image generated")
            content.append({"type": "text", "text": msg})
            
            async with image_session.post(
                url=URL + "/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=dict(model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=RECURRENT_MAX_NEW,
                    max_pixels=max_image_pixels,
                    min_pixels=min_image_pixels,
                )
            ) as resp:
                status = resp.status
                if status != 200:
                    print(f"{status=}, {model=}")
                    if attempt < retry_times - 1:
                        print(f"Non-200 status, retrying... ({attempt + 1}/{retry_times})")
                        delay = min(2 ** attempt, 30)  # Cap at 30 seconds
                        print(f"Waiting {delay} seconds before retry...")
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()
                    
                data = await resp.json()
                if idx == 0:
                    print("[Recurrent] Final:")
                    print("assistant:")
                    print(clip_long_string(data['choices'][0]['message']['content']))
                    print("-" * 100)
                _record_trajectory(trajectory, "final", msg, data['choices'][0]['message']['content'])
                trajectory.append({
                    "step": len(trajectory),
                    "kind": "last_memory",
                    "prompt": '',
                    "response": latest_memory_text,
                })
                total_time = time.time() - start_time
                performance_metrics = {
                    "total_time": total_time,
                    "trajectory": trajectory,
                }
                if USE_GAP_FILLING == "1":
                    performance_metrics['trajectory'].append({
                        "step": -1,
                        "kind": "input_gap_fill",
                        "question": question,
                        "answer": answer,
                        "response": data['choices'][0]['message']['content'],
                    })
                # assert '$LANG' not in data['choices'][0]['message']['content']
                return data['choices'][0]['message']['content'], performance_metrics
                
        except KeyboardInterrupt as e:
            await image_session.close()
            raise e
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exception = e
            import traceback
            print(f"[Retry {attempt+1}/{retry_times}] Network error in image request:")
            traceback.print_exc()
            
            if attempt + 1 < retry_times:
                # Exponential backoff with cap
                delay = min(2 ** attempt, 30)
                print(f"Waiting {delay} seconds before retry...")
                await asyncio.sleep(delay)
            else:
                print(f"Failed after {retry_times} attempts")
                await image_session.close()
                return '', {}
        except Exception as e:
            last_exception = e
            import traceback
            print(f"[Retry {attempt+1}/{retry_times}] Unexpected exception:")
            traceback.print_exc()
            
            if attempt + 1 < retry_times:
                delay = min(2 ** attempt, 30)
                print(f"Waiting {delay} seconds before retry...")
                await asyncio.sleep(delay)
            else:
                await image_session.close()
                return '', {}
        finally:
            # Always close the session after each attempt
            if not image_session.closed:
                await image_session.close()
    
    return '', {}