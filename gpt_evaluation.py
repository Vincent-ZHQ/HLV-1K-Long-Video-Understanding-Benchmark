import os
import json
import cv2
import base64
import openai
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 文件夹路径
VIDEOS_FOLDER_PATH = '/HLV-1K/videos'
JSON_FOLDER_PATH = '/HLV-1K/data'
OUTPUT_FOLDER_PATH = '/HLV-1K/output'

api_version = ""
base_url = ""
ak = ""
model_name = ""

max_tokens = 50 
frame_num = 1
max_workers = 200

client = openai.AzureOpenAI(
    azure_endpoint=base_url,
    api_version=api_version,
    api_key=ak,
)

def load_video(video_path, fix_frame):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    if len(frame_idx) > fix_frame:
        sample_fps = fix_frame
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()  # (num_frames, H, W, 3)
    num_frames = spare_frames.shape[0]
    image_size = [(spare_frames.shape[1], spare_frames.shape[2])] * num_frames
    return spare_frames, num_frames, image_size

def resize_frame(frame, max_size=768):
    height, width = frame.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    return frame

def encode_frame(frame):
    frame = resize_frame(frame)
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def frames_to_base64(frames):
    return [encode_frame(frame) for frame in frames]

def testOpenaiChatCompletion(system_message, frames):
    retries = 5
    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages = [
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": system_message,
                    },
                    *map(lambda x: {"image": x, "resize": 768}, frames),
                        ],
                    },
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            print(f'QPM Limit ... Sleep 30s ...')
            time.sleep(30)
        except openai.OpenAIError as e:
            print(f'ERROR: | {type(e)} | {e}')

    print(f">>> Failed after {retries} retries ...")
    return f"Unsuccessful: Failed after multiple retries."


def process_response(response, qa_type):
    response = response.strip()
    if response == "I don't know.":
        return ''
    if qa_type == 'qa':
        if 'yes' in response.lower().split():
            return 'Yes'
        elif 'no' in response.lower().split():
            return 'No'
        else:
            return ''
    elif qa_type == 'mcqa':
        # 提取第一个字母并检查是否是有效选项
        first_letter = response.strip()[0].upper()
        if first_letter in ['A', 'B', 'C', 'D']:
            return first_letter
        else:
            return ''
    return ''

pre_prompt_mcqa = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
post_prompt_mcqa = "The best answer is:"
pre_prompt_qa = "Determine whether the following open-ended question description is correct or not based on the video. Respond with only the correct answer (Yes or No)."
post_prompt_qa = "The answer is:"

def process_file(json_file):
    json_path = os.path.join(JSON_FOLDER_PATH, json_file)
    output_path = os.path.join(OUTPUT_FOLDER_PATH, json_file)

    # 如果目标文件已经存在，直接跳过
    if os.path.exists(output_path):
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        video_id = json_file[:-5]
        video_name = f"{video_id}.mp4"
        video_path = os.path.join(VIDEOS_FOLDER_PATH, video_name)
        # 提取视频帧
        try:
            # 提取视频帧
            frames = load_video(video_path, frame_num)[0]
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return

        # 将帧转换为base64编码
        encoded_frames = frames_to_base64(frames)
        for qa_pair in data:
            qa_type = qa_pair['qa_type']
            question = qa_pair['question']
            answer = qa_pair['answer']

            if qa_type == 'qa':
                qs = question
                qs = pre_prompt_qa + '\n' + qs + '\n' + post_prompt_qa
            elif qa_type == 'mcqa':
                options = "\n".join(qa_pair['options'])
                qs = question + "\n" + options
                qs = pre_prompt_mcqa + '\n' + qs + '\n' + post_prompt_mcqa

            qs = "<image>" + '\n' + qs
            response = testOpenaiChatCompletion(qs, encoded_frames)
            pred = process_response(response, qa_type)

            qa_pair['response'] = response
            qa_pair['pred'] = pred

            # print(qa_pair)

        with open(output_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)

def main():
    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)

    json_files = os.listdir(JSON_FOLDER_PATH)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, json_file) for json_file in json_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # 等待所有线程完成

if __name__ == '__main__':
    main()