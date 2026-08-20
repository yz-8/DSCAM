#!/usr/bin/env python3
"""LLaVA 1.5 7B inference script for AutoDL.

This script loads the LLaVA model once globally and processes the JSONL data.
Outputs are saved strictly in the requested JSONL format.
"""

from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path
from typing import Any
from PIL import Image

# 引入 LLaVA 的核心组件
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ask_model(image_path: Path, question: str, model, tokenizer, image_processor, conv_mode="llava_v1") -> str:
    """Core inference function for LLaVA 1.5."""
    # 1. 准备图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    # 2. 准备文本 Prompt (LLaVA 要求 prompt 中包含特定的图像占位符)
    qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 3. 文本 Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # 4. 模型生成 (使用贪心解码，保证客观题评估的一致性)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,  
            temperature=0,
            max_new_tokens=128, # 控制输出长度
            use_cache=True
        )

    # 5. 解码输出
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("data/questions.current.autodl.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("model_outputs/full_image_outputs.jsonl"))
    # 添加模型路径参数，建议指向 AutoDL 数据盘上的权重文件夹
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b") 
    parser.add_argument("--model-name", default="llava-1.5-7b")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # --- 核心修改：在循环外部全局加载一次模型 ---
    print(f"Loading model from {args.model_path}...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device="cuda"
    )
    print("Model loaded successfully!")
    # ------------------------------------------

    questions = read_jsonl(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # 按照你的文档建议，可以先跑前 20 条或者 100 条做 debug
    with args.output.open("w", encoding="utf-8") as f:
        for index, row in enumerate(questions, 1):
            image_path = Path(row["image_path"])
            
            # 你的 prompt 加上了限制词以满足格式要求
            prompt = row["question"] + "\nAnswer with the shortest valid answer only."
            
            # 将模型组件传入推理函数
            prediction = ask_model(
                image_path=image_path, 
                question=prompt, 
                model=model, 
                tokenizer=tokenizer, 
                image_processor=image_processor
            )
            
            out = {
                "question_id": row["question_id"],
                "prediction": prediction,
                "model": args.model_name,
                "setting": "full_image",
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            
            if index % 10 == 0:
                print(f"Processed {index}/{len(questions)}")
                
    print(f"Inference complete. Results saved to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())