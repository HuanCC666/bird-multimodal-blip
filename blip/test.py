# # blip/test.py
# import torch
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # === 参数设置 ===
# image_path = "../data/test_images/test.jpg"   # 你要测试的图片路径
# model_path = "../output/inat/checkpoints/blip-caption-epoch5"  # 你训练保存的模型目录

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # === 加载模型和 Processor ===
# processor = BlipProcessor.from_pretrained(model_path)
# model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
# model.eval()

# # === 加载图片 ===
# raw_image = Image.open(image_path).convert("RGB")

# # === 图像预处理 ===
# inputs = processor(images=raw_image, return_tensors="pt").to(device)

# # === 生成描述 ===
# with torch.no_grad():
#     out = model.generate(**inputs)
#     caption = processor.decode(out[0], skip_special_tokens=True)

# print("Predicted caption:", caption)
import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from dataset_CUB import CUBBirdDataset  # 你已有的 Dataset 类

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metadata_csv = "../data/cub_metadata.csv"
description_json = "../data/descriptions.json"
image_root = "../data/CUB_200_2011/images/"
batch_size = 1  # 一张图一个描述，便于评估

# 加载模型
processor = BlipProcessor.from_pretrained("../output/inat/checkpoints/blip-caption-epoch5")
model = BlipForConditionalGeneration.from_pretrained("../output/inat/checkpoints/blip-caption-epoch5")
model.to(device)
model.eval()

# 加载数据
dataset = CUBBirdDataset(metadata_csv, description_json, image_root, processor, split='val')
dataloader = DataLoader(dataset, batch_size=batch_size)

# 评估
smoothie = SmoothingFunction().method4
scores = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating BLEU"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        outputs = model.generate(**inputs)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # 获取 GT 描述
        img_path = dataset.samples[batch["label"].item()]["img_path"]
        gt_texts = dataset.descriptions[img_path]

        # 计算 BLEU 得分（参考可多个）
        bleu_score = sentence_bleu([t.split() for t in gt_texts], generated_text.split(), smoothing_function=smoothie)
        scores.append(bleu_score)

print(f"\n✅ Average BLEU score: {sum(scores)/len(scores):.4f}")
