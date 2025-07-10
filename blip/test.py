# blip/test.py
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# === 参数设置 ===
image_path = "../data/test_images/test1.jpg"   # 你要测试的图片路径
model_path = "../output/checkpoints/blip-caption-epoch5"  # 你训练保存的模型目录

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载模型和 Processor ===
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

# === 加载图片 ===
raw_image = Image.open(image_path).convert("RGB")

# === 图像预处理 ===
inputs = processor(images=raw_image, return_tensors="pt").to(device)

# === 生成描述 ===
with torch.no_grad():
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

print("Predicted caption:", caption)
