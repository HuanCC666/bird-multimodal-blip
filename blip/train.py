# blip/train.py
import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AdamW, get_scheduler
from tqdm import tqdm
import os

from dataset_inat import INatBirdDataset

# === 设置路径和参数 ===
metadata_csv = '../data/inat2021_birds/metadata.csv'
description_json = '../data/inat2021_birds/description.json'
image_root = '../data/inat2021_birds'
save_dir = '../output/inat/checkpoints'

# 训练超参数
batch_size = 4
lr = 5e-5
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载模型和 tokenizer ===
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model = model.float().to(device)


# === 准备数据集 ===
train_dataset = INatBirdDataset(metadata_csv, description_json, image_root, processor, split='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# === 优化器和调度器 ===
optimizer = AdamW(model.parameters(), lr=lr)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * num_epochs
)

# === 开始训练 ===
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in loop:
        dtype = next(model.parameters()).dtype
        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)



        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # 使用 teacher forcing
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")

    # === 保存模型 ===
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(os.path.join(save_dir, f"blip-caption-epoch{epoch+1}"))
    processor.save_pretrained(os.path.join(save_dir, f"blip-caption-epoch{epoch+1}"))
