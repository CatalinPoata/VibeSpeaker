import os
import random
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, get_scheduler
from transformers import RobertaTokenizer, RobertaModel
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PretrainedConfig
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import dataset
import model

def train_one_epoch(model, dataloader, optimizer, scheduler, device, criterion, accumulation_steps=1):
    model.train()
    total_loss = 0

    for step, (padded_audio_input_values, attention_mask_audio, text_inputs, spectrograms, labels) in enumerate(
        tqdm(dataloader, desc="Training")
    ):
        padded_audio_input_values = padded_audio_input_values.to(device)
        attention_mask_audio = attention_mask_audio.to(device)
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        text_inputs = {key: val.to(device) for key, val in text_inputs.items()}

        outputs = model(
            audio_input_values=padded_audio_input_values,
            audio_attention_mask=attention_mask_audio,
            text_input_ids=text_inputs["input_ids"],
            text_attention_mask=text_inputs["attention_mask"],
            spectrograms=spectrograms
        )

        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps

        loss.backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for padded_audio_input_values, attention_mask_audio, text_inputs, spectrograms, labels in tqdm(dataloader, desc="Evaluating"):
            padded_audio_input_values = padded_audio_input_values.to(device)
            attention_mask_audio = attention_mask_audio.to(device)
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            text_inputs = {key: val.to(device) for key, val in text_inputs.items()}

            outputs = model(
                audio_input_values=padded_audio_input_values,
                audio_attention_mask=attention_mask_audio,
                text_input_ids=text_inputs["input_ids"],
                text_attention_mask=text_inputs["attention_mask"],
                spectrograms=spectrograms
            )

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy

train_csv_path = "./train.csv"
test_csv_path = "./test.csv"
audio_dir = "./validated_audio"
pretrained_audio_model = "microsoft/wavlm-base-plus"
pretrained_text_model = "roberta-base"
num_classes = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_audio_model)
text_tokenizer = RobertaTokenizer.from_pretrained(pretrained_text_model)

label_map = {
    'sad': 0,
    'surprised': 1,
    'happy': 2,
    'disgusted': 3,
    'fearful': 4,
    'angry': 5,
    'joy': 6,
    'euphoria': 7
}

config = model.MultimodalSERConfigWithCNN(
    hidden_dim_audio=768, 
    hidden_dim_text=768, 
    hidden_dim_cnn=128,
    num_classes=8
)

model = model.MultimodalSERModelWithCNN(config).to(device)

# Freeze layers
for param in model.audio_model.feature_extractor.parameters():
    param.requires_grad = False

for param in model.audio_model.encoder.parameters():
    param.requires_grad = False

for param in model.text_model.embeddings.parameters():
    param.requires_grad = False

for param in model.text_model.encoder.parameters():
    param.requires_grad = False

# Data loading
train_dataset = dataset.AudioDataset(train_csv_path, audio_dir, feature_extractor, text_tokenizer, label_map, augment=False)
test_dataset = dataset.AudioDataset(test_csv_path, audio_dir, feature_extractor, text_tokenizer, label_map, augment=False)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=lambda batch: dataset.collate_fn(batch, feature_extractor)
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=lambda batch: dataset.collate_fn(batch, feature_extractor)
)

# Training setup
num_epochs = 10
learning_rate = 1e-4
accumulation_steps = 1
weight_decay = 0.01

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

num_training_steps = (len(train_dataloader) // accumulation_steps) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

train_losses, test_losses, test_accuracies = [], [], []
learning_rates = []

best_test_loss = float('inf')
best_test_accuracy = 0.0
model_save_path = "./multimodal_SER"

best_model_state_dict = None
best_processor = None
best_tokenizer = None

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, device, criterion, accumulation_steps)
    test_loss, test_accuracy = evaluate(model, test_dataloader, device, criterion)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    current_lr = scheduler.get_last_lr()
    learning_rates.append(current_lr[0])
    print(f"  Current Learning Rate: {current_lr[0]:.6f}")

    print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    if test_loss < best_test_loss and test_accuracy > best_test_accuracy:
        best_test_loss = test_loss
        best_test_accuracy = test_accuracy
        best_model_state_dict = model.state_dict()
        best_processor = feature_extractor
        best_tokenizer = text_tokenizer
        print(f"  Best model updated at epoch {epoch + 1}.")

if best_model_state_dict is not None:
    model.load_state_dict(best_model_state_dict)
    model.save_pretrained(model_save_path)
    model.config.save_pretrained(model_save_path)
    best_processor.save_pretrained(model_save_path)
    best_tokenizer.save_pretrained(model_save_path)
    print(f"Best model and processor saved to {model_save_path}")
else:
    print("No improvement found during training.")

# Save and plot losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), test_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.savefig("classification_train_val_loss.png")
plt.close()

# Save and plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), test_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid()
plt.savefig("classification_accuracy.png")
plt.close()

# Learning Rate plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(learning_rates) + 1), learning_rates, label="Learning Rate")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid()
plt.savefig("learning_rate_schedule.png")
plt.close()