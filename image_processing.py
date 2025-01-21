import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from ultralytics import YOLO  # YOLOv8 library for object detection
import cv2
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 2
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data_path = "/Users/georgijgavrilov/programm_engineering_3/document/train"
val_data_path = "/Users/georgijgavrilov/programm_engineering_3/document/val"

train_dataset = datasets.ImageFolder(train_data_path, transform=transform)
val_dataset = datasets.ImageFolder(val_data_path, transform=transform)

print(f"Классы в тренировочном наборе: {train_dataset.classes}")
print(f"Классы в валидационном наборе: {val_dataset.classes}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_labels = set(train_dataset.targets)
val_labels = set(val_dataset.targets)
print(f"Уникальные метки в тренировочном наборе: {train_labels}")
print(f"Уникальные метки в валидационном наборе: {val_labels}")

print(f"Используемое устройство: {DEVICE}")

from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        print(f"Начало эпохи {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if i == 0:
                print(f"Input shape: {inputs.shape}, Labels shape: {labels.shape}")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 0:
                print(f"Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

torch.save(model.state_dict(), "classification_model.pth")

model_yolo = YOLO('yolov8n.pt')  # Используется самая маленькая версия модели YOLOv8

image_path = '/Users/georgijgavrilov/programm_engineering_3/sample_image.jpg'

results = model_yolo(image_path)

result_img = results[0].plot()

plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

results.save(save_dir='detections/')

print("Классификация и обнаружение объектов завершены!")
