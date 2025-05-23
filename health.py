import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import numpy as np

# 모델 정의
class LogisticModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticModel, self).__init__()
        self.fc = nn.Linear(input_dim, 2)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리 및 로더
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

data_dir = "chest_xray"
train_loader = DataLoader(datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform), batch_size=32, shuffle=True)
val_loader = DataLoader(datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform), batch_size=32)
test_loader = DataLoader(datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform), batch_size=32)

# 모델 목록
input_dim = 3 * 224 * 224
models_dict = {
    'ResNet18': models.resnet18(pretrained=True),
    'Logistic': LogisticModel(input_dim),
    'DeepNN': DeepNN(input_dim),
}
models_dict['ResNet18'].fc = nn.Linear(models_dict['ResNet18'].fc.in_features, 2)

# 결과 저장용
results = {}

# 학습 함수
def train(model, name):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    total, correct = 0, 0
    start = time.time()
    for epoch in range(5):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total
    duration = time.time() - start
    print(f"[{name}] Train Acc: {acc:.4f} | Time: {duration:.1f} sec")
    return acc, duration

# 평가 함수 + 시각화
def evaluate_and_visualize(model, loader, name):
    model.eval()
    model.to(device)
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)[:, 1]  # 폐렴일 확률
            preds = torch.argmax(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.title(f'{name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{name}_confusion_matrix.png")

    # Classification Report
    print(f"--- {name} Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f'{name} - ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"{name}_roc_curve.png")

    return sum(np.array(y_pred) == np.array(y_true)) / len(y_true)

# 전체 실행
for name, model in models_dict.items():
    print(f"\n===== Training {name} =====")
    train_acc, duration = train(model, name)
    val_acc = evaluate_and_visualize(model, val_loader, name + " (Val)")
    test_acc = evaluate_and_visualize(model, test_loader, name + " (Test)")

    results[name] = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'time': duration,
    }

# 📊 시각화
models_list = list(results.keys())
train_accs = [results[m]['train_acc'] for m in models_list]
val_accs = [results[m]['val_acc'] for m in models_list]
test_accs = [results[m]['test_acc'] for m in models_list]
times = [results[m]['time'] for m in models_list]

x = range(len(models_list))

plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.bar(x, train_accs, width=0.2, label='Train', align='center')
plt.bar([i + 0.2 for i in x], val_accs, width=0.2, label='Val', align='center')
plt.bar([i + 0.4 for i in x], test_accs, width=0.2, label='Test', align='center')
plt.xticks([i + 0.2 for i in x], models_list)
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()

# Time Plot
plt.subplot(1, 2, 2)
plt.bar(models_list, times, color='orange')
plt.ylabel("Time (sec)")
plt.title("Training Time per Model")

plt.tight_layout()
plt.savefig("model_performance.png")