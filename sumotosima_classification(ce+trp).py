import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from sklearn.metrics import classification_report, f1_score
from PIL import Image
import copy
import pandas as pd
from sklearn.model_selection import KFold

# Custom dataset for triplet loss and contrastive loss
class TripletContrastiveDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_paths = {}
        self.label_map = {}
        self._load_images()

    def _load_images(self):
        for label, category in enumerate(os.listdir(self.image_folder)):
            category_folder = os.path.join(self.image_folder, category)
            if os.path.isdir(category_folder):
                self.label_to_paths[label] = []
                self.label_map[label] = category
                for img_name in os.listdir(category_folder):
                    if img_name.lower().endswith('.jpg'):
                        img_path = os.path.join(category_folder, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                        self.label_to_paths[label].append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]

        positive_path = random.choice(self.label_to_paths[anchor_label])
        negative_label = random.choice(list(set(self.labels) - {anchor_label}))
        negative_path = random.choice(self.label_to_paths[negative_label])

        contrastive_positive_label = random.choice(list(set(self.labels) - {anchor_label}))
        contrastive_positive_path = random.choice(self.label_to_paths[contrastive_positive_label])

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')
        contrastive_positive = Image.open(contrastive_positive_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            contrastive_positive = self.transform(contrastive_positive)

        return anchor, positive, negative, contrastive_positive, anchor_label

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
image_folder = '/content/drive/MyDrive/sumotosima/FinalData' # Replace with your path on Google Drive
dataset = TripletContrastiveDataset(image_folder, transform)

# Define the neural network with embedding layer
class EmbeddingNet(nn.Module):
    def __init__(self, num_classes):
        super(EmbeddingNet, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

# Define triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# Define contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, contrastive_positive):
        distance = (anchor - contrastive_positive).pow(2).sum(1)
        losses = torch.relu(distance - self.margin)
        return losses.mean()

# Training loop with triplet loss and contrastive loss
def train_model(model, train_loader, val_loader, triplet_criterion, contrastive_criterion, classification_criterion, optimizer, num_epochs=100):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            all_labels = []
            all_preds = []

            for anchors, positives, negatives, contrastive_positives, labels in loader:
                anchors, positives, negatives, contrastive_positives, labels = anchors.to(device), positives.to(device), negatives.to(device), contrastive_positives.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    anchor_embeddings = model(anchors)
                    positive_embeddings = model(positives)
                    negative_embeddings = model(negatives)
                    contrastive_embeddings = model(contrastive_positives)

                    triplet_loss = triplet_criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                    contrastive_loss = contrastive_criterion(anchor_embeddings, contrastive_embeddings)
                    classification_outputs = model.classifier(anchor_embeddings)
                    classification_loss = classification_criterion(classification_outputs, labels)

                    loss = classification_loss + triplet_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * anchors.size(0)
                _, preds = torch.max(classification_outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(loader.dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

            print(f'{phase} Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val F1: {:4f}'.format(best_f1))

    model.load_state_dict(best_model_wts)
    return model

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (train_indices, test_indices) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}')

    # Split data
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    val_size = int(0.2 * len(test_subset))
    test_size = len(test_subset) - val_size
    val_subset, test_subset = random_split(test_subset, [val_size, test_size])

    # Data loaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    # Initialize model, loss functions, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = EmbeddingNet(num_classes=len(set(dataset.labels))).to(device)
    triplet_criterion = TripletLoss(margin=1.0)
    contrastive_criterion = ContrastiveLoss(margin=1.0)
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # Train and validate the model
    model = train_model(model, train_loader, val_loader, triplet_criterion, contrastive_criterion, classification_criterion, optimizer, num_epochs=100)

    # Evaluate on the test set
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for anchors, _, _, _, labels in test_loader:
            anchors = anchors.to(device)
            labels = labels.to(device)
            anchor_embeddings = model(anchors)
            classification_outputs = model.classifier(anchor_embeddings)
            _, preds = torch.max(classification_outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    report = classification_report(all_labels, all_preds, output_dict=True)
    print(f"\nClassification Report for Fold {fold + 1}:\n", classification_report(all_labels, all_preds))

    fold_result = {
        'Fold': fold + 1,
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1 Score': report['weighted avg']['f1-score']
    }
    results.append(fold_result)

    # Save model weights for this fold
    torch.save(model.state_dict(), f'/content/drive/MyDrive/sumotosima/modeltrip100_fold_{fold + 1}.pth')

# Save results to log file
results_df = pd.DataFrame(results)
results_df.to_csv('/content/drive/MyDrive/sumotosima/trip100_cross_validation_results.csv', index=False)

print(results_df)
