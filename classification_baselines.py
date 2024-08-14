import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Define the custom dataset class
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        self._load_images()

    def _load_images(self):
        for label, category in enumerate(os.listdir(self.image_folder)):
            category_folder = os.path.join(self.image_folder, category)
            if os.path.isdir(category_folder):
                self.label_map[label] = category
                for img_name in os.listdir(category_folder):
                    if img_name.lower().endswith('.jpg'):
                        img_path = os.path.join(category_folder, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
image_folder = r"C:\Users\Anas Khan\Desktop\appi_paper\FinalData"  # Replace with your path
dataset = ImageDataset(image_folder, transform)

# Define a function to extract features using a pre-trained model
def extract_features(dataset, model, device):
    model.to(device)
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for img, label in DataLoader(dataset, batch_size=32):
            img = img.to(device)
            feature = model(img).cpu().numpy()
            features.append(feature)
            labels.extend(label.numpy())
    return np.concatenate(features), np.array(labels)

# Load pre-trained model and extract features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # Remove the classification layer

# Extract features and labels
features, labels = extract_features(dataset, model, device)

# Split data into training + validation and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=0.20, random_state=43)

# Further split training + validation into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.10/(0.20 + 0.10), random_state=43)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=43)
rf_classifier.fit(X_train, y_train)
rf_val_predictions = rf_classifier.predict(X_val)
rf_test_predictions = rf_classifier.predict(X_test)

# SVM Classifier
svm_classifier = SVC(kernel='linear', random_state=43)
svm_classifier.fit(X_train, y_train)
svm_val_predictions = svm_classifier.predict(X_val)
svm_test_predictions = svm_classifier.predict(X_test)

# k-NN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
knn_val_predictions = knn_classifier.predict(X_val)
knn_test_predictions = knn_classifier.predict(X_test)

# Metrics for Random Forest
print("Random Forest Validation Report:")
print(classification_report(y_val, rf_val_predictions))
print("Random Forest Test Report:")
print(classification_report(y_test, rf_test_predictions))

print("Random Forest Validation Precision:", precision_score(y_val, rf_val_predictions, average='weighted'))
print("Random Forest Validation Recall:", recall_score(y_val, rf_val_predictions, average='weighted'))
print("Random Forest Validation F1 Score:", f1_score(y_val, rf_val_predictions, average='weighted'))

print("Random Forest Test Precision:", precision_score(y_test, rf_test_predictions, average='weighted'))
print("Random Forest Test Recall:", recall_score(y_test, rf_test_predictions, average='weighted'))
print("Random Forest Test F1 Score:", f1_score(y_test, rf_test_predictions, average='weighted'))

# Metrics for SVM
print("SVM Validation Report:")
print(classification_report(y_val, svm_val_predictions))
print("SVM Test Report:")
print(classification_report(y_test, svm_test_predictions))

print("SVM Validation Precision:", precision_score(y_val, svm_val_predictions, average='weighted'))
print("SVM Validation Recall:", recall_score(y_val, svm_val_predictions, average='weighted'))
print("SVM Validation F1 Score:", f1_score(y_val, svm_val_predictions, average='weighted'))

print("SVM Test Precision:", precision_score(y_test, svm_test_predictions, average='weighted'))
print("SVM Test Recall:", recall_score(y_test, svm_test_predictions, average='weighted'))
print("SVM Test F1 Score:", f1_score(y_test, svm_test_predictions, average='weighted'))

# Metrics for k-NN
print("k-NN Validation Report:")
print(classification_report(y_val, knn_val_predictions))
print("k-NN Test Report:")
print(classification_report(y_test, knn_test_predictions))

print("k-NN Validation Precision:", precision_score(y_val, knn_val_predictions, average='weighted'))
print("k-NN Validation Recall:", recall_score(y_val, knn_val_predictions, average='weighted'))
print("k-NN Validation F1 Score:", f1_score(y_val, knn_val_predictions, average='weighted'))

print("k-NN Test Precision:", precision_score(y_test, knn_test_predictions, average='weighted'))
print("k-NN Test Recall:", recall_score(y_test, knn_test_predictions, average='weighted'))
print("k-NN Test F1 Score:", f1_score(y_test, knn_test_predictions, average='weighted'))
