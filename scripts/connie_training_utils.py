import os
import torch
import time
import sys

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision import models, transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core import MAIN_DIR, MODELS_FOLDER

PREDICTION_FOLDERPATH = os.path.join(MODELS_FOLDER, "prediction_output")
TRAINING_FOLDERPATH = os.path.join(MODELS_FOLDER, "training_output")

SEED = 42
IMG_SIZE = (32, 32)


def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.0
    std = 0.0
    n_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)  # Flatten HxW
        mean += images.mean(1).sum()
        std += images.std(1).sum()
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples
    return mean.item(), std.item()


def get_train_transform(mean, std):
    return transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((360), fill=(0,)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean]*3, std=[std]*3)
])

def get_test_transform(mean, std):
    return transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean]*3, std=[std]*3)
])


def resnet18_model(device):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)
    return model


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        self.indices = dataset.indices

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


class Seed:
    def __init__(self): 
        self.seed = SEED
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def seed_worker(self, worker_id):
        np.random.seed(self.seed + worker_id)
        torch.manual_seed(self.seed + worker_id)

    def generator(self):
        return torch.Generator().manual_seed(self.seed)

    def get_seed(self):
        return self.seed


class ModelTraining:
    def __init__(self):
        self.model = None
        self.best_model_params = None
        self.loss_train = []
        self.acc_train = []
        self.loss_val = []
        self.acc_val = []

        self.fold_train_acc = []
        self.fold_val_acc = []
        self.fold_train_loss = []
        self.fold_val_loss = []

        self.all_fold_train_acc = []
        self.all_fold_val_acc = []
        self.all_fold_train_loss = []
        self.all_fold_val_loss = []
        self.writer_loss_train = None
        self.writer_loss_val = None
        self.writer_acc_train = None
        self.writer_acc_val = None

    def train_model_kfold(self, device,
                          dataset, num_epochs,
                          train_transform,
                          test_transform,
                          k_folds, muon_idx,
                          seed, hyperparam,
                          patience=15):

        full_labels = [label for _, label in dataset.dataset.samples]
        subset_labels = [full_labels[i] for i in dataset.indices]
        binary_labels = [1 if label == muon_idx else 0 for label in subset_labels]
        val_accuracies, val_losses = [], []
        train_accuracies, train_losses = [], []
        fold_precisions, fold_recalls, fold_f1s = [], [], []
        best_acc_across_folds = 0.0
        best_epoch_across_folds = 0

        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed.get_seed())
        start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(binary_labels)), binary_labels)):
            print(f"\n{'='*20} Fold {fold+1}/{k_folds} {'='*20}")

            dataloaders, dataset_sizes = self.__prepare_dataloaders(dataset, train_idx, val_idx, train_transform, test_transform, seed)
            self.__initialize_model(device, dataset, train_idx, muon_idx, hyperparam)

            best_acc = best_epoch = epochs_no_improve = 0
            fold_accuracies, fold_losses = [], []

            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch+1}/{num_epochs} — Fold {fold+1}")
                print("-" * 10)

                for phase in ["train", "val"]:
                    self.model.train() if phase == "train" else self.model.eval()
                    running_loss = 0.0
                    running_corrects = 0

                    all_preds = []
                    all_labels = []

                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = (labels == muon_idx).float().unsqueeze(1).to(device)

                        self.optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == "train"):
                            outputs = self.model(inputs)
                            probs = torch.sigmoid(outputs)
                            loss = self.criterion(outputs, labels)
                            preds = (probs > 0.5).float()

                            if phase == "train":
                                loss.backward()
                                self.optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels)

                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                    if phase == "train":
                        self.scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]
                    print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                    self.__set_acc_loss_kfold(phase, epoch_acc, epoch_loss)

                    if phase == "train":
                        train_accuracies.append(epoch_acc.item())
                        train_losses.append(epoch_loss)

                    if phase == "val":
                        fold_accuracies.append(epoch_acc.item())
                        fold_losses.append(epoch_loss)

                        # NEW: Compute precision, recall, F1 for validation
                        all_preds_bin = [int(p[0]) for p in all_preds]
                        all_labels_bin = [int(l[0]) for l in all_labels]

                        precision = precision_score(all_labels_bin, all_preds_bin, zero_division=0)
                        recall = recall_score(all_labels_bin, all_preds_bin, zero_division=0)
                        f1 = f1_score(all_labels_bin, all_preds_bin, zero_division=0)

                        print(f"Val Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")

                        fold_precisions.append(precision)
                        fold_recalls.append(recall)
                        fold_f1s.append(f1)

                        if epoch_acc > best_acc:
                            best_acc = epoch_acc
                            best_epoch = epoch
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at fold {fold+1}, epoch {epoch+1}")
                    break

            print(f"\nBest Val Accuracy for Fold {fold+1}: {best_acc:.4f} at epoch {best_epoch}")

            val_accuracies.append(np.mean(fold_accuracies))
            val_losses.append(np.mean(fold_losses))

            train_accuracies.append(np.mean(fold_accuracies))
            train_losses.append(np.mean(fold_losses))

            self.__set_metrics_all_kfold()

            if best_acc > best_acc_across_folds:
                best_acc_across_folds = best_acc
                best_epoch_across_folds = best_epoch

        # final metrics
        mean_val_accuracy = np.mean(val_accuracies)
        std_val_accuracy = np.std(val_accuracies)
        mean_val_loss = np.mean(val_losses)
        std_val_loss = np.std(val_losses)
        mean_train_accuracy = np.mean(train_accuracies)
        mean_train_loss = np.mean(train_losses)
        mean_precision = np.mean(fold_precisions)
        mean_recall = np.mean(fold_recalls)
        mean_f1 = np.mean(fold_f1s)

        time_elapsed = time.time() - start_time
        print("\nTraining complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print(f"\nAverage Train Accuracy: {mean_train_accuracy:.4f}")
        print(f"Average Train Loss: {mean_train_loss:.4f}")
        print(f"\nAverage Validation Accuracy: {mean_val_accuracy:.4f} ± {std_val_accuracy:.4f}")
        print(f"Average Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
        print(f"\nMean Precision: {mean_precision:.4f}")
        print(f"Mean Recall:    {mean_recall:.4f}")
        print(f"Mean F1-score:  {mean_f1:.4f}")
        print(f"\nBest val Acc across folds: {best_acc_across_folds:.4f}")
        print(f"Best epoch across folds: {best_epoch_across_folds}")
        return self.model, mean_val_accuracy

    def __prepare_dataloaders(self, dataset, train_idx, val_idx, train_transform, test_transform, seed):
        train_subset = TransformedSubset(Subset(dataset, train_idx), train_transform)
        val_subset = TransformedSubset(Subset(dataset, val_idx), test_transform)

        dataloaders = {
            'train': DataLoader(train_subset, batch_size=64, shuffle=True, generator=seed.generator()),
            'val': DataLoader(val_subset, batch_size=64, shuffle=False, generator=seed.generator())
        }
        dataset_sizes = {'train': len(train_subset), 'val': len(val_subset)}
        return dataloaders, dataset_sizes

    def __initialize_model(self, device, dataset, train_idx, muon_idx, hyperparam):
        self.model = resnet18_model(device)
        raw_labels = [dataset[idx][1] for idx in train_idx]
        num_positive = sum(1 for label in raw_labels if label == muon_idx)
        num_negative = sum(1 for label in raw_labels if label != muon_idx)
        pos_weight_value = min(max(num_negative / num_positive, 1.0), 10.0)
        pos_weight_tensor = torch.tensor([pos_weight_value]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.optimizer = optim.Adam(self.model.parameters(), lr=hyperparam["lr"], weight_decay=hyperparam["wd"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=hyperparam["step"], gamma=hyperparam["gamma"])

    def __set_acc_loss_kfold(self, phase, epoch_acc, epoch_loss):
        if phase == "train":
            self.fold_train_loss.append(epoch_loss)
            self.fold_train_acc.append(float(epoch_acc))
        elif phase == "val":
            self.fold_val_loss.append(epoch_loss)
            self.fold_val_acc.append(float(epoch_acc))

    def __set_metrics_all_kfold(self):
        self.all_fold_train_acc.append(self.fold_train_acc)
        self.all_fold_val_acc.append(self.fold_val_acc)
        self.all_fold_train_loss.append(self.fold_train_loss)
        self.all_fold_val_loss.append(self.fold_val_loss)

    def get_acc_loss_kfold(self):
        return self.all_fold_train_acc, self.all_fold_val_acc, self.all_fold_train_loss, self.all_fold_val_loss
