import copy
import os
import torch
import time
import sys
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, DenseNet121_Weights
from torchvision import models, transforms
from skimage.morphology import skeletonize
from skimage.measure import find_contours
from scipy.ndimage import convolve


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core import MODELS_FOLDER

PREDICTION_FOLDERPATH = os.path.join(MODELS_FOLDER, "prediction_output")
TRAINING_FOLDERPATH = os.path.join(MODELS_FOLDER, "training_output")

SEED = 42
#IMG_SIZE = (32, 32)
IMG_SIZE = (64, 64)


def extract_skeleton_features(x_coord, y_coord):
    mask = create_mask_from_coords(x_coord, y_coord)

    total_area = mask.sum()

    skeleton = skeletonize(mask)

    # Feature 1: total skeleton length
    skeleton_length = skeleton.sum()

    # Endpoint and branchpoint count
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    # Convolve the skeleton with the kernel to get the number of neighbors for each pixel
    neighbors = convolve(skeleton.astype(np.uint8),
                         kernel, mode='constant', cval=0)
    
    # Identify skeleton pixels
    skeleton_pixels = skeleton.astype(bool)
    # Feature 2: Endpoint count
    endpoints = (neighbors == 1) & skeleton_pixels
    num_endpoints = endpoints.sum()
 
    # Feature 3: Branchpoint count
    branch_points = (neighbors >= 3) & skeleton_pixels
    num_branches = branch_points.sum()
    
    # Feature 4: Ratio of branches to endpoints
    if num_endpoints > 0:
        branch_to_end_ratio = num_branches/num_endpoints
    else:
        branch_to_end_ratio = float(num_branches) 

    # Feature 5: Skeleton area ratio
    skeleton_area_ratio = skeleton_length/total_area if total_area > 0 else 0.0

    # NEW FEATURES
    skel_y, skel_x = np.where(skeleton)
    
    # Default values for insufficient data
    skeleton_tortuosity = 1.0
    skeleton_linearity_deviation = 0.0
    skeleton_curvature_sum = 0.0
    
    if len(skel_x) >= 3:
        # Feature 1: Tortuosity (path length / end-to-end distance)
        # Sort points to approximate path
        points = np.column_stack([skel_x, skel_y])
        
        # Simple path: sort by primary axis
        if np.std(skel_x) > np.std(skel_y):
            # Mostly horizontal, sort by x
            sorted_indices = np.argsort(skel_x)
        else:
            # Mostly vertical, sort by y
            sorted_indices = np.argsort(skel_y)
        
        sorted_x = skel_x[sorted_indices]
        sorted_y = skel_y[sorted_indices]
        
        # Calculate path length
        dx = np.diff(sorted_x)
        dy = np.diff(sorted_y)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        path_length = np.sum(segment_lengths)
        
        # End-to-end distance
        end_to_end = np.sqrt((sorted_x[-1] - sorted_x[0])**2 + 
                            (sorted_y[-1] - sorted_y[0])**2)
        
        if end_to_end > 1e-6:
            skeleton_tortuosity = path_length / end_to_end
        
        # Feature 2: Deviation from linear fit (wobbliness measure)
        if len(sorted_x) > 2:
            # Check for sufficient variation in x coordinates
            x_range = np.max(sorted_x) - np.min(sorted_x)
            
            if x_range < 1e-6:
                # All x values nearly identical (vertical line)
                # Use distance from mean x instead
                skeleton_linearity_deviation = np.std(sorted_x)
            else:
                # Remove duplicate x coordinates to avoid singular matrix
                unique_mask = np.concatenate([[True], sorted_x[1:] != sorted_x[:-1]])
                unique_x = sorted_x[unique_mask]
                unique_y = sorted_y[unique_mask]
                
                if len(unique_x) >= 2:
                    try:
                        # Fit a straight line through the skeleton
                        coeffs = np.polyfit(unique_x, unique_y, 1)
                        
                        # Calculate deviation for all points (including duplicates)
                        y_fit = np.polyval(coeffs, sorted_x)
                        
                        # Mean absolute deviation from the line
                        deviations = np.abs(sorted_y - y_fit)
                        skeleton_linearity_deviation = np.mean(deviations)
                    except (np.linalg.LinAlgError, ValueError, np.RankWarning):
                        # Polyfit failed, use standard deviation as fallback
                        skeleton_linearity_deviation = np.std(sorted_y)
                else:
                    # Only one unique point after filtering
                    skeleton_linearity_deviation = 0.0
        
        # Feature 3: Curvature (sum of angle changes)
        if len(sorted_x) > 2:
            # Calculate angles between consecutive segments
            # Only calculate for non-zero length segments
            non_zero_segments = segment_lengths > 1e-10
            
            if np.any(non_zero_segments):
                dx_valid = dx[non_zero_segments]
                dy_valid = dy[non_zero_segments]
                
                angles = np.arctan2(dy_valid, dx_valid)
                
                if len(angles) > 1:
                    # Angle changes between segments
                    angle_changes = np.diff(angles)
                    # Normalize to [-π, π]
                    angle_changes = np.abs((angle_changes + np.pi) % (2 * np.pi) - np.pi)
                    # Sum of absolute angle changes (total curvature)
                    skeleton_curvature_sum = np.sum(angle_changes)

    features = {
        'skeleton_length': float(skeleton_length),
        'num_branches': int(num_branches),
        'num_endpoints': int(num_endpoints),
        'branch_to_end_ratio': float(branch_to_end_ratio),
        'skeleton_area_ratio': float(skeleton_area_ratio),
        'skeleton_tortuosity': float(skeleton_tortuosity),
        'skeleton_linearity_deviation': float(skeleton_linearity_deviation),
        'skeleton_curvature_sum': float(skeleton_curvature_sum)
    }

    return features


def extract_fourier_descriptors(x_coord,
                                y_coord,
                                num_descriptors=10):
    """
    Calculates rotation, scale, and translation-invariant Fourier Descriptors (FDs)
    by first extracting the contour from the interior pixel coordinates.
    """
    
    x_coord = np.asarray(x_coord)
    y_coord = np.asarray(y_coord)

    fd_features_dict = {f'FD_{i}': 0.0 for i in range(2, 2 + num_descriptors)}

    if len(x_coord) < 5:
        return fd_features_dict

    mask = create_mask_from_coords(x_coord, y_coord)
    
    if mask.sum() == 0:
        return fd_features_dict
        
    # 2. Extract Ordered Contour Points
    # The 0.5 threshold finds the boundary between 0 (background) and 1 (shape)
    contours = find_contours(mask, 0.5) 
    
    if not contours:
        return fd_features_dict
        
    # Take the largest contour (main object boundary)
    contour_coords = max(contours, key=len) 
    
    # Separate y and x (find_contours returns (row, col) = (y, x))
    y_contour = contour_coords[:, 0]
    x_contour = contour_coords[:, 1]
    
    # 3. Create the Complex Contour Sequence (s(k) = x(k) + i * y(k))
    complex_contour = x_contour + 1j * y_contour

    # 4. Apply DFT and Calculate Magnitudes
    fourier_coeffs = np.fft.fft(complex_contour)
    magnitudes = np.abs(fourier_coeffs)

    # 5. Normalize for Invariance (using |S(1)| as the scale reference)
    s1_magnitude = magnitudes[1]
    
    if s1_magnitude == 0:
        return fd_features_dict

    # Select and normalize features: FD_n = |S(n)| / |S(1)|, starting from n=2.
    start_idx, end_idx = 2, 2 + num_descriptors
    end_idx = min(end_idx, len(magnitudes))
    
    fd_features = magnitudes[start_idx:end_idx] / s1_magnitude
    
    # Pad with zeros if necessary and store in dictionary
    for i, val in enumerate(fd_features, start=2):
        fd_features_dict[f'FD_{i}'] = float(val)

    return fd_features_dict


def create_mask_from_coords(x_array, y_array):
    """Creates a small binary mask containing the object
    from all interior pixel coordinates."""
    x_coords = np.asarray(x_array).astype(int)
    y_coords = np.asarray(y_array).astype(int)

    if x_coords.size < 3:
        return np.zeros((3, 3), dtype=bool)

    # Shift coordinates to start at (1, 1) for a safe border
    min_y = y_coords.min()
    min_x = x_coords.min()

    shifted_y = y_coords - min_y + 1
    shifted_x = x_coords - min_x + 1

    mask_rows = shifted_y.max() + 2
    mask_cols = shifted_x.max() + 2
    
    mask = np.zeros((mask_rows, mask_cols), dtype=bool)
    mask[shifted_y, shifted_x] = True 
    
    return mask


class PadToSizeTensor:
    def __init__(self, target_size):
        self.target_h, self.target_w = target_size

    def __call__(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        _, h, w = x.shape

        pad_h = max(self.target_h - h, 0)
        pad_w = max(self.target_w - w, 0)

        padding = (
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
        )

        return F.pad(x, padding)

class ScaleNormalizeTensor:
    def __init__(self, max_value):
        self.max_value = max_value

    def __call__(self, x):
        return x / (self.max_value + 1e-8)


def get_train_transform(max_value):
    return transforms.Compose([
        #PadToSizeTensor(IMG_SIZE),
        transforms.Resize(IMG_SIZE),
        ScaleNormalizeTensor(max_value),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((360), fill=(0,))
    ])

def get_test_transform(max_value):
    return transforms.Compose([
        #PadToSizeTensor(IMG_SIZE),
        transforms.Resize(IMG_SIZE),
        ScaleNormalizeTensor(max_value)
    ])


class ScaleNormalize:
    def __init__(self, max_value):
        self.max_value = max_value

    def __call__(self, x):
        x = x.astype(np.float32)
        return x / (self.max_value + 1e-8)


class NPYFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, preload=True):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.preload = preload
        self.classes = {}

        # Store arrays if preload=True
        self.data = []
        # Store max value per sample
        self.sample_max = []

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                for file in os.listdir(class_path):
                    if file.endswith(".npy"):
                       self.samples.append((os.path.join(class_path, file), idx))
        if self.preload:
            for path, label in self.samples:
                arr = np.load(path)
                self.data.append((arr, label))
                self.sample_max.append(arr.max())
        else:
            for path, _ in self.samples:
                arr = np.load(path, mmap_mode='r')
                self.sample_max.append(arr.max())
        self.classes = sorted(self.class_to_idx.keys())


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload:
            arr, label = self.data[idx]
        else:
            path, label = self.samples[idx]
            arr = np.load(path)
        if self.transform:
            arr = self.transform(arr)
        # Convert to tensor only once here
        if isinstance(arr, np.ndarray):
            arr = torch.tensor(arr, dtype=torch.float32)
        if arr.dim() == 2:
            arr = arr.unsqueeze(0)
        return arr, label

def resnet18_model(device, class_qty=1, pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if pretrained:
        old_weights = model.conv1.weight.data.clone()
    model.conv1 = nn.Conv2d(1, 64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
    if pretrained:
        model.conv1.weight.data = old_weights.mean(dim=1, keepdim=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, class_qty)
    return model.to(device)


def densenet121_model(device, class_qty=1, pretrained=True):
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    if pretrained:
        old_weights = model.features.conv0.weight.data.clone()
    model.features.conv0 = nn.Conv2d(
        1, 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    if pretrained:
        model.features.conv0.weight.data = old_weights.mean(dim=1, keepdim=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, class_qty)
    return model.to(device)


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]  # use the wrapped dataset
        if self.transform:
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

    def train_model_kfold(self, device,
                          dataset, num_epochs,
                          k_folds, current_class_idx,
                          seed, hyperparam,
                          model, patience=15):

        full_labels = [label for _, label in dataset.dataset.samples]
        subset_labels = [full_labels[i] for i in dataset.indices]
        binary_labels = [1 if label == current_class_idx else 0 for label in subset_labels]

        self.val_accuracies, self.val_losses = [], []
        self.train_accuracies, self.train_losses = [], []
        self.fold_precisions, self.fold_recalls = [], []
        self.fold_f1s_binary, self.fold_f1s_macro = [], []
        self.best_epochs = []  # store best epoch (1-based) per fold

        kf = StratifiedKFold(n_splits=k_folds, shuffle=True,
                             random_state=seed.get_seed())
        start_time = time.time()

        all_val_true_list = []
        all_val_preds_list = []

        for fold, (train_idx, val_idx) in enumerate(
                kf.split(np.zeros(len(binary_labels)), binary_labels)):

            print(f"\n{'='*20} Fold {fold+1}/{k_folds} {'='*20}")

            train_max = max(dataset.dataset.sample_max[i] for i in train_idx)

            train_subset = TransformedSubset(Subset(dataset, train_idx),
                                             get_train_transform(train_max))
            val_subset = TransformedSubset(Subset(dataset, val_idx),
                                           get_test_transform(train_max))

            dataloaders = {
                'train': DataLoader(train_subset, batch_size=64, shuffle=True,
                                    generator=seed.generator(), num_workers=0, pin_memory=True),
                'val': DataLoader(val_subset, batch_size=64, shuffle=False,
                                  generator=seed.generator(), num_workers=0, pin_memory=True)
            }
            dataset_sizes = {'train': len(train_subset), 'val': len(val_subset)}

            self.__initialize_model(device, dataset, train_idx,
                                    current_class_idx, hyperparam, model)

            best_metric = -1.0
            best_epoch = 0
            epochs_no_improve = 0

            best_val_preds = None
            best_val_labels = None
            best_model_wts = copy.deepcopy(self.model.state_dict())

            # Best-epoch snapshot (val)
            best_fold_val_acc = None
            best_fold_val_loss = None
            best_fold_precision = None
            best_fold_recall = None
            best_fold_f1_binary = None
            best_fold_f1_macro = None

            best_fold_train_acc = None
            best_fold_train_loss = None

            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch+1}/{num_epochs} — Fold {fold+1}")
                print("-" * 10)

                epoch_train_acc = None
                epoch_train_loss = None

                for phase in ["train", "val"]:
                    self.model.train() if phase == "train" else self.model.eval()

                    running_loss = 0.0
                    running_corrects = 0

                    all_preds = []
                    all_labels = []

                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = (labels == current_class_idx).float().unsqueeze(1).to(device)

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

                        all_preds.extend(preds.detach().cpu().numpy())
                        all_labels.extend(labels.detach().cpu().numpy())

                    if phase == "train":
                        self.scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

                    print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                    if phase == "train":
                        epoch_train_acc = epoch_acc
                        epoch_train_loss = epoch_loss

                    if phase == "val":
                        all_preds_bin = np.array(all_preds).flatten().astype(int)
                        all_labels_bin = np.array(all_labels).flatten().astype(int)

                        precision = precision_score(all_labels_bin, all_preds_bin,
                                                    average="binary", zero_division=0)
                        recall = recall_score(all_labels_bin, all_preds_bin,
                                              average="binary", zero_division=0)
                        f1_binary = f1_score(all_labels_bin, all_preds_bin,
                                             average="binary", zero_division=0)
                        f1_macro = f1_score(all_labels_bin, all_preds_bin,
                                            average="macro", zero_division=0)

                        print(f"Val Precision: {precision:.4f} Recall: {recall:.4f} "
                            f"F1_binary: {f1_binary:.4f} F1_macro: {f1_macro:.4f}")

                        # Select best epoch by positive-class F1 for OvA
                        current_metric = f1_binary

                        if current_metric > best_metric:
                            best_metric = current_metric
                            best_epoch = epoch
                            epochs_no_improve = 0

                            best_val_preds = all_preds_bin.copy()
                            best_val_labels = all_labels_bin.copy()
                            best_model_wts = copy.deepcopy(self.model.state_dict())

                            best_fold_val_acc = epoch_acc
                            best_fold_val_loss = epoch_loss
                            best_fold_precision = precision
                            best_fold_recall = recall
                            best_fold_f1_binary = f1_binary
                            best_fold_f1_macro = f1_macro

                            best_fold_train_acc = epoch_train_acc
                            best_fold_train_loss = epoch_train_loss
                        else:
                            epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at fold {fold+1}, epoch {epoch+1}")
                    break

            print(f"\nBest F1_binary for Fold {fold+1}: {best_metric:.4f} at epoch {best_epoch+1}")

            self.model.load_state_dict(best_model_wts)

            all_val_true_list.extend(best_val_labels)
            all_val_preds_list.extend(best_val_preds)

            self.train_accuracies.append(best_fold_train_acc)
            self.train_losses.append(best_fold_train_loss)

            self.val_accuracies.append(best_fold_val_acc)
            self.val_losses.append(best_fold_val_loss)

            self.fold_precisions.append(best_fold_precision)
            self.fold_recalls.append(best_fold_recall)
            self.fold_f1s_binary.append(best_fold_f1_binary)
            self.fold_f1s_macro.append(best_fold_f1_macro)

            self.best_epochs.append(best_epoch + 1)  # 1-based

        metrics = self.__compute_metrics()
        metrics["all_val_true"] = np.array(all_val_true_list)
        metrics["all_val_preds"] = np.array(all_val_preds_list)
        metrics["best_epochs_per_fold"] = np.array(self.best_epochs)
        metrics["training_time_s"] = time.time() - start_time

        return metrics

    def train_model_kfold_multiclass(self, device,
                                     dataset, num_epochs,
                                     k_folds, seed, hyperparam,
                                     model, patience=15):
    
        full_labels = [label for _, label in dataset.dataset.samples]
        subset_labels = [full_labels[i] for i in dataset.indices]
        class_qty = len(np.unique([label for _, label in dataset]))

        self.val_accuracies, self.val_losses = [], []
        self.train_accuracies, self.train_losses = [], []
        self.fold_precisions, self.fold_recalls, self.fold_f1s = [], [], []
        self.best_epochs = []

        kf = StratifiedKFold(n_splits=k_folds, shuffle=True,
                            random_state=seed.get_seed())

        all_val_true_list = []
        all_val_preds_list = []

        for fold, (train_idx, val_idx) in enumerate(
                kf.split(np.zeros(len(subset_labels)), subset_labels)):

            print(f"\n{'=' * 20} Fold {fold + 1}/{k_folds} {'=' * 20}")

            train_max = max(dataset.dataset.sample_max[i] for i in train_idx)

            train_subset = TransformedSubset(Subset(dataset, train_idx),
                                            get_train_transform(train_max))
            val_subset = TransformedSubset(Subset(dataset, val_idx),
                                        get_test_transform(train_max))

            dataloaders = {
                'train': DataLoader(train_subset, batch_size=64, shuffle=True,
                                    generator=seed.generator(), num_workers=0, pin_memory=True),
                'val': DataLoader(val_subset, batch_size=64, shuffle=False,
                                generator=seed.generator(), num_workers=0, pin_memory=True)
            }

            dataset_sizes = {'train': len(train_subset), 'val': len(val_subset)}

            self.__initialize_model(device, dataset, train_idx,
                                    None, hyperparam, model, class_qty)

            best_metric = -1.0
            best_epoch = 0
            epochs_no_improve = 0

            best_val_preds = None
            best_val_labels = None
            best_model_wts = copy.deepcopy(self.model.state_dict())

            best_fold_val_acc = None
            best_fold_val_loss = None
            best_fold_precision = None
            best_fold_recall = None
            best_fold_f1 = None

            best_fold_train_acc = None
            best_fold_train_loss = None

            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs} — Fold {fold + 1}")
                print("-" * 10)

                epoch_train_acc = None
                epoch_train_loss = None

                for phase in ["train", "val"]:
                    self.model.train() if phase == "train" else self.model.eval()

                    running_loss = 0.0
                    running_corrects = 0

                    all_preds = []
                    all_labels = []

                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        self.optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == "train"):
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)

                            if phase == "train":
                                loss.backward()
                                self.optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels)

                        all_preds.extend(preds.detach().cpu().numpy())
                        all_labels.extend(labels.detach().cpu().numpy())

                    if phase == "train":
                        self.scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

                    print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                    if phase == "train":
                        epoch_train_acc = epoch_acc
                        epoch_train_loss = epoch_loss

                    if phase == "val":
                        precision = precision_score(all_labels, all_preds,
                                                    average="macro", zero_division=0)
                        recall = recall_score(all_labels, all_preds,
                                            average="macro", zero_division=0)
                        f1 = f1_score(all_labels, all_preds,
                                    average="macro", zero_division=0)

                        print(f"Val Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")

                        current_metric = f1

                        if current_metric > best_metric:
                            best_metric = current_metric
                            best_epoch = epoch
                            epochs_no_improve = 0

                            best_val_preds = all_preds.copy()
                            best_val_labels = all_labels.copy()
                            best_model_wts = copy.deepcopy(self.model.state_dict())

                            best_fold_val_acc = epoch_acc
                            best_fold_val_loss = epoch_loss
                            best_fold_precision = precision
                            best_fold_recall = recall
                            best_fold_f1 = f1

                            best_fold_train_acc = epoch_train_acc
                            best_fold_train_loss = epoch_train_loss
                        else:
                            epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at fold {fold + 1}, epoch {epoch + 1}")
                    break

            print(f"\nBest F1_macro for Fold {fold + 1}: {best_metric:.4f} at epoch {best_epoch+1}")

            self.model.load_state_dict(best_model_wts)

            all_val_true_list.extend(best_val_labels)
            all_val_preds_list.extend(best_val_preds)

            self.train_accuracies.append(best_fold_train_acc)
            self.train_losses.append(best_fold_train_loss)

            self.val_accuracies.append(best_fold_val_acc)
            self.val_losses.append(best_fold_val_loss)

            self.fold_precisions.append(best_fold_precision)
            self.fold_recalls.append(best_fold_recall)
            self.fold_f1s.append(best_fold_f1)

            self.best_epochs.append(best_epoch + 1)

        metrics = self.__compute_metrics()
        metrics["all_val_true"] = np.array(all_val_true_list)
        metrics["all_val_preds"] = np.array(all_val_preds_list)
        metrics["best_epochs_per_fold"] = np.array(self.best_epochs)
        return metrics

    def __compute_metrics(self):
        metrics = {
            "mean_train_accuracy": np.mean(self.train_accuracies),
            "std_train_accuracy": np.std(self.train_accuracies),
    
            "mean_train_loss": np.mean(self.train_losses),
    
            "mean_val_accuracy": np.mean(self.val_accuracies),
            "std_val_accuracy": np.std(self.val_accuracies),
    
            "mean_val_loss": np.mean(self.val_losses),
            "std_val_loss": np.std(self.val_losses),
    
            "mean_precision": np.mean(self.fold_precisions),
            "std_precision": np.std(self.fold_precisions),
    
            "mean_recall": np.mean(self.fold_recalls),
            "std_recall": np.std(self.fold_recalls),
        }
        if hasattr(self, "fold_f1s_binary") and hasattr(self, "fold_f1s_macro"):
            metrics.update({
                "mean_f1_binary": np.mean(self.fold_f1s_binary),
                "std_f1_binary": np.std(self.fold_f1s_binary),
    
                "mean_f1_macro": np.mean(self.fold_f1s_macro),
                "std_f1_macro": np.std(self.fold_f1s_macro),
            })
        elif hasattr(self, "fold_f1s"):
            metrics.update({
                "mean_f1_macro": np.mean(self.fold_f1s),
                "std_f1_macro": np.std(self.fold_f1s),
            })
    
        return metrics

    def __initialize_model(self, device, dataset,
                           train_idx, current_class_idx,
                           hyperparam, model=resnet18_model,
                           class_qty=1):
        raw_labels = np.array([dataset[idx][1] for idx in train_idx])
        self.model = model(device, class_qty)

        if class_qty == 1:
            pos_count = int((raw_labels == current_class_idx).sum())
            neg_count = len(raw_labels) - pos_count

            if pos_count == 0:
                pos_count = 1
            if neg_count == 0:
                neg_count = 1
            pos_weight = torch.tensor([neg_count / pos_count], device=device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            class_counts = np.bincount(raw_labels)
            nonzero_mask = class_counts > 0
            class_weights = np.zeros_like(class_counts, dtype=np.float32)
            class_weights[nonzero_mask] = len(raw_labels) / (len(class_counts[nonzero_mask]) * class_counts[nonzero_mask])
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=hyperparam["lr"],
                                    weight_decay=hyperparam["wd"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=hyperparam["step"],
                                                   gamma=hyperparam["gamma"])

    def train_model_final(self, device, dataset, num_epochs, 
                          current_class_idx, seed, hyperparam,
                           model, is_binary=True):
        """ Final model training with full dataset.
        
        Args:
            device: torch device
            dataset: Full train+val dataset (combined)
            num_epochs: Fixed number of epochs (from cross-validation)
            current_class_idx: Target class index (for OVA), None for multiclass
            seed: Random seed object
            hyperparam: Best hyperparameters from Optuna
            model: Model architecture (resnet18_model or densenet121_model)
            is_binary: True for OVA, False for multiclass
        
        Returns:
            trained_model: The final trained model
            metrics: Training metrics dictionary
        """

        print("="*80)
        print("FINAL MODEL TRAINING")
        print("Training on combined train+val set with fixed epochs")
        print("="*80)
        
        # Prepare labels
        full_labels = [label for _, label in dataset.dataset.samples]
        subset_labels = [full_labels[i] for i in dataset.indices]
        
        if is_binary:
            binary_labels = [1 if label == current_class_idx else 0 for label in subset_labels]
            class_qty = 1
            pos_count = sum(binary_labels)
            neg_count = len(binary_labels) - pos_count
            print(f"\nBinary classification: Class {current_class_idx} vs Rest")
            print(f"Total samples: {len(binary_labels)}")
            print(f"  Positive: {pos_count} ({100*pos_count/len(binary_labels):.1f}%)")
            print(f"  Negative: {neg_count} ({100*neg_count/len(binary_labels):.1f}%)")
        else:
            #class_qty = len(np.unique(subset_labels))
            class_qty = len(dataset.dataset.classes)
            print(f"\nMulticlass classification: {class_qty} classes")
            print(f"Total samples: {len(subset_labels)}")
            uniq, cnts = np.unique(subset_labels, return_counts=True)
            for class_id, count in zip(uniq.tolist(), cnts.tolist()):
                count = subset_labels.count(class_id)
                print(f"  Class {class_id}: {count} ({100*count/len(subset_labels):.1f}%)")
        
        print(f"\nTraining for {num_epochs} epochs")
        print(f"Hyperparameters: {hyperparam}")
        
        # Prepare full dataset
        all_indices = list(range(len(dataset)))
        train_max = max(dataset.dataset.sample_max[i] for i in all_indices)
        
        train_subset = TransformedSubset(dataset, get_train_transform(train_max))
        train_loader = DataLoader(
            train_subset, 
            batch_size=64, 
            shuffle=True,
            generator=seed.generator(), 
            num_workers=0, 
            pin_memory=True
        )
        
        # Initialize model with full dataset
        self.__initialize_model(device, dataset, all_indices, current_class_idx, 
                            hyperparam, model, class_qty)
        
        # Training tracking
        train_acc_history = []
        train_loss_history = []
        train_f1_history = []
        train_precision_history = []
        train_recall_history = []
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")
            
            self.model.train()
            
            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            
            epoch_start = time.time()
            
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                
                if is_binary:
                    labels = (labels == current_class_idx).float().unsqueeze(1).to(device)
                else:
                    labels = labels.to(device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                if is_binary:
                    probs = torch.sigmoid(outputs)
                    loss = self.criterion(outputs, labels)
                    preds = (probs > 0.5).float()
                else:
                    loss = self.criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum()
                #running_corrects += torch.sum(preds == labels if is_binary else preds == labels.data)
                
                if is_binary:
                    all_preds.extend(preds.detach().cpu().numpy().flatten())
                    all_labels.extend(labels.detach().cpu().numpy().flatten())
                else:
                    all_preds.extend(preds.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())
            
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_subset)
            epoch_acc = (running_corrects.double() / len(train_subset)).item()

            
            all_preds = np.array(all_preds).astype(int)
            all_labels = np.array(all_labels).astype(int)
            
            if is_binary:
                precision = precision_score(all_labels, all_preds, zero_division=0)
                recall = recall_score(all_labels, all_preds, zero_division=0)
                f1 = f1_score(all_labels, all_preds, zero_division=0)
            else:
                precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            # Store history
            train_acc_history.append(epoch_acc)
            train_loss_history.append(epoch_loss)
            train_f1_history.append(f1)
            train_precision_history.append(precision)
            train_recall_history.append(recall)
            
            # Print epoch summary
            print(f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {epoch_time:.1f}s")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        time_elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Total time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Final metrics:")
        print(f"  Accuracy:  {train_acc_history[-1]:.4f}")
        print(f"  Precision: {train_precision_history[-1]:.4f}")
        print(f"  Recall:    {train_recall_history[-1]:.4f}")
        print(f"  F1-score:  {train_f1_history[-1]:.4f}")
        print("="*80)
        
        # Prepare metrics dictionary
        metrics = {
            "final_train_accuracy": train_acc_history[-1],
            "final_train_loss": train_loss_history[-1],
            "final_train_f1": train_f1_history[-1],
            "final_train_precision": train_precision_history[-1],
            "final_train_recall": train_recall_history[-1],
            "train_acc_history": train_acc_history,
            "train_loss_history": train_loss_history,
            "train_f1_history": train_f1_history,
            "train_precision_history": train_precision_history,
            "train_recall_history": train_recall_history,
            "training_time": time_elapsed,
            "total_samples": len(train_subset),
            "num_epochs": num_epochs,
            "hyperparameters": hyperparam
        }
        
        return self.model, metrics
