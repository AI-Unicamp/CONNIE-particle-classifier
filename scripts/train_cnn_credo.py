# Model training with CREDO image dataset

import os
import sys
import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from collections import Counter
from torchsummary import summary
from torchvision import transforms
from sklearn import metrics

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core import DATA_FOLDER
from scripts.credo_training_utils import TRAINING_FOLDERPATH,\
    ModelTraining, ImageFolderWithPath, Seed,\
    resnet18_model, predict_model


IMG_SIZE_CONFIG = (60, 60)
FOLDERS_LIST_CONFIG = ["train", "val", "test"]
BATCH_SIZE_CONFIG = 64
NUM_WORKERS_CONFIG = 2
NUM_EPOCHS_CONFIG = 150
MODEL_SUMMARY_INPUT_SHAPE = (3, 64, 64)


def imshow(input_img, title=None):
    """ Imshow for Tensor """
    if isinstance(input_img, torch.Tensor):
        input_img = input_img.numpy()
    img = np.asarray(input_img).transpose((1, 2, 0))
    plt.imshow(img, vmin=0, vmax=5 if img.max() > 1 else 1)
    if title:
        plt.title(title)
    plt.pause(0.001) # pause a bit to update plots


def initialize_environment():
    """Sets up the device and random seed."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seed_obj = Seed()
    return device, seed_obj


def define_and_create_paths(base_data_folder, base_training_folder):
    """Defines and creates necessary directory paths."""
    processed_data_folder = os.path.join(base_data_folder, "credo_processed_dataset")
    model_data_folder = os.path.join(base_training_folder, "best_model_weight")
    metrics_folder = os.path.join(base_training_folder, "metrics")
    os.makedirs(model_data_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)
    best_model_filepath = os.path.join(model_data_folder, "best_model_params.pt")
    return processed_data_folder, best_model_filepath, metrics_folder


def get_data_transformations(img_size):
    """Defines and returns data transformations for train, val, and test sets."""
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0, 360), fill=(0,)),
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ]),
        "val": transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0, 360), fill=(0,)),
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ]),
        "test": transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ])
    }
    return data_transforms


def prepare_dataloaders(processed_data_path, data_transforms_dict, folders_list,
                        batch_size_val, num_workers_val, seed_worker_fn):
    """Loads datasets and creates dataloaders."""
    image_datasets = {
        x: ImageFolderWithPath(os.path.join(processed_data_path, x), data_transforms_dict[x])
        for x in folders_list
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size_val,
            shuffle=True if x == "train" else False,
            num_workers=num_workers_val,
            worker_init_fn=seed_worker_fn
        ) for x in folders_list
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in folders_list}
    class_names = image_datasets["train"].classes
    dataset_class_qty = {x: dict(Counter(image_datasets[x].targets)) for x in folders_list}
    class_qty = len(class_names)

    print(f"Class quantity: {class_qty}")
    print(f"Class names: {class_names}")
    for i in folders_list:
        print(f"{i} dataset size: {dataset_sizes[i]}, class distribution: {dataset_class_qty[i]}")
    return dataloaders, dataset_sizes, class_names, class_qty


def build_and_summarize_model(device_to_use, num_classes, input_shape_for_summary):
    """Initializes the ResNet18 model and prints its summary."""
    model = resnet18_model(device_to_use, num_classes)
    print("\nModel Architecture:")
    print(model)
    print("\nModel Summary:")
    summary(model, input_shape_for_summary)
    return model


def run_model_training(model_to_train, device_to_use, dataloaders_dict,
                       dataset_sizes_dict, num_epochs_to_train, best_model_save_path):
    """Manages the model training process."""
    print("\nStarting Model Training...")
    model_training_util = ModelTraining(model_to_train)
    trained_model = model_training_util.train_model(
        device_to_use,
        dataloaders_dict,
        dataset_sizes_dict,
        num_epochs_to_train,
        best_model_save_path
    )
    acc_train, acc_val, loss_train, loss_val = model_training_util.get_acc_loss()
    print("Training complete.")
    return trained_model, acc_train, acc_val, loss_train, loss_val


def plot_training_history(acc_train_hist, acc_val_hist, loss_train_hist, loss_val_hist, num_epochs_val):
    """Plots training and validation accuracy and loss curves."""
    print("\nPlotting Training History...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_train_hist, label="Train Loss")
    plt.plot(loss_val_hist, label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.xlim(0, num_epochs_val -1)
    plt.grid(True)
    plt.title("Loss vs. Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(acc_train_hist, label="Train Accuracy")
    plt.plot(acc_val_hist, label="Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.xlim(0, num_epochs_val -1)
    plt.grid(True)
    plt.title("Accuracy vs. Epochs")
    plt.tight_layout()
    plt.show()


def evaluate_model_performance(device_to_use, num_classes, saved_model_path,
                               class_names_list, test_dataloader, metrics_output_folder):
    """Loads the trained model and evaluates its performance on the test set."""
    print("\nEvaluating Model Performance on Test Set...")
    # Load the best model
    evaluation_model = resnet18_model(device_to_use, num_classes)
    evaluation_model.load_state_dict(torch.load(saved_model_path, map_location=device_to_use))

    predicted_labels, true_labels = predict_model(device_to_use, evaluation_model,
                                                  class_names_list, test_dataloader)

    # Calculate metrics
    test_accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    test_precision = metrics.precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
    test_recall = metrics.recall_score(true_labels, predicted_labels, average="macro", zero_division=0)
    test_bal_accuracy = metrics.balanced_accuracy_score(true_labels, predicted_labels)
    test_f1 = metrics.f1_score(true_labels, predicted_labels, average="macro", zero_division=0)

    print("\nTest Metrics:")
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"\nTest Precision (Macro): {test_precision:.4f}")
    print(f"\nTest Recall (Macro): {test_recall:.4f}")
    print(f"\nTest Balanced Accuracy: {test_bal_accuracy:.4f}")
    print(f"\nTest F1-Score (Macro): {test_f1:.4f}")

    # Save metrics to file
    metrics_filepath = os.path.join(metrics_output_folder, "metrics.txt")
    with open(metrics_filepath, "w") as metrics_txt:
        metrics_txt.write(f"Test Accuracy\t {test_accuracy:.4f}\n")
        metrics_txt.write(f"Test Precision (Macro)\t {test_precision:.4f}\n")
        metrics_txt.write(f"Test Recall (Macro)\t {test_recall:.4f}\n")
        metrics_txt.write(f"Test Balanced Accuracy\t {test_bal_accuracy:.4f}\n")
        metrics_txt.write(f"Test F1-Score (Macro):\t {test_f1:.4f}\n")
    print(f"\nMetrics saved to {metrics_filepath}")

    # Confusion Matrix
    print("\nConfusion Matrix - Test data")
    confusion_mtx = metrics.confusion_matrix(true_labels, predicted_labels)
    print(confusion_mtx)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_mtx, display_labels=class_names_list)
    cm_display.plot()
    plt.title("Confusion Matrix - Test data")
    plt.grid(False)
    cm_path = os.path.join(metrics_output_folder, "confusion_mtx.png")
    plt.savefig(cm_path)
    plt.show()
    print(f"Confusion matrix plot saved to {cm_path}")

    # Classification Report
    print("\nClassification Report - Test data")
    try:
        # Ensure target_names matches the unique labels present in true_labels and predicted_labels if they are subset of class_names_list
        unique_labels_in_data = sorted(list(set(true_labels + predicted_labels)))
        current_target_names = [class_names_list[i] for i in unique_labels_in_data if i < len(class_names_list)]
        
        classification_report_dict = metrics.classification_report(
            true_labels, predicted_labels,
            labels=unique_labels_in_data, # Use labels present in the data for report
            target_names=current_target_names, # Ensure target_names align with labels
            zero_division=0, output_dict=True
        )
        # Filter out average rows for heatmap if they exist
        report_df = pd.DataFrame(classification_report_dict)
        if 'accuracy' in report_df.columns: # if accuracy is a column, it's not what we want for heatmap items
             report_df_filtered = report_df.iloc[:-1, :-1] # Adjust if 'accuracy' is present as a column rather than just a summary row
        else: # Standard output_dict format has 'accuracy' as a separate key, or 'macro avg', 'weighted avg' as rows
            report_df_filtered = report_df.drop(columns=[col for col in ['accuracy', 'macro avg', 'weighted avg'] if col in report_df.columns], errors='ignore')
            report_df_filtered = report_df_filtered.T # Transpose to have classes as rows
            # Remove summary rows if they are still there after transpose
            report_df_filtered = report_df_filtered.drop(index=[idx for idx in ['accuracy', 'macro avg', 'weighted avg'] if idx in report_df_filtered.index], errors='ignore')


        plt.figure(figsize=(10, max(5, len(current_target_names) * 0.5))) # Adjust figure size
        sns.heatmap(report_df_filtered, annot=True, fmt=".2f", cmap="viridis") # Use filtered DataFrame
        plt.title("Classification report - Test data")
        cr_path = os.path.join(metrics_output_folder, "classification_report.png")
        plt.savefig(cr_path, bbox_inches="tight")
        plt.show()
        print(f"Classification report plot saved to {cr_path}")

        # Save full classification report text
        classification_report_text = metrics.classification_report(
            true_labels, predicted_labels,
            labels=unique_labels_in_data,
            target_names=current_target_names,
            zero_division=0
        )
        print(classification_report_text)
        with open(os.path.join(metrics_output_folder, "classification_report.txt"), "w") as cr_file:
            cr_file.write(classification_report_text)
        print(f"Classification report text saved to {os.path.join(metrics_output_folder, 'classification_report.txt')}")


    except ValueError as e:
        print(f"Could not generate classification report heatmap: {e}")
        print("Classification report (text only):")
        print(metrics.classification_report(true_labels, predicted_labels, target_names=class_names_list, zero_division=0))


def main():
    print("--- Starting CREDO Image Model Training ---")
    device, seed_obj = initialize_environment()
    processed_data_folder, best_model_filepath, metrics_folder = define_and_create_paths(
        DATA_FOLDER, TRAINING_FOLDERPATH
    )

    data_transforms = get_data_transformations(IMG_SIZE_CONFIG)

    dataloaders, dataset_sizes, class_names, class_qty = prepare_dataloaders(
        processed_data_folder,
        data_transforms,
        FOLDERS_LIST_CONFIG,
        BATCH_SIZE_CONFIG,
        NUM_WORKERS_CONFIG,
        seed_obj.seed_worker
    )


    model = build_and_summarize_model(device, class_qty, MODEL_SUMMARY_INPUT_SHAPE)

    trained_model, acc_train, acc_val, loss_train, loss_val = run_model_training(
        model,
        device,
        dataloaders,
        dataset_sizes,
        NUM_EPOCHS_CONFIG,
        best_model_filepath
    )

    if acc_train and acc_val and loss_train and loss_val: # Ensure lists are not empty
        plot_training_history(acc_train, acc_val, loss_train, loss_val, NUM_EPOCHS_CONFIG)
    else:
        print("Training history data not available for plotting.")


    if os.path.exists(best_model_filepath):
        evaluate_model_performance(
            device,
            class_qty,
            best_model_filepath,
            class_names,
            dataloaders["test"], 
            metrics_folder
        )
    else:
        print(f"Best model file not found at {best_model_filepath}. Skipping evaluation.")

    print("\n--- CREDO Image Model Training Finished ---")

if __name__ == "__main__":
    main()