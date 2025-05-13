import os
import copy
import configparser
import torch
import time
import sys

import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torchvision import models, datasets
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core import MAIN_DIR, MODELS_FOLDER


PREDICTION_FOLDERPATH = os.path.join(MODELS_FOLDER, "prediction_output")
TRAINING_FOLDERPATH = os.path.join(MODELS_FOLDER, "training_output")


def resnet18_model(device, class_qty):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, class_qty)
    model = model.to(device)
    return model


class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, path


class Seed:
    def __init__(self): 
        self.seed = 2
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def seed_worker(self, worker_id):
        np.random.seed(self.seed + worker_id)
        torch.manual_seed(self.seed + worker_id)

class ModelTraining:
    def __init__(self, model):
        self.model = model
        self.best_model_params = copy.deepcopy(self.model.state_dict())
        self.loss_train = []
        self.acc_train = []
        self.loss_val = []
        self.acc_val = []
        self.writer_loss_train = None
        self.writer_loss_val = None
        self.writer_acc_train = None
        self.writer_acc_val = None  

    def train_model(self, device,
                    dataloaders, dataset_sizes,
                    num_epochs,
                    best_model_filepath):
        criterion = nn.CrossEntropyLoss()
        cfg_parser = ConfigIni()
        cfg_read_float = cfg_parser.read_cfg_float

        optimizer = optim.Adam(self.model.parameters(),
                               lr=cfg_read_float("learning_rate"),
                               weight_decay=cfg_read_float("weight_decay"))
        # Decay LR by a factor of gamma every step_size epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=cfg_read_float("step_size"),
                                              gamma=cfg_read_float("gamma"))   

        self.__writer_creation()
        start_time = time.time()
        best_acc = 0.0
        for epoch in range(num_epochs):
            print(f"-" * 10)
            print(f"Epoch {epoch}/{num_epochs - 1}\n")
            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                self.model.train() if phase == "train" else self.model.eval() 
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels, _ in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                self.__set_acc_loss(phase, epoch, epoch_acc, epoch_loss)

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    self.best_model_params = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - start_time
        print("\nTraining complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best val Acc: {:4f}".format(best_acc))
        self.__save_best_model_weights(best_model_filepath)
        self.__writer_clear()
        return self.model

    def __save_best_model_weights(self, best_model_filepath):
        # load best model weights
        self.model.load_state_dict(self.best_model_params)
        torch.save(self.model.state_dict(), os.path.join(best_model_filepath))

    def __writer_creation(self):
        self.writer_loss_train = self.writer_acc_train = SummaryWriter(os.path.join(TRAINING_FOLDERPATH,
                                                                                    "tb_summary",
                                                                                    "train"))
        self.writer_loss_val = self.writer_acc_val = SummaryWriter(os.path.join(TRAINING_FOLDERPATH,
                                                                                "tb_summary",
                                                                                "validation"))
    
    def __writer_clear(self):
        self.writer_loss_train.flush()
        self.writer_loss_train.close()
        self.writer_loss_val.flush()
        self.writer_loss_val.close()
        self.writer_acc_train.flush()
        self.writer_acc_train.close()
        self.writer_acc_val.flush()
        self.writer_acc_val.close()

    def __set_acc_loss(self, phase, epoch, epoch_acc, epoch_loss):
        if phase == "train":
            self.writer_loss_train.add_scalar("Loss x Epoch", epoch_loss, epoch)
            self.writer_acc_train.add_scalar("Accuracy x Epoch", epoch_acc, epoch)
            self.loss_train.append(epoch_loss)
            self.acc_train.append(float(epoch_acc))
        elif phase == "val":
            self.writer_loss_val.add_scalar("Loss x Epoch", epoch_loss, epoch)
            self.writer_acc_val.add_scalar("Accuracy x Epoch", epoch_acc, epoch)
            self.loss_val.append(epoch_loss)
            self.acc_val.append(float(epoch_acc))
    
    def get_acc_loss(self):
        return self.acc_train, self.acc_val, self.loss_train, self.loss_val


class ConfigIni:
    def __init__(self):
        config_file_path = os.path.join(MODELS_FOLDER, "config.ini")
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path)

    def read_cfg_float(self, key, section="parameters"):
        return self.config.getfloat(section, key)


def predict_model(device, model, label_list, dataloader,
                  folder_save_pred=None):
    model.eval()
    predicted = []
    label = []

    with torch.no_grad():
        for inputs, labels, filename in dataloader:
            inputs_device = inputs.to(device)
            labels_device = labels.to(device)
            outputs = model(inputs_device)

            _, preds = torch.max(outputs, 1)

            batch_size = inputs_device.size()[0]
            
            for idx in range(batch_size):
                curr_label = int(labels_device[idx])
                curr_pred = int(preds[idx])

                predicted.append(label_list[curr_pred])
                label.append(label_list[curr_label])

                if folder_save_pred:
                    print(f"Saving image {idx}/{int(batch_size)}", end='\r')
                    save_image(inputs_device[idx],
                               os.path.join(folder_save_pred, label_list[curr_pred],
                                            Path(str(filename)).stem + ".png"))

    return predicted, label


