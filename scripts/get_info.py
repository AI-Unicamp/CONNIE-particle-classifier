import os
import sys
import uproot
import random
from pathlib import Path
from glob import glob
from datetime import datetime
from scripts.database import Database
from PyQt5.QtWidgets import QMessageBox

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core import CATALOG_FOLDERPATH


class RootFileInfo:
    def __init__(self):
        self.file = None
        self.file_data = None
        self.total_num_events = None
        self.available_idx = None
        self.count = 0
        self.num_user_relabel = 3
        self.current_filepath = ""
        self.all_root_files = glob(os.path.join(CATALOG_FOLDERPATH, "*.root"))
        self.database = Database()
        self.current_username = get_username()

    def __get_random_root_file(self):
        """ Get a random root file in the catalog folder"""
        if not self.all_root_files:
            raise Exception(f"There is no root file in the folder {CATALOG_FOLDERPATH}")
        self.current_filepath = random.choice(self.all_root_files)
        self.current_filename = Path(self.current_filepath).name

    def open_root_file(self):
        """ Open the root file """
        branch_name = "hitSumm"    # ROOT branch
        # Open ROOT file and acess "hitSumm"
        root_file = uproot.open(self.current_filepath)
        self.file = root_file[branch_name]

    def __get_new_root_file(self):
        """ Get a unlabeled root file in the catalog folder """
        other_files = [file for file in self.all_root_files if
                       self.current_filepath not in file]
        try:
            self.current_filepath = random.choice(other_files)
            self.current_filename = Path(self.current_filepath).name
            self.open_root_file()
            self.total_num_events = self.file.num_entries
            self.relabel_idx = self.__get_idx_relabel()
            self.__get_available_idx()
        except IndexError:
            self.no_more_files()

    def no_more_files(self):
        """No more files available message """
        print("No more root files available to be labeled.")
       
        QMessageBox.warning(
                None,
                "Warning",
                f"No more root files available to be labeled")
        sys.exit(1)

    def __get_available_idx(self):
        """ Get available indices """
        curr_file_labeled_events = self.database.search_events(condition=f"filename=\"{self.current_filename}\"")
        all_idx = list(range(self.total_num_events))
        if len(curr_file_labeled_events) > 0:
            self.available_idx = []
            remove_idx = []
            curr_file_labels = list(zip(*set(curr_file_labeled_events)))[1] 
            for event in curr_file_labeled_events:
                label_idx = event[1]
                username = event[2]

                # Condition 1: If not for relabeling, or the current user already labeled the idx, remove the index
                if ((label_idx not in self.relabel_idx) or
                    (username == self.current_username)):
                    remove_idx.append(label_idx)
                # Condition 2: If relabeling and idx already labeled num_user_relabel times, remove the index
                elif ((label_idx in self.relabel_idx) and
                      (curr_file_labels.count(label_idx) >= self.num_user_relabel)):
                    remove_idx.append(label_idx)

            self.available_idx = [idx for idx in all_idx
                                  if (idx not in remove_idx)]
        else:
            self.available_idx = all_idx

    def open_unlabeled_root_file(self):
        """ Open unlabeled root file """
        self.__get_random_root_file()
        self.open_root_file()
        self.total_num_events = self.file.num_entries
        self.relabel_idx = self.__get_idx_relabel()
        self.__get_available_idx()
        keep_searching = True

        while keep_searching:
            if (not self.available_idx and self.count < len(self.all_root_files)):
                # File is fully labeled, get a new root file
                self.__get_new_root_file()
                keep_searching = True
                self.count+=1

            elif self.count == len(self.all_root_files):
                self.no_more_files()
            else:
                keep_searching = False

    def __get_idx_relabel(self):
        """ Get pseudorandom indexes to be labeled by the same users
        See Annotation Redundancy with Targeted Quality Assurance method
        """
        random.seed(42)
        percentage = int(self.total_num_events * 0.1)
        pseudo_random_idx = random.sample(range(self.total_num_events), percentage)
        random.seed(None)
        return pseudo_random_idx

    def get_current_filepath(self):
        """ Get current filepath """
        return self.current_filepath

    def get_root_file_info(self):
        """ Extracting only the necessary info for the interface """
        info_keys_list = ["xPix","yPix", "ePix", "runID", "imgID", "chid",
                          "skpID", "EventID", "E0", "nSavedPix",
                          "xBary0", "yBary0", "xMin", "xMax", "yMin", "yMax"]
        self.file_data = self.file.arrays(info_keys_list, library="np")
        return self.file_data
    
    def get_new_img_idx(self, skip:bool = False):
        """ Get a new image idx

        Args:
            skip (bool): if the current indice was skipped

        Returns:
            int: indice to be labeled
        """
        candidate_indices = self.__get_candidate_indices(skip)
        if not candidate_indices:
            print(f"All events from the file {self.current_filename}"
                  + " are labeled. Searching for new file.\n")
            QMessageBox.information(
                None,
                "Information",
                f"All events from the file {self.current_filename} are labeled."
                + "\n Searching for new file.")
            self.open_unlabeled_root_file()
            self.get_root_file_info()
            candidate_indices = self.__get_candidate_indices(False)
        new_idx = random.choice(candidate_indices)
        while not self.__check_idx_is_valid(new_idx):
            new_idx = random.choice(candidate_indices)
        return new_idx

    def __get_candidate_indices(self, skip:bool):
        """ Get candidates indices

        Args:
            skip (bool): if the current indice was skipped

        Returns:
            list: new candidate indices to be labeled
        """
        if any(idx in self.relabel_idx for idx in self.available_idx):
            candidate_indices = [idx for idx in self.relabel_idx
                                 if idx in self.available_idx]
            if skip and len(candidate_indices) == 1:
                candidate_indices = self.available_idx
        else:
            candidate_indices = self.available_idx
        return candidate_indices

    def __check_idx_is_valid(self, new_idx: int):
        """ Check if current index is valid to be labeled.
        If it is not, remove it from the available index list

        Args:
            new_idx (int): image index

        Returns:
            bool: True if the image is valid, False if it is not
        """
        label_status = True
        is_event_labeled = (
            self.database.search_events(
            condition=f"filename=\"{self.current_filename}\" AND img_idx={new_idx}"))
        
        is_event_labeled_same_user = self.database.search_events(
            condition=f"filename=\"{self.current_filename}\" AND "
                      + f"img_idx={new_idx} AND username=\"{self.current_username}\"")
  
        if len(is_event_labeled):
            if (new_idx in self.relabel_idx and
                len(is_event_labeled) < self.num_user_relabel and
                len(is_event_labeled_same_user) == 0):
                pass
            else:
                self.remove_idx(new_idx)
                label_status = False
        return label_status

    def remove_idx(self, idx: int):
        """Remove indice from the list to be labeled

        Args:
            idx (int): indice to be removed
        """
        self.available_idx.remove(idx)
        if idx in self.relabel_idx:
            self.relabel_idx.remove(idx)
    
def get_username():
    """ Get current username """
    return os.getlogin()


def get_datetime():
    """ Get current date and time """
    date_time = datetime.now()
    return date_time.strftime("%d/%m/%Y %H:%M:%S")
