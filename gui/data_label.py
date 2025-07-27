import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core import ClassLabel, REVIEW_MODE_PASSWORD, REVIEW_USERNAME
from scripts.plot import PlotCanvas
from scripts.database import Database, ImageData
from scripts.get_info import RootFileInfo, get_username, get_datetime
from pathlib import Path
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout,\
     QHBoxLayout, QRadioButton, QPushButton, QLabel, QGroupBox, QButtonGroup,\
     QMessageBox, QGridLayout, QAction, QInputDialog, QLineEdit, QTextEdit


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.create_menu()
        self.version = "1.0"
        self.is_review_mode = False
        self.username = get_username()

        self.license = self.read_license()

        self.database = Database()        
        self.database.create_database()

        self.root_file = RootFileInfo()
        self.root_file.open_unlabeled_root_file()

        self.title_str = "CONNIE Data Label"
        self.setWindowTitle(self.title_str + " - Annotation mode")

        # Main widget
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        self.layout = QGridLayout()

        # Title
        page_title = QLabel(self.title_str, self)
        page_title.setAlignment(Qt.AlignHCenter)
        page_title.setStyleSheet("QLabel"
                                 "{"
                                 "font : 30px Arial;"
                                 "font-weight: bold;"
                                 "}")
        self.layout.addWidget(page_title, 0, 0, 1, 4)

        self.get_new_file_information()
        
        # Plot canvas
        self.canvas = PlotCanvas(self, self.file_data, self.img_idx)
        self.layout.addWidget(self.canvas, 1, 0, 3, 3)

        self.file_information()
        self.event_information()        
        self.event_classes()
        self.decision_buttons()

        self.layout.addWidget(self.file_info_group, 1, 3)
        self.layout.addWidget(self.event_info_group, 2, 3)
        self.layout.addWidget(self.classes_group, 3, 3)
        self.layout.addWidget(self.buttons_group, 4, 3)
        main_widget.setLayout(self.layout)
        self.show()

    def file_information(self):
        """ Get file information """
        self.curr_filepath = QLabel(f"Root file path: {self.filepath}")
        self.curr_filepath.setWordWrap(True)
        self.curr_filepath.setStyleSheet("QLabel"
                                         "{"
                                         "font : 15px Arial;"
                                         "}")
        self.curr_filename = QLabel(f"Root file: {self.filename}")
        self.curr_filename.setStyleSheet("QLabel"
                                         "{"
                                         "font : 15px Arial;"
                                         "}")
        self.label_run = QLabel(f"Run: {self.run_id}")
        self.label_run.setStyleSheet("QLabel"
                                     "{"
                                     "font : 15px Arial;"
                                     "}")
        self.label_skipper_id = QLabel(f"Skipper ID: {self.skipper_id}")
        self.label_skipper_id.setStyleSheet("QLabel"
                                             "{"
                                             "font : 15px Arial;"
                                             "}")
        self.label_index = QLabel(f"Index: {self.img_idx}")
        self.label_index.setStyleSheet("QLabel"
                                       "{"
                                       "font : 15px Arial;"
                                       "}")
        self.label_image_id = QLabel(f"Image ID: {self.img_id}")
        self.label_image_id.setStyleSheet("QLabel"
                                          "{"
                                          "font : 15px Arial;"
                                          "}")

                
        self.file_info_group = QGroupBox("File Information")
        self.file_info_layout = QVBoxLayout()
        self.file_info_layout.addWidget(self.curr_filepath)
        self.file_info_layout.addWidget(self.curr_filename)
        self.file_info_layout.addWidget(self.label_run)
        self.file_info_layout.addWidget(self.label_skipper_id)
        self.file_info_layout.addWidget(self.label_index)
        self.file_info_layout.addWidget(self.label_image_id)

        self.file_info_layout.addStretch(1)
        self.file_info_group.setLayout(self.file_info_layout)

    def event_information(self):
        """ Get event information """
        self.label_pixels_number = QLabel(f"Total number of pixels: {self.pixels_number}")
        self.label_pixels_number.setStyleSheet("QLabel"
                                               "{"
                                               "font : 15px Arial;"
                                               "}")
        self.label_total_energy = QLabel(f"Total energy: {self.total_energy} eV")
        self.label_total_energy.setStyleSheet("QLabel"
                                              "{"
                                              "font : 15px Arial;"
                                              "}")
        
        self.label_x_bary0 = QLabel(f"X-Barycenter 0: {self.x_bary0}")
        self.label_x_bary0.setStyleSheet("QLabel"
                                         "{"
                                         "font : 15px Arial;"
                                         "}")
        self.label_y_bary0 = QLabel(f"Y-Barycenter 0: {self.y_bary0}")
        self.label_y_bary0.setStyleSheet("QLabel"
                                         "{"
                                         "font : 15px Arial;"
                                         "}")
        self.event_info_group = QGroupBox("Event Information")
        self.event_info_layout = QVBoxLayout()
        self.event_info_layout.addWidget(self.label_pixels_number)
        self.event_info_layout.addWidget(self.label_total_energy)
        self.event_info_layout.addWidget(self.label_x_bary0)
        self.event_info_layout.addWidget(self.label_y_bary0)

        self.event_info_layout.addStretch(1)
        self.event_info_group.setLayout(self.event_info_layout)

    def event_classes(self):
        """Create label radio buttons for all available classes."""
        label_definitions = {
            "Muon": (ClassLabel.Muon, QRadioButton("Muon")),
            "Electron": (ClassLabel.Electron, QRadioButton("Electron")),
            "Blob": (ClassLabel.Blob, QRadioButton("Blob")),
            "Diffusion hit (blob < 600 eV)": (
                ClassLabel.Diffusion_Hit,
                QRadioButton("Diffusion hit (blob < 600 eV)")),
            "Alpha": (ClassLabel.Alpha, QRadioButton("Alpha")),
            "Others": (ClassLabel.Others, QRadioButton("Others")),
        }

        for _, (_, btn) in label_definitions.items():
            btn.setStyleSheet("QRadioButton { font: 15px Arial; }")

        self.classes_buttom_layout = QButtonGroup()
        self.classes_group = QGroupBox("Event classification")
        layout = QVBoxLayout()

        for label_text, (label_enum, radio_button) in label_definitions.items():
            self.classes_buttom_layout.addButton(radio_button,
                                                 id=label_enum.value)
            layout.addWidget(radio_button)

        layout.addStretch(1)
        self.classes_group.setLayout(layout)

    def clear_class_selection(self):
        self.classes_buttom_layout.setExclusive(False)
        checked_button = self.classes_buttom_layout.checkedButton()
        if checked_button:
            checked_button.setChecked(False)
        self.classes_buttom_layout.setExclusive(True)

    def create_previous_labels_box(self, label_dict: dict):
        """Create a read-only text box listing previously selected labels.

        Args:
            label_dict (dict): Labels previously
            selected by other users.
        """
        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setStyleSheet("QTextEdit { font: 13px Arial; }")
        self.text_box.setFixedSize(350, 60)
        box_text = "\n   - " + "\n   - ".join(f"{label} x{count}"
                                        for label, count in
                                        label_dict.items())
        self.text_box.setText("Previously selected labels:" + box_text)
        self.layout.addWidget(self.text_box, 4, 0,
                              alignment=Qt.AlignLeft)

    def create_menu(self):
        """ Create menu tab """
        menubar = self.menuBar()
        review_menu = menubar.addMenu("Review")
        help_menu = menubar.addMenu("Help")

        self.resolve_action = QAction("Resolve Annotation Conflicts", self)
        self.resolve_action.triggered.connect(self.enter_review_mode)
        review_menu.addAction(self.resolve_action)
        info_action = QAction("Info", self)
        info_action.triggered.connect(self.show_info)
        help_menu.addAction(info_action)

    def decision_buttons(self):
        """ Decision buttons """
        self.buttons_group = QGroupBox()
        self.buttons_group.setStyleSheet("QGroupBox { border: 1px;}")

        buttons_layout = QHBoxLayout()
        self.button_skip = QPushButton("Skip")
        self.button_submit = QPushButton("Submit")
        self.button_submit.clicked.connect(self.submit_info)
        self.button_skip.clicked.connect(self.skip_image)
        buttons_layout.addWidget(self.button_skip)
        buttons_layout.addWidget(self.button_submit)
        self.buttons_group.setLayout(buttons_layout)

    def submit_info(self):
        """ Submit buttom """
        self.block_buttons()
        buttom_id = self.classes_buttom_layout.checkedId()
        if buttom_id < 0:
            QMessageBox.warning(self, 
                                "Invalid Entry",
                                "Checkbox cannot be empty")
            self.release_buttons()
            return
        self.clear_class_selection()
        curr_class = ClassLabel(buttom_id).name.replace("_", " ")
        image_data = ImageData(Path(self.filename).name, self.img_idx,
                               self.username, get_datetime(),
                               curr_class)
        print(f"image_data = {image_data}\n")
        self.database.insert_event_info(image_data)
        if not self.is_review_mode:
            self.root_file.remove_idx(self.img_idx)
        self.get_new_file_information()
        self.update_file_info()
        self.release_buttons()

    def skip_image(self):
        """ Skip buttom """
        self.block_buttons()
        self.clear_class_selection()
        self.get_new_file_information(skip=True)
        self.update_file_info()
        self.release_buttons()

    def block_buttons(self):
        """ Block skip and submit buttons """
        self.button_skip.setEnabled(False)
        self.button_submit.setEnabled(False)
        QApplication.processEvents()

    def release_buttons(self):
        """ Release skip and submit buttons """
        QApplication.processEvents()
        self.button_skip.setEnabled(True)
        self.button_submit.setEnabled(True)

    def get_new_file_information(self, skip=False):
        """ Get new file information """
        if not self.is_review_mode:
            self.img_idx = self.root_file.get_new_img_idx(skip)
        else:
            self.filename, self.img_idx = self.root_file.get_discrepant_event_idx(skip)
            previous_labels = self.database.get_labels_for_event(self.filename, self.img_idx)
            self.create_previous_labels_box(previous_labels)
        self.file_data = self.root_file.get_root_file_info()
        self.run_id = str(self.file_data["runID"][self.img_idx])
        self.img_id = str(self.file_data["imgID"][self.img_idx])
        self.pixels_number = str(self.file_data["nSavedPix"][self.img_idx])
        self.total_energy = str(round(self.file_data["E0"]
                                      [self.img_idx]*3.745))
        self.skipper_id = str(self.file_data["skpID"][self.img_idx])
        self.x_bary0 = str(round(self.file_data["xBary0"][self.img_idx]))
        self.y_bary0 = str(round(self.file_data["yBary0"][self.img_idx]))
        complete_filepath = self.root_file.get_current_filepath()
        self.filename = Path(complete_filepath).name
        self.filepath = Path(complete_filepath).parent

    def update_file_info(self):
        """ Update file information """
        self.curr_filepath.setText(f"Root file path: {self.filepath}")
        self.curr_filename.setText(f"Root file: {self.filename}")
        self.label_run.setText(f"Run: {self.run_id}")
        self.label_skipper_id.setText(f"Skipper ID: {self.skipper_id}")
        self.label_index.setText(f"Index: {self.img_idx}")
        self.label_image_id.setText(f"Image ID: {self.img_id}")
        self.label_pixels_number.setText(f"Total number of pixels: {self.pixels_number}")
        self.label_total_energy.setText(f"Total energy: {self.total_energy} eV")
        self.label_x_bary0.setText(f"X-Barycenter 0: {self.x_bary0}")
        self.label_y_bary0.setText(f"Y-Barycenter 0: {self.y_bary0}")
        self.canvas.plot(self.file_data, self.img_idx)

    def enter_review_mode(self):
        """ Enter review mode """
        password, ok = QInputDialog.getText(self,
                                     "Enter Password",
                                     "Password:",
                                     echo=QLineEdit.Password)
        if ok:
            if password == REVIEW_MODE_PASSWORD:
                self.activate_review_mode()
            else:
                QMessageBox.warning(self, "Access Denied", "Incorrect password.")

    def activate_review_mode(self):
        """ Activates review mode to analyze events with discrepancies """
        self.is_review_mode = True
        self.username = REVIEW_USERNAME
        self.resolve_action.setEnabled(False)
        self.setWindowTitle(self.title_str + " - Review Mode")

        self.button_resolve_majority = QPushButton("Resolve by majority")
        self.button_resolve_majority.setStyleSheet("QPushButton { font-size: 10px; }")
        self.button_resolve_majority.setFixedSize(110, 20)
        self.button_resolve_majority.clicked.connect(self.resolve_by_majority)

        self.button_exit_review = QPushButton("Exit review mode")
        self.button_exit_review.setStyleSheet("QPushButton { font-size: 10px; }")
        self.button_exit_review.setFixedSize(110, 20)
        self.button_exit_review.clicked.connect(self.exit_review_mode)
        
        self.button_container = QWidget()
        vbox = QVBoxLayout(self.button_container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(2)
        vbox.addWidget(self.button_resolve_majority)
        vbox.addWidget(self.button_exit_review)
        self.layout.addWidget(self.button_container, 0, 0, alignment=Qt.AlignLeft)

        self.get_new_file_information()
        self.update_file_info()

    def resolve_by_majority(self):
        """ Checks for events with label conflicts, and if the
        majority of users agree on a label, inserts it in the dataset
        with the review user """

        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            events = self.database.get_discrepant_label_votes_by_event()
            resolved_count = 0

            for (filename, img_idx), label_counts in events.items():
                majority_label = self.root_file.resolve_majority_label(label_counts)
                if majority_label is not None:
                    image_data = ImageData(filename, img_idx,
                                           REVIEW_USERNAME, get_datetime(),
                                           majority_label)
                    self.database.insert_event_info(image_data)
                    resolved_count += 1
        finally:
            QApplication.restoreOverrideCursor()

        if resolved_count > 0:
            QMessageBox.information(
                self,
                "Conflicts resolved by majority",
                f"{resolved_count} event(s) were successfully resolved based on majority vote."
            )
        else:
            QMessageBox.information(
                self,
                "No resolvable conflicts",
                "No events had majority. Need to analyse manually."
            )
        self.get_new_file_information()
        self.update_file_info()

    def exit_review_mode(self):
        """ Exit review mode and return to annotation mode """
        confirm = QMessageBox.question(
            self,
            "Exit review mode",
            "Are you sure you want to return to annotation mode?",
            QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.is_review_mode = False
            self.username = get_username()
            self.root_file.clear_discrepant_events_idx()
            self.resolve_action.setEnabled(True)
            if hasattr(self, "text_box"):
                self.layout.removeWidget(self.text_box)
                self.text_box.deleteLater()
                del self.text_box
            self.layout.removeWidget(self.button_container)
            self.button_container.hide()
            self.button_container.deleteLater()

            self.get_new_file_information()
            self.update_file_info()
            self.setWindowTitle(self.title_str + " - Annotation mode")
            QTimer.singleShot(0, lambda: QMessageBox.information(
                self,
                "Annotation Mode",
                "You have returned to annotation mode."
            ))

    def show_info(self):
        """ Show information about software """
        info_text = (
            f"CONNIE Event Labeling Tool\n"
            f"Version: {self.version}\n"
            f"© 2025\n"
            f"License: {self.license}\n\n"
            "This interface is designed for labeling events in the CONNIE experiment.\n"
            "Developed as part of research with the CONNIE Collaboration.\n"
            "Special thanks to collaborators from UNICAMP, UFRJ, and associated institutions.\n\n"
            "—\n\n"
            "Contact:\n"
            "Sara Mirthis Dantas dos Santos\n"
            "Dept. of Computer Engineering and Automation (DCA)\n"
            "Universidade Estadual de Campinas (UNICAMP)\n"
            "s224018@dac.unicamp.br | saramirthis@gmail.com"
        )

        QMessageBox.information(self, "About This Tool", info_text)

    def read_license(self):
        """ Read license """
        try:
            with open(os.path.join(ROOT_DIR, "LICENSE"), "r") as f:
                return f.readline().strip()
        except Exception:
            return "Unknown"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
