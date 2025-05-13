import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core import ClassLabel
from scripts.plot import PlotCanvas
from scripts.database import Database, ImageData
from scripts.get_info import RootFileInfo, get_username, get_datetime
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout,\
     QHBoxLayout, QRadioButton, QPushButton, QLabel, QButtonGroup, QGroupBox,\
     QMessageBox, QGridLayout



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.database = Database()        
        self.database.create_database()

        self.root_file = RootFileInfo()
        self.root_file.open_unlabeled_root_file()

        title_str = "CONNIE Data Label"
        self.setWindowTitle(title_str)

        # Main widget
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        self.layout = QGridLayout()

        # Title
        page_title = QLabel(title_str, self)
        page_title.setAlignment(Qt.AlignHCenter)
        page_title.setStyleSheet("QLabel"
                                 "{"
                                 "font : 30px Arial;"
                                 "font-weight: bold;"
                                 "}")
        self.layout.addWidget(page_title, 0, 0, 1, 4)

        self.get_new_file_information()

        # File information
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

        # Event information
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
        
        # Plot canvas
        self.canvas = PlotCanvas(self, self.file_data, self.img_idx)
        self.layout.addWidget(self.canvas, 1, 0, 3, 3)

        
        # Classes

        self.radio_muon = QRadioButton("Muon")
        self.radio_muon.setStyleSheet("QRadioButton"
                                      "{"
                                      "font : 15px Arial;"
                                      "}")
        self.radio_eletron = QRadioButton("Electron")
        self.radio_eletron.setStyleSheet("QRadioButton"
                                         "{"
                                         "font : 15px Arial;"
                                         "}")
        self.radio_blob = QRadioButton("Blob")
        self.radio_blob.setStyleSheet("QRadioButton"
                                     "{"
                                     "font : 15px Arial;"
                                     "}")
        self.radio_diffusion_hit = QRadioButton("Diffusion hit (blob < 600 eV)")
        self.radio_diffusion_hit.setStyleSheet("QRadioButton"
                                               "{"
                                               "font : 15px Arial;"
                                               "}")
        self.radio_alpha = QRadioButton("Alpha")
        self.radio_alpha.setStyleSheet("QRadioButton"
                                       "{"
                                       "font : 15px Arial;"
                                       "}")
        self.radio_others = QRadioButton("Others")
        self.radio_others.setStyleSheet("QRadioButton"
                                        "{"
                                        "font : 15px Arial;"
                                        "}")
        
        self.classes_buttom_layout = QButtonGroup()
        self.classes_buttom_layout.addButton(self.radio_muon,
                                             id=ClassLabel.Muon.value)
        self.classes_buttom_layout.addButton(self.radio_eletron,
                                             id=ClassLabel.Electron.value)
        self.classes_buttom_layout.addButton(self.radio_blob,
                                             id=ClassLabel.Blob.value)
        self.classes_buttom_layout.addButton(self.radio_diffusion_hit,
                                             id=ClassLabel.Diffusion_Hit.value)
        self.classes_buttom_layout.addButton(self.radio_alpha,
                                             id=ClassLabel.Alpha.value)
        self.classes_buttom_layout.addButton(self.radio_others,
                                             id=ClassLabel.Others.value)
        classes_group = QGroupBox("Event classification")
        classes_layout = QVBoxLayout()
        classes_layout.addWidget(self.radio_muon)
        classes_layout.addWidget(self.radio_eletron)
        classes_layout.addWidget(self.radio_blob)
        classes_layout.addWidget(self.radio_diffusion_hit)
        classes_layout.addWidget(self.radio_alpha)
        classes_layout.addWidget(self.radio_others)
        classes_layout.addStretch(1)
        classes_group.setLayout(classes_layout)
        
        # Buttons
        buttons_group = QGroupBox()
        buttons_group.setStyleSheet("QGroupBox { border: 1px;}")

        buttons_layout = QHBoxLayout()
        self.button_skip = QPushButton("Skip")
        self.button_submit = QPushButton("Submit")
        self.button_submit.clicked.connect(self.submit_info)
        self.button_skip.clicked.connect(self.skip_image)
        buttons_layout.addWidget(self.button_skip)
        buttons_layout.addWidget(self.button_submit)
        buttons_group.setLayout(buttons_layout)

        self.layout.addWidget(self.file_info_group, 1, 3)
        self.layout.addWidget(self.event_info_group, 2, 3)
        self.layout.addWidget(classes_group, 3, 3)
        self.layout.addWidget(buttons_group, 4, 3)
        main_widget.setLayout(self.layout)
        self.show()

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
        curr_class = ClassLabel(buttom_id).name.replace("_", " ")
        image_data = ImageData(Path(self.filename).name, self.img_idx,
                               get_username(), get_datetime(),
                               curr_class)
        print(f"image_data = {image_data}\n")
        self.database.insert_event_info(image_data)
        self.root_file.remove_idx(self.img_idx)
        self.get_new_file_information()
        self.update_file_info()
        self.release_buttons()

    def skip_image(self):
        """ Skip buttom """
        self.block_buttons()
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
        self.img_idx = self.root_file.get_new_img_idx(skip)
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
