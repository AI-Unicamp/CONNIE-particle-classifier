import os
import argparse
import sys

import cv2 as cv
import numpy as np

from shutil import rmtree
from glob import glob
from astropy.io import fits
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core import DATA_FOLDER


def get_calibrated_imgs(calibrated_img_path):
    file_ext = "fits"
    calibrated_img_path_list = glob(os.path.join(calibrated_img_path, "*" + file_ext))

    calibrated_img = []
    calibrated_img_filename = []

    for img in calibrated_img_path_list:
        calibrated_img.append(fits.open(img)[0])
        calibrated_img_filename.append(Path(img).stem)
    calibrated_img_list = []
    for img in calibrated_img:
        calibrated_img_list.append(img.data)
    calibrated_img_np = np.array(calibrated_img_list)
    print(f"Shape calibrated_img_np = {np.shape(calibrated_img_np)}")
    return calibrated_img_np, calibrated_img_filename


def apply_mask(calibrated_img_np):
    frame = 5
    new_mask = np.full(np.shape(calibrated_img_np[0]), False)
    new_mask[:, :frame] = True
    new_mask[:, -frame:] = True
    new_mask[:frame, :] = True
    new_mask[-frame:, :] = True
    return new_mask


def replace_negative(img):
    # 4 sigma (0.16) == 0.64
    return img.clip(0.64)


def replace_negative_imgs(calibrated_img_np, new_mask):
    masked_border_img = []
    for img in calibrated_img_np:
        masked_border_img.append(replace_negative(
            np.ma.masked_array(img, new_mask, fill_value=np.nan)))
    return masked_border_img


def save_cropped_events(cropped_imgs_path):
    if os.path.exists(cropped_imgs_path):
        rmtree(cropped_imgs_path)
    os.makedirs(cropped_imgs_path)
    return cropped_imgs_path

def apply_transformations(masked_border_img, cropped_imgs_path, calibrated_img_filename):
    factor = 3.745
    threshold_ev = 10
    threshold = threshold_ev/factor
    print(f"Threshold in e- value equivalent to {threshold_ev}eV = {threshold}")

    for idx, _ in enumerate(masked_border_img):
        img_clip = masked_border_img[idx]
        img_original = masked_border_img[idx].copy()
        blurred_img = cv.GaussianBlur(img_clip, (3, 3), 0)
        _, img_thresh = cv.threshold(blurred_img, threshold, 255, cv.THRESH_BINARY)
        mask = cv.morphologyEx(img_thresh.astype(np.uint8), cv.MORPH_OPEN,
                               cv.getStructuringElement(cv.MORPH_CROSS,(3,3)))
        mask = cv.dilate(mask,
                         cv.getStructuringElement(cv.MORPH_CROSS,(1,1)),
                         iterations=1)
        mask = cv.erode(mask,
                        cv.getStructuringElement(cv.MORPH_CROSS,(4, 4)),
                        iterations=1)
        contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)[0]
        contour_area_min = 2

        boxes = []
        for contour in contours:
            if (cv.contourArea(contour) >= contour_area_min):
                boxes.append(cv.boundingRect(contour))

        for box_idx, box in enumerate(boxes):
            top_left = (box[0], box[1])
            bottom_right = (box[0] + box[2], box[1] + box[3])
            cv.rectangle(img_original, top_left, bottom_right, (255, 0, 0), 5)
            cropped_img = img_clip[box[1]:box[1] + box[3], box[0]: box[0]+box[2]]
            padding_img = cv.copyMakeBorder(cropped_img, 30, 30, 30, 30,
                                            cv.BORDER_CONSTANT, None, value=0) 
            cv.imwrite(os.path.join(cropped_imgs_path, calibrated_img_filename[idx]
                                    + "_" + str(box_idx) + ".png"), padding_img)


def main():
    calibrated_img_path = os.path.join(DATA_FOLDER,
                                       "connie_calibrated_images")
    cropped_imgs_path = os.path.join(DATA_FOLDER,
                                     "connie_cropped_events",
                                     "unknown_label")
    parser = argparse.ArgumentParser(description="Extract CONNIE events")
    parser.add_argument("--input_folder", type=str,
                        default=calibrated_img_path,
                        help="Calibrated CONNIE images path")
    parser.add_argument("--output_folder", type=str,
                        default=cropped_imgs_path,
                        help="Output folder path for cropped events")

    calibrated_img_np, calibrated_img_filename = get_calibrated_imgs()
    new_mask = apply_mask(calibrated_img_np)
    masked_border_img = replace_negative_imgs(calibrated_img_np, new_mask)
    save_cropped_events(cropped_imgs_path)
    apply_transformations(masked_border_img, cropped_imgs_path, calibrated_img_filename)

if __name__ == "__main__":
    main()