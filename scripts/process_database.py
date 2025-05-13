import os
import sqlite3
import sys
import uproot
import pathlib
import numpy as np
import pandas as pd
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core import DATA_FOLDER, CATALOG_FOLDERPATH, DATABASE_FOLDER, ClassLabel


# Constants
OUTPUT_DIR = os.path.join(DATA_FOLDER, "processed_data")
PNG_DIR = os.path.join(DATA_FOLDER, "png_processed_data")
PNG_DISCREPANCY_DIR = os.path.join(DATA_FOLDER, "png_discrepancies")
DISCREPANCY_DIR = os.path.join(DATA_FOLDER, "discrepancies")


def load_events():
    database_name = "connie_label.db"
    table_name = "events"
    conn = sqlite3.connect(os.path.join(DATABASE_FOLDER, database_name))
    df = pd.read_sql_query((f"SELECT * FROM {table_name}"), conn)
    conn.close()
    return df


def identify_discrepancies(df):
    grouped = df.groupby(['img_idx', 'filename'])
    discrepancies = grouped.filter(lambda x: x['label'].nunique() > 1)
    return discrepancies


def is_valid_label(label):
    return label in ClassLabel.__members__


def convert_numpy_scalar(val):
    if isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.floating, float)):
        return float(val)
    else:
        raise ValueError(f"Unsupported scalar type: {type(val)}")


def extract_and_save_root_and_png(root_path, idx, save_root_path, mode="log", eps=1e-3, png_base_dir=PNG_DIR):
    """
    Extract a specific index from a ROOT file and save it as:
    - A new ROOT file (same structure, single entry)
    - A PNG image of the ePix data (normalized for CNN training)
    """
    branch_name = "hitSumm"
    try:
        file = uproot.open(root_path)
        branch = file[branch_name]
        branch_keys = branch.keys()

        image_matrix = None
        extracted_data = {}
        branch_types = {}

        for key in branch_keys:
            array = branch[key].array(library="np")
            val = array[idx]
            if key == "xPix":
                xPix = array[idx]
            elif key == "yPix":
                yPix = array[idx]
            elif key == "ePix":
                ePix = array[idx]
            if isinstance(val, np.ndarray):
                if val.ndim == 0:
                    extracted_data[key] = np.array([convert_numpy_scalar(val.item())])
                    branch_types[key] = extracted_data[key].dtype
                else:
                    extracted_data[key] = np.expand_dims(val, axis=0)
                    branch_types[key] = extracted_data[key].dtype
            else:
                extracted_data[key] = np.array([convert_numpy_scalar(val)])
                branch_types[key] = extracted_data[key].dtype

        os.makedirs(os.path.dirname(save_root_path), exist_ok=True)
        with uproot.recreate(save_root_path) as new_file:
            new_file[branch_name] = extracted_data

        width = np.max(xPix) - np.min(xPix) + 1
        height = np.max(yPix) - np.min(yPix) + 1
        image_matrix = np.ones((height, width)) * np.nan
        for x, y, value in zip(xPix, yPix, ePix):
            image_matrix[y - np.min(yPix), x - np.min(xPix)] = value

        file_name = pathlib.Path(save_root_path).stem
        label_folder = pathlib.Path(save_root_path).parents[0].name
        png_file_path = os.path.join(png_base_dir, label_folder, file_name + ".png")
        save_image_preserving_data(image_matrix, png_file_path, mode=mode, eps=eps)
        return True
    except Exception as e:
        print(f"[!] Error processing {root_path} idx {idx}: {e}")
        return False


def save_image_preserving_data(image, out_path, mode="log", eps=1e-3):
    """
    Save image as 8-bit PNG while preserving structure via scaling.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image = np.nan_to_num(image)

    if mode == "log":
        image = np.log(np.clip(image, eps, None))
    elif mode == "clip":
        p99 = np.percentile(image, 99)
        image = np.clip(image, 0, p99)
    elif mode != "linear":
        raise ValueError("Mode must be 'log', 'linear', or 'clip'")

    image_min, image_max = np.min(image), np.max(image)
    if image_max - image_min == 0:
        scaled = np.zeros_like(image)
    else:
        scaled = 255 * (image - image_min) / (image_max - image_min)

    img = Image.fromarray(scaled.astype(np.uint8), mode="L")
    img.save(out_path)


def organize_dataset(df, discrepancy_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DISCREPANCY_DIR, exist_ok=True)
    os.makedirs(PNG_DIR, exist_ok=True)
    os.makedirs(PNG_DISCREPANCY_DIR, exist_ok=True)

    for class_label in ClassLabel:
        os.makedirs(os.path.join(OUTPUT_DIR, class_label.name), exist_ok=True)
        os.makedirs(os.path.join(PNG_DIR, class_label.name), exist_ok=True)

    for _, row in df.iterrows():
        label = row['label'].strip().replace(" ", "_")
        if not is_valid_label(label):
            print(f"[!] Invalid label '{label}' in row {row}")
            continue

        filename = row['filename']
        img_idx = row['img_idx']
        root_path = os.path.join(CATALOG_FOLDERPATH, filename)

        if not os.path.exists(root_path):
            print(f"[!] File not found: {root_path}")
            continue

        out_name = f"{os.path.splitext(filename)[0]}_idx_{img_idx}.root"

        is_discrepant = ((discrepancy_df['img_idx'] == img_idx) &
                         (discrepancy_df['filename'] == filename)).any()

        if is_discrepant:
            save_path = os.path.join(DISCREPANCY_DIR, out_name)
            png_dir = PNG_DISCREPANCY_DIR
        else:
            save_path = os.path.join(OUTPUT_DIR, label, out_name)
            png_dir = PNG_DIR

        success = extract_and_save_root_and_png(
            root_path, img_idx, save_path, png_base_dir=png_dir
        )

        if not success:
            print(f"[!] Failed to save ROOT file for {filename} idx {img_idx}")


def main():
    df = load_events()
    discrepancy_df = identify_discrepancies(df)
    organize_dataset(df, discrepancy_df)
    print("Dataset organized successfully.")


if __name__ == "__main__":
    main()
