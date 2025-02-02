import argparse
import os
import pickle
from os import path
import keras

import cv2
import numpy as np
import pandas as pd

import pydicom as dicom

from keras.models import load_model
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#MODEL_PATH = "/home/end2end/weights/"
MODEL_PATH = ""


# From Li Shen's end2end dm_image.py
def read_resize_img(fname, target_size=None, target_height=None, target_scale=None, gs_255=False, rescale_factor=None):
    """Read an image (.png, .jpg, .dcm) and resize it to target size."""
    if target_size is None and target_height is None:
        raise Exception('One of [taget_size, target_height] must not be None')
    if path.splitext(fname)[1] == '.dcm':
        img = dicom.read_file(fname).pixel_array
    else:
        if gs_255:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if target_height is not None:
        target_width = int(float(target_height) / img.shape[0] * img.shape[1])
    else:
        target_height, target_width = target_size
    if (target_height, target_width) != img.shape:
        img = cv2.resize(img, dsize=(target_width, target_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype('float32')
    if target_scale is not None:
        img_max = img.max() if img.max() != 0 else target_scale
        img *= target_scale / img_max
    if rescale_factor is not None:
        img *= rescale_factor
    return img


# Load weights
# Iterate over files and pass to read_resize_img
# Generate predictions
# Save predictions 
def evaluate(model, model_name, pkl_path, image_path, mean_pixel_intensity, rescale_factor):
    
    malignant_pred = []
    #malignant_label = []
    #im_path="/content/gdrive/MyDrive/polygence/datasets/inbreast/test/pos/20586934_6c613a14b80a8591_MG_L_CC_ANON.jpeg"

    #im_path="/content/gdrive/MyDrive/polygence/datasets/inbreast/test/neg/53586388_dda3c6969a34ff8e_MG_R_ML_ANON.jpeg"

    #im_path="/content/gdrive/MyDrive/polygence/datasets/inbreast/AllDICOMs/53586388_dda3c6969a34ff8e_MG_R_ML_ANON.dcm"
    directory = '/content/gdrive/MyDrive/polygence/datasets/inbreast/AllPNGs'
    for filename in os.listdir(image_path):
        im_path = os.path.join(image_path, filename)
        print("***Image  is **** ",im_path)

        im = read_resize_img(im_path, target_size=(1152, 896))
        im *= rescale_factor
        im -= mean_pixel_intensity
        if model_name.find("YaroslavNet") == -1:
            three_channel_image = np.zeros(im.shape + (3,))
            three_channel_image[:, :, 0] = im
            three_channel_image[:, :, 1] = im
            three_channel_image[:, :, 2] = im
            batch = np.array([three_channel_image])
        else:
            single_channel_image = np.zeros(im.shape + (1,))
            single_channel_image[:, :, 0] = im
            batch = np.array([single_channel_image])

        # Predict
        pred = model.predict(batch)

        malignant_pred.append(pred[0][1])
        print("Image  is **** ",im_path)

        print("Pred is **** ",pred[0][1])
        
        df = pd.DataFrame()
        df['malignant_pred'] = malignant_pred
   
    return df


def main(pkl_file, image_path, prediction_file, model_name, mean_pixel_intensity, rescale_factor):
    #model = load_model(MODEL_PATH + model_name, compile=False)
    model = load_model(model_name,compile=False)
    df = evaluate(model, model_name,pkl_file,image_path, mean_pixel_intensity, rescale_factor)
    df.to_csv(prediction_file)
    keras.backend.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate end2end model on a data set")
    parser.add_argument('--input-data-folder', required=True)
    parser.add_argument('--exam-list-path', required=True)
    parser.add_argument('--prediction-file', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--rescale-factor', required=True, type=float)
    parser.add_argument('--mean-pixel-intensity', required=True, type=float)

    args = parser.parse_args()

    print("\nGenerating predictions.")
    main(args.exam_list_path, args.input_data_folder, args.prediction_file, args.model, args.mean_pixel_intensity, args.rescale_factor)

