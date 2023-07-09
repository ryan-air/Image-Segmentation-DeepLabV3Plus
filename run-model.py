## Code for running the model provided within this repository 
#Importing Dependencies 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
from train import create_dir

#Setting Global Parameters
H = 512
W = 512

if __name__ == "__main__":
    
    #Loading in images to run the model on (change directory as required)
    data = glob("folder_containing_all_images/image/*")

    #Creating directory to store results (change directory as required)
    create_dir("folder_containing_all_images/masks")

    #Random seeding
    tf.random.set_seed(42)
    np.random.seed(42)
    
    #Loading in model to work with (change directory as required)
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("folder_containing_model/segmentation-model.h5")

for path in tqdm(data, total=len(data)):
        name = path.split("/")[-1].split(".")[0]

        #Image Reading
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))
        x = x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        #Segmentation
        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)

        #Final (change directory as required)
        segmented_image = image * y
        line = np.ones((h, 10, 3)) * 128
        cat_images = np.concatenate([image, line, segmented_image], axis=1)
        cv2.imwrite(f"folder_containing_all_images/masks/{name}.png", cat_images)