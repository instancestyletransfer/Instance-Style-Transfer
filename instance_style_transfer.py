import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import trace
import tensorflow as tf
import keras.backend as K
import scripts.ist_utils as pu
import scripts.ist_model as pm

###################################################### adopted from external source
sys.path.append("./")

K.clear_session()
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from scripts import utils
import scripts.model as modellib
from scripts import visualize
from scripts import config
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

import importlib
importlib.reload(modellib)

class InferenceConfig(config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81
    NAME = "TESTING"

config = InferenceConfig()


# Create model object in inference mode.
with tf.device("/cpu:0"):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


#############################################################

## the images passed over from command line
content_image = sys.argv[1]
style_image = sys.argv[2]

image5 = skimage.io.imread(content_image)
results = model.detect([image5], verbose=0)
r = results[0]

## get the mask of the instance
first_mask = r["masks"][:, :, 0]

## Forward Stretching
o = pm.PixelAssignment(first_mask, image5)
a, b = o.transform()
   
## stylized the instance
stylized_obj = o.obj_sty_transfer(style_image = [style_image])

## Backward Stretching
model_output = pu.restore(stylized_obj, first_mask, image5)

c = os.path.basename(content_image)
c = os.path.splitext(c)[0]
s = os.path.basename(style_image)

## Save output
skimage.io.imsave(".//output//" + c + "_" + s, model_output)
