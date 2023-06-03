import os
import numpy as np
import torch

import cv2

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from segment_anything import SamPredictor, sam_model_registry

HOME = os.getcwd()
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_h"
device = "cuda"

model = {1: 'sam_vit_b_01ec64.pth',
         2: 'sam_vit_h_4b8939.pth',
         3: 'sam_vit_l_0b3195.pth',
         }

directory = f'{model[2]}'
image = 'CinebenchCPU.png'
image = cv2.imread(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()

CHECKPOINT_PATH = os.path.join(HOME, directory)
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

## ---------------------------------------------------------- ##
#                    Loading the model
## ---------------------------------------------------------- ##
sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH)
sam.to(device=device)
predictor = SamPredictor(sam)
# Predicting image
predictor.set_image(image)

image_embedding = predictor.get_image_embedding().cpu().numpy()
print(image_embedding.shape)
