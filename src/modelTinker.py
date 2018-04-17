from keras.models import load_model
from PIL import Image
import os
import numpy as np

model = load_model("fourth_try.h5")

porn_basePath = "Data/Data_Keras/train/Porn/"
for image_file in os.listdir(porn_basePath):
    img = Image.open(porn_basePath + image_file).convert("RGB")
    img.show()
    img = img.resize((150, 150))
    npImg = np.asarray(img)
    npImg = npImg.reshape((1,150,150,3)) 
    predictionAcc = model.predict(npImg)[0][0]
