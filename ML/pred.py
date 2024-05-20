import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tkinter.filedialog import askopenfilename

loaded_model = load_model("Trained_Model/model_mobilenet.h5")
width = 224
height = 224

def prediction(path):
    image=cv2.imread(path)
    resize_image = cv2.resize(image,(height,width))
    out_image=np.expand_dims(resize_image,axis=0)/255
    print(out_image.shape)

    my_pred = loaded_model.predict(out_image)
    print(my_pred)
    my_pred=np.argmax(my_pred,axis=1)
    my_pred = my_pred[0]
    print(my_pred)

    if my_pred ==0:
        print("Distraction")
    elif my_pred ==1:
        print("Normal")
    

if __name__=='__main__':
    path=askopenfilename()
    prediction(path)