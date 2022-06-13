import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps


x,y=fetch_openml("mnist_784",version=1,return_X_y=True)
classes=["0","1","2","3","4","5","6","7","8","9"]
len=len(classes)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=3,train_size=7500,test_size=2500)
xtrain=xtrain/255
xtest=xtest/255

model=LogisticRegression(solver="saga",multi_class="multinomial").fit(xtrain,ytrain)
ypread=model.predict(xtest)
print(accuracy_score(ytest,ypread))

cam=cv2.VideoCapture(0)
while True:
    ret,img=cam.read()
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height,width=grey.shape
    upperleft=(width/2-50,height/2-50)
    bottomright=(width/2+50,height/2+50)
    cv2.rectangle(grey,upperleft,bottomright,(255,0,0),2)
    roi=grey[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
    #Converting cv2 image to pil format
    impil=Image.fromarray(roi)
    # convert to grayscale image - 'L' format means each pixel is 
    # represented by a single value from 0 to 255
    imbw=impil.convert("L")
    image_bw_resized = imbw.resize((28,28), Image.ANTIALIAS)

    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = model.predict(test_sample)
    print("Predicted class is: ", test_pred)

    # Display the resulting frame
    cv2.imshow('frame',grey)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cam.release()
cv2.destroyAllWindows
