# Importing the required modules to do the invisible task

import cv2   # The module which will deals with the images
import numpy as np    # The module which will deal with image's pixel array
import time   # The module which will deal with the time
from flask import Flask,render_template,Response  # The  modules which deal with our backend

app=Flask(__name__)


def generate_frames():
    cap = cv2.VideoCapture(0)  # I am going to use my integrated webcam that's why I have use 0 if you do not have you integrated webcam then you can use 1 as the parameter

    # The video consists of the infinite frame so we have to make use of the while loop until we dont required the video input.

    time.sleep(4)
    # Now I am going to provide the camera 3 sec  to setup accrding the environment

    background = 0

    # Now we are going to give 30 iteration to the camera to capture the backgorund

    for i in range(30):

        ret, background = cap.read()

    # Now we are approching to capture ouself in the form of array

    while cap.isOpened(): # This loop will end only when we will close the webcam 

        # .read() returns two value one is in boolean and other the frame of the capture
        ret, img = cap.read()
        
        #initially your cv image
        cv2.imshow('Real',img)
        # If ret is false i.e the read function does not work then in that case this loop will exit
        if not ret: 
            break
        # Reason why we have use hsv instead of brg because hsv is know as how the human see the color
        # Now we are going to convert the img to hsv which is in BGR form
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        # Now we are defining the two array
        # We can make it by using BLue,Green and Red colour because these colours adapts the extra colours in the backkground
        # I have used blue cloth we can choose your
        
        lower = np.array([90,103,20])
        upper = np.array([119,255,255])

        # This will create a mask in this range
        mask1 =  cv2.inRange(hsv,lower,upper)

        lower = np.array([180,98,20])
        upper = np.array([170,255,255])
        
        # Same as the mask1 
        mask2 =  cv2.inRange(hsv,lower,upper)

        mask1 = mask1 + mask2
        
        # Theory behind the morphing 
        """
        Morphological transformations are some simple operations based on the 
        image shape. It is normally performed on binary images. 
        It needs two inputs, one is our original image, second one is called 
        structuring element or kernel which decides the nature of operation. 
        Two basic morphological operators are Erosion and Dilation. 
        Then its variant forms like Opening, Closing, Gradient etc also comes 
        into play. We will see them one-by-one with help of following image:
        """
        
        """
        MORPH_OPEN is a errosion morph which is just like soil errosion, it
        removes the outer boundaries
        """
        mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8), iterations = 3)
        
        mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8), iterations = 0)
        
        mask2 = cv2.bitwise_not(mask1)
        
        res1 = cv2.bitwise_and(background,background,mask = mask1)
        res2 = cv2.bitwise_and(img,img,mask = mask2)

        final = cv2.addWeighted(res1,1,res2,1,0)
        ret,buffer=cv2.imencode('.jpg',final)
        final=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + final + b'\r\n')
            


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)