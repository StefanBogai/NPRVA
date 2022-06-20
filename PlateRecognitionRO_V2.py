from pickle import NONE
import cv2
import os
import numpy as np
import imutils
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
TESSDATA_PREFIX = 'C:/Program Files/Tesseract-OCR'
car_classifier = cv2.CascadeClassifier(os.path.join('assets','haarcascade_car.xml'))

vid = cv2.VideoCapture(os.path.join('assets','VID3.mp4'))
ctframe = 0

def NPR(frame):
    bfilter = cv2.bilateralFilter(frame,11,17,17)
    blur = cv2.GaussianBlur(bfilter, (5,5),0)
    contrasted = cv2.addWeighted(blur,2.5,blur,0,-127)
    edged = cv2.Canny(contrasted, 170, 200)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse = True) [:10]

    

    location = NONE
    for contour in contours:
        peri = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour, 0.018*peri, closed = True)
        if len(approx) == 6:
            location = approx
            break
        
    if location!= NONE:
        mask = np.zeros(frame.shape, np.uint8)
        temp = cv2.drawContours(mask, [location], 0, 255, -1)
        temp = cv2.bitwise_and(frame,frame,mask=mask)
        (x,y) = np.where(mask==255)
        (x1,y1) = (np.min(x), np.min(y))
        (x2,y2) = (np.max(x), np.max(y))
        cropped_image = frame[x1:x2+1, y1:y2+1]
        
        text = pytesseract.image_to_string(cropped_image, config='--psm 7 ')
        print(text)
     
        x = re.search("[A-Z]{1,2}[0-9][0-9]{1,2}[A-Z]{3}|[A-Z]{1,2} [0-9][0-9]{1,2}[A-Z]{3}|[A-Z]{1,2} [0-9][0-9]{1,2} [A-Z]{3}|[A-Z]{1,2}[0-9][0-9]{1,2} [A-Z]{3}|[A-Z]{1,2}[O][0-9]{1,2}[A-Z]{3}|[A-Z]{1,2} [O][0-9]{1,2}[A-Z]{3}|[A-Z]{1,2} [O][0-9]{1,2} [A-Z]{3}|[A-Z]{1,2}[O][0-9]{1,2} [A-Z]{3}",text)
        if x is not None:
            text = x.group(0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            res = cv2.putText(cropped_image, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace = font, fontScale=1, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
            res = cv2.rectangle(cropped_image, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0), 3)
            cv2.imwrite(os.path.join('frames','frame' + str(ctframe) + '.jpg'), res)

def CDE(frame,edgedframe):
    cars = car_classifier.detectMultiScale(frame, 1.4, 2)
    
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cropped_car = edgedframe[x:w, y: h]
        cv2.imshow('Plate', frame)
        NPR(cropped_car)

for frame_idx in range(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
    if frame_idx%2 == 0:
    
        ret, frame = vid.read()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(frame,9,13,13)
        blur = cv2.GaussianBlur(bfilter, (5,5),0)
        contrasted = cv2.addWeighted(blur,2.5,blur,0,-127)
        edged = cv2.Canny(contrasted, 170, 200)

        CDE(gray,gray)
    
    else:
        success = vid.grab()
  
    if cv2.waitKey(10)== 13: 
        break
        
vid.release()
cv2.destroyAllWindows()