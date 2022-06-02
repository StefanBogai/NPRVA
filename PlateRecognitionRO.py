from pickle import NONE
import cv2
import os
import numpy as np
import imutils
import pytesseract
from matplotlib import pyplot as plt
import easyocr
import re

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
TESSDATA_PREFIX = 'C:/Program Files/Tesseract-OCR'

vid = cv2.VideoCapture(os.path.join('assets','VID7.mp4'))
#vid.set(cv2.cv.CV_CAP_PROP_FPS_5)

for frame_idx in range(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
    if frame_idx%2 == 0:
        success, frame = vid.read()
        bdframe = cv2.resize(frame,(0,0), fx=0.4,fy=0.4, interpolation = cv2.INTER_LINEAR)
        cv2.imshow('Video Player', bdframe)
        gray = cv2.cvtColor(bdframe, cv2.COLOR_BGR2GRAY)

        bfilter = cv2.bilateralFilter(gray,11,17,17)
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
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(bdframe,bdframe,mask=mask)

        (x,y) = np.where(mask==255)
        (x1,y1) = (np.min(x), np.min(y))
        (x2,y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]

        
        text = pytesseract.image_to_string(cropped_image, config='--psm 7 ')
        print(text)
     
        x = re.search("[A-Z]{1,2}[0-9][0-9]{1,2}[A-Z]{3}|[A-Z]{1,2} [0-9][0-9]{1,2}[A-Z]{3}|[A-Z]{1,2} [0-9][0-9]{1,2} [A-Z]{3}|[A-Z]{1,2}[0-9][0-9]{1,2} [A-Z]{3}|[A-Z]{1,2}[O][0-9]{1,2}[A-Z]{3}|[A-Z]{1,2} [O][0-9]{1,2}[A-Z]{3}|[A-Z]{1,2} [O][0-9]{1,2} [A-Z]{3}|[A-Z]{1,2}[O][0-9]{1,2} [A-Z]{3}",text)
        if x is not None:
            text = x.group(0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            res = cv2.putText(bdframe, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace = font, fontScale=1, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
            res = cv2.rectangle(bdframe, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0), 3)
            cv2.imwrite(os.path.join('frames','frame' + str(frame_idx) + '.jpg'), res)
        
    else:
        success = vid.grab()

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
