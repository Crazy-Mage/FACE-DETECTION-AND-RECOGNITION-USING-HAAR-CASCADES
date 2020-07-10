#for an video using Haar features
import cv2
import numpy as np
import pickle

face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade =cv2.CascadeClassifier('haarcascade_eye.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
labels={"person_name":1}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = { v:k for k,v in og_labels.items()}


cap=cv2.VideoCapture(0)
#img=cv2.imread('2.mp4')
while (True):
	ret,img=cap.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
	#print (len(faces))
	for(x,y,w,h) in faces:
	    cv2.rectangle(img, (x,y), (x+w , y+h), (255,0,0) , 2)#BGR
	    #region of interest
	    #shows the region in which the  face is there.
	    roi_gray=gray[y:y+h , x:x+w]#(ycoord_start: ycoord_end, xcoord_start:xcoord_end)
	    roi_color=img[y:y+h, x:x+w]

	    id_,conf = recognizer.predict(roi_gray)
	    if conf>=65:#and conf<=85:
	        print(labels[id_])
	        font=cv2.FONT_HERSHEY_SIMPLEX
	        name=labels[id_]
	        color=(255,255,255)
	        stroke=2
	        cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA )
	    img_item="myimg.jpg"
	    cv2.imwrite(img_item,roi_color)

	    #roi_color=img[y:y+h, x:x+w]
	    #for the eyes
	    #eyes=eye_cascade.detectMultiScale(roi_gray)
	    #for (ex,ey,ew,eh) in eyes:
	    #    cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

	cv2.imshow('img',img)
	#cv2.waitKey(0)
	k=cv2.waitKey(30) & 0xff== ord('q')
	if k == 27:
	    break

#cap.release()
cv2.destroyAllWindows()
