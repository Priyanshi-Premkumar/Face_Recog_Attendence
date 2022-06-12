import cv2
import numpy as np
import face_recognition
#LOADING IMAGES
imgElon = face_recognition.load_image_file('images/elon-musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('images/Elon-Musk-Reuters-Event.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
imgBill = face_recognition.load_image_file('images/104891709-Bill_Gates_the_co-Founder.jpg')
imgBill = cv2.cvtColor(imgBill,cv2.COLOR_BGR2RGB)



#FINDING FACE AND THEIR ENCODINGS

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
print(faceLoc)


faceLocBill = face_recognition.face_locations(imgBill)[0]
encodeBill = face_recognition.face_encodings(imgBill)[0]
cv2.rectangle(imgBill,(faceLocBill[3],faceLocBill[0]),(faceLocBill[1],faceLocBill[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
print(faceLocTest)

#comparing and finding distance

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results, faceDis)
fontScale = 2
cv2.putText(imgTest,f'{results},{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,fontScale,(0,0,255),2)



imgElonS = cv2.resize(imgElon, (960, 540))
cv2.imshow('Elon Musk',imgElonS)
imgTestS = cv2.resize(imgTest,(960,540))
cv2.imshow('Elon Test',imgTestS)
cv2.imshow('Bill',imgBill)
cv2.waitKey(0)


