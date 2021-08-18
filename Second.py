import cv2 as cv

cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0)

while True:
    label , img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face_rect = cascade.detectMultiScale(gray,1.1,3)

    for (x,y,w,h) in face_rect:
        cv.rectangle(img, (x,y),(x+w, y+h),(0,255,0), 3)

    cv.imshow("Detected Face", img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
