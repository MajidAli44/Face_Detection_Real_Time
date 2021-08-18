import cv2 as cv

cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv.imread("/home/majid/Pictures/OpenCV/dost.jpg")

resized = cv.resize(img,(img.shape[1]//2, img.shape[0]//2))
# cv.imshow("Resized", resized)

gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

face_rect = cascade.detectMultiScale(gray, 1.1, 5)
for (x,y,w,h) in face_rect:
    cv.rectangle(resized, (x,y), (x+w, y+h), (0,255,0),3)

cv.imshow("face Detected",resized)
cv.imshow("Gray",gray)

cv.waitKey(0)
