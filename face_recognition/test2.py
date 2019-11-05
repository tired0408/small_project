import cv2
import face_recognition
import dlib
img = cv2.imread("./data/test.jpg")
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)  # 对照片进行1/4缩放
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将照片转为灰度图
face_locations = face_recognition.face_locations(gray_img)
detector = dlib.get_frontal_face_detector()
dets = detector(gray_img, 1)
print(dets[0].top())
print(dets[0].left())
print(dets)
print(face_locations)