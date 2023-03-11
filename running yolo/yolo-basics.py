from ultralytics import YOLO
import cv2 as cv

model = YOLO('../yolo-weights/yolov8l.pt')
result = model("Images/1.png", show =True)
cv.waitKey(0)