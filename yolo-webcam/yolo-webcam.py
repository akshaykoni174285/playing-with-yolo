from ultralytics import YOLO
import cv2 as cv

model  = YOLO("../yolo-weights/yolov8n.pt")

cam  = cv.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)
# setting the resolutions
while cam.isOpened():
    ret, frame = cam.read()
    frame = cv.flip(frame,1)
    if ret == False:
        print("issue with the cam closing now")
        exit(0)
    results = model(frame,stream=True)
    # as we are using video we are streaming
    for r in results:
        # tapping into boxes
        boxes = r.boxes
        for box in boxes:
            # getting cordinates of each boxes
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # and drawing those boxes on cam frame using color and thickness
            cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)

    cv.imshow("cam",frame)
    cv.waitKey(1)
