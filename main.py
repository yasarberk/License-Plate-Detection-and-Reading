from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}
obj_tracker = Sort()

# models
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO("./models/licence_plate.pt")

vehicles = [2, 3, 5, 7]  # coco model's id numbers. Car: 2 Motorbike: 3 Bus: 5 Truck: 7

# load video
cap = cv2.VideoCapture("./sample2.mp4")

# read frames
frame_number = -1
read = True
while read:
    frame_number += 1
    read, frame = cap.read()
    if read: 
        results[frame_number] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Vehicle tracking
        track_ids = obj_tracker.update(np.asarray(detections_))

        # Licence plate detection
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Cropping licens plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_threshold =  cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                #cv2.imshow('original_crop', license_plate_crop)
                #cv2.imshow('threshold', license_plate_crop_threshold)

                #cv2.waitKey(0)

                # Read License Plate
                license_plate_text, license_plate_text_conf_score = read_license_plate(license_plate_crop_threshold)

                if license_plate_text is not None:
                    results[frame_number][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2], 
                                                                    'text': license_plate_text, 
                                                                    'bbox_score': score, 
                                                                    'text_score': license_plate_text_conf_score}}

# Write results
write_csv(results, './sample2.csv')
