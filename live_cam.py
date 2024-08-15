import cv2
import numpy as np
from ultralytics import YOLO
from yacs.config import CfgNode as CN
from datasets.annotate import draw, get_dart_scores

est_cal_pts_cnt = 0

def bboxes_to_xy(bboxes, max_darts=3):
    '''
    Converts bounding box output from YOLOv8 of all classes to an xy centre point.
    Handles darts and calibration points.
    '''
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    num_darts = 0
    dartboard_detected = False

    print(f"Shape of xy: {xy.shape}")
    print(f"Detected bounding boxes: {bboxes.data}")

    for bbox in bboxes.data: 
        cls = int(bbox[5])
        print(f"Processing class {cls}")

        if cls == 0 and not dartboard_detected:  # Dartboard detection
            dartboard_detected = True  # Dartboard detected
            print("Dartboard detected.")

        elif cls == 0 and num_darts < max_darts:  # bbox is around a dart
            dart_xywhn = bbox[:4]  # center coordinates
            dart_x_centre = float(dart_xywhn[0])
            dart_y_centre = float(dart_xywhn[1])
            dart_xy_centre = np.array([dart_x_centre, dart_y_centre])

            collumn = 4 + num_darts
            xy[collumn, :2] = dart_xy_centre
            num_darts += 1
            print(f"Dart detected at {dart_xy_centre}.")

        elif 1 <= cls <= 4:  # Handle calibration points
            cal_xywhn = bbox[:4]  # center coordinates
            cal_x_centre = float(cal_xywhn[0])
            cal_y_centre = float(cal_xywhn[1])
            cal_xy_centre = np.array([cal_x_centre, cal_y_centre])

            collumn = cls - 1
            xy[collumn, :2] = cal_xy_centre

        else:
            print(f"Unexpected class {cls} detected, skipping...")

    # Mark valid points
    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1

    # Check if all 4 calibration points are detected
    if np.sum(xy[:4, -1]) == 4:
        return xy, dartboard_detected
    else:
        # Estimate missing calibration points
        xy = est_cal_pts(xy)
    
    return xy, dartboard_detected

def est_cal_pts(xy):
    '''
    Estimates any missed calibration points
    '''
    global est_cal_pts_cnt
    est_cal_pts_cnt += 1
    missing_idx = np.where(xy[:4, -1] == 0)[0]
    if len(missing_idx) == 1:
        if missing_idx[0] <= 1:
            center = np.mean(xy[2:4, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 0:
                xy[0, 0] = -xy[1, 0]
                xy[0, 1] = -xy[1, 1]
                xy[0, 2] = 1
            else:
                xy[1, 0] = -xy[0, 0]
                xy[1, 1] = -xy[0, 1]
                xy[1, 2] = 1
            xy[:, :2] += center
        else:
            center = np.mean(xy[:2, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 2:
                xy[2, 0] = -xy[3, 0]
                xy[2, 1] = -xy[3, 1]
                xy[2, 2] = 1
            else:
                xy[3, 0] = -xy[2, 0]
                xy[3, 1] = -xy[2, 1]
                xy[3, 2] = 1
            xy[:, :2] += center
    else:
        print('Missed more than 1 calibration point')
    return xy

# Load configuration
cfg = CN(new_allowed=True)
cfg.merge_from_file('configs/deepdarts_d1.yaml')

# Load the YOLOv8 model
model_path = "C:\\Users\\USER\\Documents\\raspberrypi\\dart\\darts2\\deeper_darts\\models\\yolov8n.pt"
model = YOLO(model_path)

# Set up video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify the camera index

while True:
    # ret, frame = cap.read()
    # if not ret:
    #     break
    print("reading images...")
    image_path = "C:\\Users\\USER\\Documents\\raspberrypi\\dart\\darts2\\deeper_darts\\images\\1.jpg"  # Change to your image path
    image_path="C:\\Users\\USER\\Documents\\raspberrypi\\dart\\darts2\\deeper_darts\\datasets\\800\\d1_02_04_2020\\IMG_1081.JPG"
    frame = cv2.imread(image_path)
    if frame is None:
        print("frame is none")
        break
    # Run model prediction on the frame
    results = model.predict(frame)
    image_result = results[0]

    # Process the results
    boxes = image_result.boxes
    xy, dartboard_detected = bboxes_to_xy(boxes)
    # Perform detection
    results = model(frame)

    # Display results
    annotated_frame = results.render()[0]
    cv2.imshow('Live Feed', annotated_frame)
    
    # if dartboard_detected:
    #     # Remove any empty rows
    #     xy = xy[xy[:, -1] == 1]

    #     # Calculate and display score
    #     predicted_score = get_dart_scores(xy, cfg, numeric=False)
    #     print(f"Predicted Score: {predicted_score}")

    #     # Draw the results on the frame
    #     img = draw(frame, xy[:, :2], cfg, circles=False, score=True)
    # else:
    #     print("No dartboard detected.")

    # # Display the result
    # cv2.imshow('Dart Board', img if dartboard_detected else frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
