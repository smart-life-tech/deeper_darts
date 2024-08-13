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
    max_darts_exceeded = False
    
    print(f"Shape of xy: {xy.shape}")
    print(f"Value of boxes: {bboxes}")

    for bbox in bboxes: 
        if int(bbox.cls) == 0 and not max_darts_exceeded:  # bbox is around a dart, add dart center to xy array
            dart_xywhn = bbox.xywhn[0]  # center coordinates
            print(f"Dart found with xywhn: {dart_xywhn}")

            dart_x_centre = float(dart_xywhn[0])
            dart_y_centre = float(dart_xywhn[1])
            dart_xy_centre = np.array([dart_x_centre, dart_y_centre])
            
            print(f"Dart centre xyn: {dart_xy_centre}")
            print(f"Num_darts: {num_darts}")
            
            collumn = 4 + num_darts
            
            if collumn < xy.shape[0]:  # Ensure we're within bounds
                xy[collumn, :2] = dart_xy_centre
                num_darts += 1
            else:
                print(f"Couldn't add dart {num_darts+1}, index error")

            if num_darts >= max_darts:  # Stop if max darts is reached
                print("Max number of darts exceeded, ignoring any other detected darts")
                max_darts_exceeded = True
        else:  # Handle calibration points
            cal_xywhn = bbox.xywhn[0]  # center coordinates
            cal_x_centre = float(cal_xywhn[0])
            cal_y_centre = float(cal_xywhn[1])
            cal_xy_centre = np.array([cal_x_centre, cal_y_centre])

            collumn = int(bbox.cls) - 1
            if collumn < xy.shape[0]:  # Ensure within bounds
                xy[collumn, :2] = cal_xy_centre

    # Mark valid points
    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1

    # Check if all 4 calibration points are detected
    if np.sum(xy[:4, -1]) == 4:
        return xy
    else:
        # Estimate missing calibration points
        xy = est_cal_pts(xy)
    
    return xy


def est_cal_pts(xy):
    '''
    From DeepDarts
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
        # TODO: if len(missing_idx) > 1
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
    ret, frame = cap.read()
    if not ret:
        break

    # Run model prediction on the frame
    results = model.predict(frame)
    image_result = results[0]

    # Process the results
    boxes = image_result.boxes
    xy = bboxes_to_xy(boxes)
    xy = xy[xy[:, -1] == 1]  # Remove any empty rows
    predicted_score = get_dart_scores(xy, cfg, numeric=False)

    # Draw the results on the frame
    img = draw(frame, xy[:, :2], cfg, circles=False, score=True)

    # Display the result
    cv2.imshow('Dart Board', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
