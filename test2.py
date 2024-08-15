import cv2
import os
import numpy as np
from ultralytics import YOLO
import random
import sys
import logging
import csv
import time
from PIL import Image
import pytesseract  # For OCR
import os.path as osp


# Global variable for estimated calibration points count
est_cal_pts_cnt = 0

def bboxes_to_xy(bboxes, max_darts=3):
    '''
    Converts bounding box output from YOLOv8 of all classes to an xy center point.
    Handles darts and calibration points.
    '''
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    num_darts = 0
    max_darts_exceeded = False
    
    print(f"Shape of xy: {xy.shape}")
    print(f"Value of boxes: {bboxes}")

    for bbox in bboxes: 
        if int(bbox.cls) == 0 and not max_darts_exceeded:  # bbox is around a dart
            dart_xywhn = bbox.xywhn[0]  # center coordinates
            dart_x_centre = float(dart_xywhn[0])
            dart_y_centre = float(dart_xywhn[1])
            dart_xy_centre = np.array([dart_x_centre, dart_y_centre])
            
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
        # TODO: Handle cases with more than one missing calibration point
        print('Missed more than 1 calibration point')
    return xy

def list_images_in_folder(folder_path):
    # List to store image file paths
    image_list = []

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a regular file and has an image extension
        if os.path.isfile(os.path.join(folder_path, filename)) and \
           filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # If the file is an image, add its path to the list
            image_list.append(os.path.join(folder_path, filename))

    return image_list

def get_label_xy(image_name, folder_path, max_darts=3):
    '''
    Gets the xy points from the label. Used for calculating 'actual' dart score.
    '''
    label_name = image_name.replace("JPG", "txt")
    print(f"Label name: {label_name}")
    label_path = f"{folder_path}/{label_name}.txt".replace("images", "labels")
    print(f"Label path: {label_path}")
    label_xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    num_darts = 0
    with open(label_path, 'r') as f:
        labels = f.readlines()
        print(f"Length of labels: {len(labels)}")
        for label in labels:
            split_label = label.split(" ")
            class_num = int(float(split_label[0]))
            x_centre = float(split_label[1])
            y_centre = float(split_label[2])
            xy_centre = np.array([x_centre, y_centre])
            
            if class_num == 0:
                label_xy[4+num_darts, :2] = xy_centre
                num_darts += 1
                print(f"Dart {num_darts}: {xy_centre}")
            else:
                label_xy[class_num - 1, :2] = xy_centre

    print(f"{num_darts} darts found in labels")
    label_xy[(label_xy[:, 0] > 0) & (label_xy[:, 1] > 0), -1] = 1
    return label_xy

def extract_score_from_image(image):
    '''
    Extracts score from the dartboard image using OCR
    '''
    # Convert image to grayscale for better OCR performance
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply OCR
    text = pytesseract.image_to_string(gray_image, config='--psm 6')
    
    # Extract score from text (assuming score is a number)
    try:
        score = int(text.strip())
    except ValueError:
        score = 0

    return score

def predict(model_directory):
    '''
    Used to predict dart location in 'image_folder_path' using the model in 'model_directory' 
    '''
    # Path to test images
    image_folder_path = 'C:\\Users\\USER\\Documents\\raspberrypi\\dart\\darts2\\deeper_darts\\datasets\\800\\d1_02_04_2020'

    # Make a list of all image paths
    images = list_images_in_folder(image_folder_path)

    # Directory to search for weights
    results_directory = 'DeeperDarts'
    best_weights_path = f"C:\\Users\\USER\\Documents\\raspberrypi\\dart\\darts2\\deeper_darts\\models\\yolov8n.pt"
    print(f"Loading {best_weights_path}")
    # Load model
    model = YOLO(best_weights_path)
    errors = []
    no_error_total = 0

    recent_results_img_dir = f"datasets/test/{model_directory}"
    os.makedirs(recent_results_img_dir, exist_ok=True)

    labeled_img_dir = f"{recent_results_img_dir}/scored_images"
    os.makedirs(labeled_img_dir, exist_ok=True)

    predicted_img_dir = f"{recent_results_img_dir}/predicted_images"
    os.makedirs(predicted_img_dir, exist_ok=True)

    log = logging.getLogger()
    for hdlr in log.handlers[:]:
        log.removeHandler(hdlr)

    # Set up logging
    log_dir = f"test_logs/{model_directory}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/processing_output.log"
    fileh = logging.FileHandler(log_file, 'a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)
    log.addHandler(fileh)
    print(f"Predicting {best_weights_path}")
    speeds = []
    
    for i, img_path in enumerate(images):
        try:
            print(f"Processing image {i + 1}/{len(images)}: {img_path}")
            img = cv2.imread(img_path)
            original_img = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Make prediction
            results = model(img, conf=0.1, iou=0.6)
            detections = results[0].boxes

            if detections is not None:
                bboxes = detections.xywhn
                xy = bboxes_to_xy(bboxes)
                if xy is None:
                    raise ValueError("Error: xy is None after processing bboxes")
                
                score = extract_score_from_image(original_img)

                # Save processed image with annotations
                save_path = osp.join(predicted_img_dir, osp.basename(img_path))
                annotated_image = results[0].plot()
                cv2.imwrite(save_path, annotated_image)
                
                # Save score to CSV
                csv_file = f"{recent_results_img_dir}/scores.csv"
                with open(csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([img_path, score])
                
                print(f"Processed image {img_path} with score {score}")
            else:
                print(f"No detections for image {img_path}")
        except Exception as e:
            errors.append([img_path, str(e)])
            print(f"Error processing {img_path}: {e}")
            no_error_total += 1
    
    print("Errors:", errors)
    print(f"Total errors: {len(errors)}, Total no errors: {no_error_total}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_directory = sys.argv[1]
    else:
        model_directory = 'default_model_directory'
    
    predict(model_directory)
