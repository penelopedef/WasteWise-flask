# object_detection.py

# Install required packages
# !pip install ultralytics
# !pip install opencv-python

# Import libraries
import os
import cv2
from ultralytics import YOLO
import pathlib

# Load the YOLO model
test_model = YOLO('C:/Users/penel/Documents/Full stack/Waste Project/best.pt')

# Function to get predictions
def get_pred(image_path):
    pred = test_model.predict([image_path], save=True, imgsz=416, conf=0.5, iou=0.7)
    return pred

# Function to extract bounding boxes
def get_boundingboxes(pred_results):
    # Iterate over each element (result) in the results list
    for result in pred_results:
        # Access the bounding boxes for the current result
        boxes = result.boxes

        # Iterate over each bounding box in the current result
        for bbox in boxes.xyxy:
            # Access the bounding box coordinates
            xmin, ymin, xmax, ymax = bbox[:4]

            # Return the bounding box coordinates
            return (xmin, ymin, xmax, ymax)

# Function to get class labels and confidence scores
def get_class_and_confidence(pred_results):
    class_and_conf = []
    # Iterate over each prediction result in the list
    for result in pred_results:
        # Access the class labels for the current result
        class_labels = result.names

        # Access the bounding box coordinates, confidence scores, and class labels
        boxes = result.boxes

        # Iterate over each bounding box in the current result
        for bbox, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            # Extract the class index and confidence score
            class_index = int(cls)
            confidence_score = conf

            # Get the predicted label from the class labels
            predicted_label = class_labels[class_index]

            # Append the predicted label and confidence score to the list
            class_and_conf.append(predicted_label + ',' + str(confidence_score))
            
    return class_and_conf

# Function to format results
def format_results(bboxes, class_and_conf):
    
    # Extract the class label and confidence score from class_and_conf list
    class_label, confidence_score = class_and_conf[0].split(',')
    
    # Extract the values from bboxes tensor
    xmin, ymin, xmax, ymax = bboxes

    # Convert the values to strings
    xmin_str = str(xmin.item())
    ymin_str = str(ymin.item())
    xmax_str = str(xmax.item())
    ymax_str = str(ymax.item())

    # Remove any unwanted characters
    class_label = class_label.strip()
    confidence_score = confidence_score.strip().lstrip('tensor(').rstrip(')')

    # Concatenate all values together with commas
    formatted_result = ','.join([xmin_str, ymin_str, xmax_str, ymax_str, class_label, confidence_score])

    return formatted_result

# Function to get results
def get_results(image_path):
    pred_results = get_pred(image_path)
    
    bboxes = get_boundingboxes(pred_results)
    class_and_conf = get_class_and_confidence(pred_results)
    
    if not bboxes:
        return 'No detection'
    
    results = format_results(bboxes, class_and_conf)
    
    return results # returned as a string 'xmin_str, ymin_str, xmax_str, ymax_str, class_label, confidence_score'

# Example usage
if __name__ == "__main__":
    results = get_results('C:/Users/penel/Documents/Full stack/Waste Project/sample images/image0.jpg')
    print(results)
