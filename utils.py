import numpy as np
import cv2

# Function to read YOLO labels
def read_yolo_label(label_path):
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            if line == '\n':
                continue
            class_id, x_center, y_center, width, height = map(float, line.split())
            labels.append((class_id, x_center, y_center, width, height))
    return labels

# Function to convert YOLO format to rectangle coordinates
def yolo_to_coords(yolo_coords, image_shape):
    x_center, y_center, width, height = yolo_coords
    x_center *= image_shape[1]
    y_center *= image_shape[0]
    width *= image_shape[1]
    height *= image_shape[0]

    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    return (x_min, y_min, x_max, y_max)

# Function to convert rectangle coordinates to YOLO format
def coords_to_yolo(bboxes, image_shape):
    yolo_bboxes = []
    for bbox in bboxes:
        if bbox is None:
            continue

        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        x_center /= image_shape[1]  # Normalize by image width
        y_center /= image_shape[0]  # Normalize by image height
        width /= image_shape[1]
        height /= image_shape[0]

        yolo_bboxes.append((x_center, y_center, width, height))

    return yolo_bboxes

# Function to find minimal bounding rectangle from a segmentation mask
def find_minimal_rectangles(masks):
    minimal_rectangles = []
    for mask in masks:
        # Convert to uint8 
        mask = (mask * 255).astype(np.uint8)

        # Apply threshold to convert mask to binary format
        _, binary_mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            bounding_rects = [cv2.boundingRect(c) for c in contours]
            x_min = min([x for x, y, w, h in bounding_rects])
            y_min = min([y for x, y, w, h in bounding_rects])
            x_max = max([x + w for x, y, w, h in bounding_rects])
            y_max = max([y + h for x, y, w, h in bounding_rects])

            minimal_rectangles.append((x_min, y_min, x_max, y_max))
        else:
            minimal_rectangles.append(None)

    return minimal_rectangles



# Function to draw bounding boxes on an image
def draw_bounding_boxes(image_path, labels):
    image = cv2.imread(image_path)
    for label in labels:
        _, x_center, y_center, width, height = label
        x_min, y_min, x_max, y_max = yolo_to_coords((x_center, y_center, width, height), image.shape)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    return image