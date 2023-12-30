import streamlit as st
import cv2
import os
from utils import read_yolo_label, draw_bounding_boxes 

# Streamlit interface
st.title('YOLO Bounding Box Visualizer')

# Input fields for directory paths
image_dir = st.text_input('Image Directory', 'datasets/images')
label_dir = st.text_input('Label Directory', 'datasets/labels')
refined_label_dir = st.text_input('Refined Label Directory', 'datasets/labels_refine')

if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
    st.error('Invalid directory path(s)')
else:
    image_files = sorted(os.listdir(image_dir))
    if image_files:
        # Choose an index to visualize
        max_index = len(image_files) - 1
        index = st.number_input('Image Index', min_value=0, max_value=max_index, value=0, step=1)

        image_name = image_files[index]
        image_path = os.path.join(image_dir, image_name)
        original_label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
        refined_label_path = os.path.join(refined_label_dir, os.path.splitext(image_name)[0] + '.txt')

        col1, col2 = st.columns(2)

        # Display original bounding boxes
        if os.path.exists(original_label_path):
            original_labels = read_yolo_label(original_label_path)
            original_image_with_boxes = draw_bounding_boxes(image_path, original_labels)
            col1.image(original_image_with_boxes, caption="Original Bounding Boxes", use_column_width=True)
        else:
            st.error('Label file not found')

        # Display refined bounding boxes
        if os.path.exists(refined_label_path):
            refined_labels = read_yolo_label(refined_label_path)
            refined_image_with_boxes = draw_bounding_boxes(image_path, refined_labels)
            col2.image(refined_image_with_boxes, caption="Refined Bounding Boxes", use_column_width=True)
        else:
            st.error('Refined label file not found')
    else:
        st.error('No images found in the specified directory')
