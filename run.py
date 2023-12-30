import cv2
import os
import torch
import click
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from utils import read_yolo_label, yolo_to_coords, coords_to_yolo, find_minimal_rectangles

@click.command()
@click.option('--image_dir', default='datasets/images', help='Directory containing images.')
@click.option('--label_dir', default='datasets/labels', help='Directory containing YOLO format labels.')
@click.option('--refined_label_dir', default='datasets/labels_refine', help='Directory to save refined labels.')
@click.option('--checkpoint', default='sam_vit_h_4b8939.pth', help='Path to the SAM model checkpoint.')
@click.option('--model_type', default='vit_h', help='Type of the SAM model.')
def refine_bounding_boxes(image_dir, label_dir, refined_label_dir, checkpoint, model_type):
    print(f"Refining bounding boxes for images in {image_dir}..., saving to {refined_label_dir}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the Segment Anything model
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    if not os.path.exists(refined_label_dir):
        os.makedirs(refined_label_dir)

    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')

        if not os.path.exists(label_path):
            print(f"Label file not found for {image_name}")
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape[:2]
        predictor.set_image(image)

        labels = read_yolo_label(label_path)
        input_boxes = []

        if len(labels) == 0:
            yolo_bboxes = []
        else:
            for label in labels:
                class_id, x_center, y_center, width, height = label
                rect_coords = yolo_to_coords((x_center, y_center, width, height), image_shape)
                input_boxes.append(rect_coords)

            input_boxes = torch.tensor(input_boxes, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            minimal_rectangles = find_minimal_rectangles(masks[0].cpu().numpy())
            yolo_bboxes = coords_to_yolo(minimal_rectangles, image_shape)

        # Save refined labels
        refined_label_path = os.path.join(refined_label_dir, os.path.splitext(image_name)[0] + '.txt')
        with open(refined_label_path, 'w') as file:
            for idx, yolo_bbox in enumerate(yolo_bboxes):
                if yolo_bbox:
                    class_id = labels[idx][0]
                    file.write(f"{int(class_id)} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")

if __name__ == '__main__':
    refine_bounding_boxes()
