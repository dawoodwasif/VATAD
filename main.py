import cv2
import os
import torch
import yaml
import random

from background import detect_background_bbox, load_annotations, convert_bbox_format, get_frame_shape_from_folder
from utils import superimpose_mask, bbox_within_background, update_annotation_file
from mask_extraction import generate_action_mask
from segment_anything import sam_model_registry

def read_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def weighted_random_label(label_weights):
    labels, weights = zip(*[(label['id'], label['weight']) for label in label_weights])
    return random.choices(labels, weights)[0]

def process_video(dataset_folder, video_folder, annotation_file, m, action_label, sam_model):
    video_path = os.path.join(dataset_folder, video_folder)
    frame_shape = get_frame_shape_from_folder(video_path)

    # Detect background bounding boxes for m frames
    background_bboxes = detect_background_bbox(video_path, annotation_file, m, video_folder)

    # Load and process annotations
    annotations = load_annotations(annotation_file, video_folder)
    annotations = convert_bbox_format(annotations, frame_shape[1], frame_shape[0])
    
    # Iterate through the background bounding boxes
    for frame_interval, background_bbox in background_bboxes:
        start_frame, end_frame = frame_interval
        target_frames = list(range(start_frame, end_frame + 1))

        for index, row in annotations.iterrows():
            frame_number = row['frame']
            if not (start_frame <= frame_number <= end_frame):  # Actions outside the interval
                class_id = row['class_label']
                person_id = row['person_id']
                if class_id == action_label:
                    action_bbox = (row['x1'], row['y1'], row['x2'], row['y2'])
                    if bbox_within_background(action_bbox, background_bbox):
                        frame_path = os.path.join(video_path, f'img_{int(frame_number):05d}.jpg')
                        input_frame = cv2.imread(frame_path)

                        mask = generate_action_mask(sam_model, frame_path, class_id, action_bbox)

                        if target_frames:
                            target_frame_number = target_frames.pop(0)
                            target_frame_path = os.path.join(video_path, f'img_{int(target_frame_number):05d}.jpg')
                            target_frame = cv2.imread(target_frame_path)
                            
                            superimposed_frame = superimpose_mask(input_frame, mask, target_frame)
                            cv2.imwrite(os.path.join(video_path,f'img_{int(target_frame_number):05d}.jpg'),superimposed_frame)
                            update_annotation_file(annotation_file,video_folder,target_frame_number,action_bbox,class_id,person_id)
                            

if __name__ == '__main__':
    config = read_config('config.yaml')

    # Determine device for model
    device = torch.device(config['model']['device'])
    if config['model']['device'] == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize SAM model
    sam = sam_model_registry[config['model']['model_type']](checkpoint=config['paths']['checkpoint_path']).to(device=device)

    # Update processing with weighted random label selection
    target_action_label_id = weighted_random_label(config['labels']['label_weights'])

    # Arguments from the YAML file
    dataset_path = config['paths']['dataset_path']
    video_name = config['processing']['video_name']
    annotation_file = config['paths']['annotation_file']
    m = config['processing']['frame_numbers']

    process_video(dataset_path, video_name, annotation_file, m, target_action_label_id, sam)



