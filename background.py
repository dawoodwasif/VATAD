import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool,Manager


def load_annotations(annotation_file, video_folder):
    # Load annotations from the file
    annotations = pd.read_csv(annotation_file, header=None)
    # Name the columns
    annotations.columns = ['video_name', 'frame', 'x1', 'y1', 'x2', 'y2', 'class_label', 'person_id']
    # Filter rows where video_name matches video_folder
    filtered_annotations = annotations[annotations['video_name'] == video_folder]
    return filtered_annotations


def convert_bbox_format(annotations, frame_width, frame_height):
    # Convert normalized coordinates to pixel coordinates
    annotations[['x1', 'x2']] *= frame_width
    annotations[['y1', 'y2']] *= frame_height
    return annotations

def invert_bbox_format(annotations, frame_width, frame_height):
    # Convert normalized coordinates to pixel coordinates
    annotations[['x1', 'x2']] /= frame_width
    annotations[['y1', 'y2']] /= frame_height
    return annotations

########################

def find_largest_bbox(largest_bboxes,annotations, frame_shape,start_frame,frame_interval):
    # Initialize a mask for the frame with all zeros (no bounding box)
    mask = np.zeros(frame_shape, dtype=np.uint8)    

    # Mark the regions covered by the existing bounding boxes in the mask
    for _, row in annotations.iterrows():
        mask[int(row['y1']):int(row['y2']), int(row['x1']):int(row['x2'])] = 1

    def find_max_rectangle_from_point(x, y):
        # Initialize maximum width and height
        max_width = max_height = 0

        # Find maximum height (column-wise)
        for h in range(y, frame_shape[0]):
            if mask[h, x] == 1:
                break
            max_height += 1

        # Find maximum width (row-wise)
        for w in range(x, frame_shape[1]):
            if mask[y, w] == 1:
                break
            # Check if the entire column up to max_height is free
            if np.any(mask[y:y+max_height, w] == 1):
                break
            max_width += 1

        return max_width, max_height

    # Define the largest bounding box found
    largest_bbox = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
    largest_area = 0

    # Iterate over the mask to find the largest bounding box
    y = x = 0
    while y < frame_shape[0] and x < frame_shape[1]:
        # Skip over marked areas
        if mask[y, x] == 1:
            x += 1
            if x >= frame_shape[1]:
                x = 0
                y += 1
            continue

        # Find the maximum rectangle from the current point
        width, height = find_max_rectangle_from_point(x, y)
        area = width * height
        # Update if this is the largest area found so far
        if area > largest_area:
            largest_area = area
            largest_bbox = {'x1': x, 'y1': y, 'x2': x + width, 'y2': y + height}

        # Move to the next point
        x += width
        if x >= frame_shape[1]:
            x = 0
            y += 1
            
    largest_bbox_output = (
    largest_bbox['x1'],
    largest_bbox['y1'],
    largest_bbox['x2'],
    largest_bbox['y2']
)   
    if largest_bbox_output:
        largest_bboxes.append(((start_frame, start_frame + frame_interval - 1), largest_bbox_output))

##########################


def visualize_bboxes(image, bbox, annotations, frame_number):
    
    # Filter annotations for the specific frame number
    frame_annotations = annotations[annotations['frame'] == frame_number]

    # Loop through each annotation and draw a rectangle on the image
    for index, row in frame_annotations.iterrows():
        color = (0, 255, 0)  # Alternate colors for visual distinction
        cv2.rectangle(image, (int(row['x1']), int(row['y1'])), (int(row['x2']), int(row['y2'])), color, 2)

    # Draw a rectangle around the largest bounding box
    # The bbox is expected to be a tuple or list with (x1, y1, x2, y2) format
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
    
    # Display the image with the bounding box
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def get_frame_shape_from_folder(folder_path):
    # Assumes the first image is representative of all frame sizes
    first_image_path = os.path.join(folder_path, sorted(os.listdir(folder_path))[1])
    image = cv2.imread(first_image_path)
    return image.shape[:2]

def detect_background_bbox(folder_name, annotation_file, m, video_folder):
    frame_interval = m
    frame_shape = get_frame_shape_from_folder(folder_name)
    annotations = load_annotations(annotation_file, video_folder)
    annotations = convert_bbox_format(annotations, frame_shape[1], frame_shape[0])
    p = Pool()
    m = Manager()
    largest_bboxes = m.list()
    for start_frame in range(0, annotations['frame'].max(), frame_interval):
        frame_annotations = annotations[(annotations['frame'] >= start_frame) & (annotations['frame'] < start_frame + frame_interval)]
        if not frame_annotations.empty:
            # multiprocessing for every m frames
            p.apply_async(find_largest_bbox,args= [largest_bboxes,frame_annotations,frame_shape,start_frame,frame_interval])
        
    p.close()
    p.join()
    return largest_bboxes

