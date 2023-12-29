import csv
import cv2
import os
import numpy as np

def random_augmentation(bbox_region):
    """
    Applies random augmentations to the bounding box region.

    :param bbox_region: The region of the bounding box to augment.
    :return: Augmented bounding box region.
    """
    # Randomly adjust brightness
    if np.random.rand() < 0.5:
        value = float(np.random.uniform(0.7, 1.3))  # Ensure the value is a float
        hsv = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = cv2.multiply(hsv[:,:,2], value).astype(hsv.dtype)
        bbox_region = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Randomly apply color jitter
    if np.random.rand() < 0.5:
        value = float(np.random.uniform(0.8, 1.2))  # Ensure the value is a float
        multiplier = np.ones_like(bbox_region) * value
        multiplier = multiplier.astype(bbox_region.dtype)
        bbox_region = cv2.multiply(bbox_region, multiplier)

    # Random rotation
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-10, 10)  # random rotation between -10 to 10 degrees
        center = (bbox_region.shape[1]//2, bbox_region.shape[0]//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        bbox_region = cv2.warpAffine(bbox_region, rot_mat, bbox_region.shape[1::-1], flags=cv2.INTER_LINEAR)

    # Random blur to simulate motion or focus changes
    if np.random.rand() < 0.5:
        ksize = np.random.choice([3, 5])  # Kernel size can be adjusted
        bbox_region = cv2.GaussianBlur(bbox_region, (ksize, ksize), 0)

    # Random perspective shift
    if np.random.rand() < 0.5:
        pts1 = np.float32([[0, 0], [bbox_region.shape[1], 0], [0, bbox_region.shape[0]], [bbox_region.shape[1], bbox_region.shape[0]]])
        shift = bbox_region.shape[0] * 0.1  # Max shift of 10% of the bbox size
        pts2 = np.float32([[np.random.uniform(-shift, shift), np.random.uniform(-shift, shift)],
                           [bbox_region.shape[1] - np.random.uniform(-shift, shift), np.random.uniform(-shift, shift)],
                           [np.random.uniform(-shift, shift), bbox_region.shape[0] - np.random.uniform(-shift, shift)],
                           [bbox_region.shape[1] - np.random.uniform(-shift, shift), bbox_region.shape[0] - np.random.uniform(-shift, shift)]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        bbox_region = cv2.warpPerspective(bbox_region, matrix, (bbox_region.shape[1], bbox_region.shape[0]))

    return bbox_region



def superimpose_bbox(frame1, bbox1, frame2):
    """
    Extracts a bounding box from frame1, applies random augmentation,
    and superimposes it onto frame2.

    :param frame1: The original frame from which the bbox will be extracted.
    :param bbox1: A tuple of (x_min, y_min, x_max, y_max) for the bounding box in frame1.
    :param frame2: The frame onto which the extracted bbox will be superimposed.
    :return: frame2 with the augmented bbox from frame1 superimposed.
    """
    x_min, y_min, x_max, y_max = map(int, bbox1)
    bbox_region = frame1[y_min:y_max, x_min:x_max]

    # Apply random augmentation to the bbox_region
    augmented_bbox = random_augmentation(bbox_region)

    # Superimpose the augmented bbox_region onto frame2
    frame2[y_min:y_min+augmented_bbox.shape[0], x_min:x_min+augmented_bbox.shape[1]] = augmented_bbox

    return frame2


def superimpose_mask(frame, mask, target_frame):
    # Pastes the mask in target_frame
    
    # Ensure mask is boolean
    mask = mask.astype(bool)

    # Copy target frame
    target_frame_2 = target_frame.copy()

    # Apply mask to the frame (assuming frame and target_frame are of the same shape)
    for c in range(3):  # Loop over color channels
        target_frame_2[:, :, c][mask] = frame[:, :, c][mask]

    return target_frame_2


import csv

def update_annotation_file(dest_annotation_file, activity, frame_num, bbox, label, person_id):
    # Open the file in append mode ('a') or write mode ('w')
    with open(dest_annotation_file, 'a', newline='') as file:  # 'a' can be replaced with 'w' based on your requirement
        writer = csv.writer(file)
        writer.writerow([activity, frame_num, bbox[0], bbox[1], bbox[2], bbox[3], label, person_id])

def bbox_within_background(action_bbox, background_bbox):
    # Check if action_bbox is completely inside background_bbox
    ax_min, ay_min, ax_max, ay_max = action_bbox
    bx_min, by_min, bx_max, by_max = background_bbox
    return ax_min >= bx_min and ay_min >= by_min and ax_max <= bx_max and ay_max <= by_max


def get_unique_video_names(csv_path):
    unique_videos = set()

    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            video_name = row[0]
            unique_videos.add(video_name)

    return list(unique_videos)
