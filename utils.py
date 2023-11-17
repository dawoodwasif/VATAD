import csv


import cv2
import os

# def superimpose_mask(frame, mask,target_frame):
#     """
#     Pastes the mask in target_frame
#     """
#     # dest_path = "/home/ubuntu/Furqan/Rumman/VATAD/data/temp"
#     # cv2.imwrite(os.path.join(dest_path,f'img_mask.jpg'),mask)
#     # cv2.imwrite(os.path.join(dest_path,f'img_frmae.jpg'),frame)
#     # cv2.imwrite(os.path.join(dest_path,f'img_tg_grame.jpg'),target_frame)
#     print(target_frame.shape, mask.shape)
#     target_frame_2 = target_frame.copy()
#     q= frame * mask
#     target_frame_2[mask==True] = q[q>0]
#     return target_frame_2

def superimpose_mask(frame, mask, target_frame):
    """
    Pastes the mask in target_frame
    """
    print(target_frame.shape, mask.shape)
    
    # Ensure mask is boolean
    mask = mask.astype(bool)

    # Copy target frame
    target_frame_2 = target_frame.copy()

    # Apply mask to the frame (assuming frame and target_frame are of the same shape)
    for c in range(3):  # Loop over color channels
        target_frame_2[:, :, c][mask] = frame[:, :, c][mask]

    return target_frame_2


def update_annotation_file(annot_file,activity,frame_num,bbox,label,person_id):
    file = open(annot_file)
    writer = csv.writer(file)
    writer.writerow([activity,frame_num,bbox[0],bbox[1],bbox[2],bbox[3],label,person_id])

def bbox_within_background(action_bbox, background_bbox):
    # Check if action_bbox is completely inside background_bbox
    ax_min, ay_min, ax_max, ay_max = action_bbox
    bx_min, by_min, bx_max, by_max = background_bbox
    return ax_min >= bx_min and ay_min >= by_min and ax_max <= bx_max and ay_max <= by_max