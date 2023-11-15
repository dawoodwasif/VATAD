import csv
def superimpose_mask(frame, mask,target_frame):
    """
    Pastes the mask in target_frame
    """
    q= frame * mask
    target_frame[mask==True] = q[q>0]
    return target_frame

def update_annotation_file(annot_file,activity,frame_num,bbox,label,person_id):
    file = open(annot_file)
    writer = csv.writer(file)
    writer.writerow([activity,frame_num,bbox[0],bbox[1],bbox[2],bbox[3],label,person_id])

def bbox_within_background(action_bbox, background_bbox):
    # Check if action_bbox is completely inside background_bbox
    ax_min, ay_min, ax_max, ay_max = action_bbox
    bx_min, by_min, bx_max, by_max = background_bbox
    return ax_min >= bx_min and ay_min >= by_min and ax_max <= bx_max and ay_max <= by_max