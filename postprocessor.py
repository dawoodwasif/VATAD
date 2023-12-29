import pandas as pd
import os
import shutil
from utils import get_unique_video_names


def sync_frames(video_names, base_path_original, base_path_augmented):
    """
    Synchronizes frames between original and augmented folders for each video.
    If the augmented folder does not exist, the whole original folder is copied.
    Otherwise, missing frames in the augmented folder are copied from the original folder.

    :param video_names: List of video names to process.
    :param base_path_original: Base path where original frames are stored.
    :param base_path_augmented: Base path where augmented frames are stored.
    """
    for video in video_names:
        # Construct paths for original and augmented folders
        original_folder = os.path.join(base_path_original, video)
        augmented_folder = os.path.join(base_path_augmented, video)

        # Check if original folder exists
        if not os.path.exists(original_folder):
            print(f"Missing original folder for video: {video}")
            continue

        # If augmented folder doesn't exist, copy the entire original folder
        if not os.path.exists(augmented_folder):
            shutil.copytree(original_folder, augmented_folder)
            print(f"Copied entire folder for video: {video}")
        else:
            # Sync frames from original to augmented
            for frame in os.listdir(original_folder):
                original_frame_path = os.path.join(original_folder, frame)
                augmented_frame_path = os.path.join(augmented_folder, frame)

                # Copy missing frames to augmented folder
                if not os.path.exists(augmented_frame_path):
                    shutil.copy2(original_frame_path, augmented_frame_path)



def merge_ava_annotations(file1, file2, output_file):
    """
    Merges two AVA annotation files, sorts the merged file in ascending order by frame number,
    and ensures the data types are maintained.

    :param file1: Path to the first AVA annotation file.
    :param file2: Path to the second AVA annotation file.
    :param output_file: Path where the merged and sorted file will be saved.
    """
    # Read the CSV files
    df1 = pd.read_csv(file1, dtype=str, header=None)
    df2 = pd.read_csv(file2, dtype=str, header=None)

    # Merging the dataframes
    merged_df = pd.concat([df1, df2])
    
    # Sort the merged dataframe by frame number
    # sorted_df = merged_df.sort_values(by=df1.columns[1])

    # # Reset index after sorting
    # sorted_df.reset_index(drop=True, inplace=True)

    # Saving the merged and sorted dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False,  header=False)




original_annotation_file = '/home/ubuntu/Dawood/activity_dataset/weller_only/20231011_weller_only_combined_valtest_ke_activity_train_avastyle_17.csv'
augmented_annotation_file = '/home/ubuntu/Dawood/Augmentation/activity_dataset/normalized_merged_train_annotations.csv'    
merged_annotation_file = '/home/ubuntu/Dawood/Augmentation/activity_dataset/final_train_annotations.csv'    

base_path_original = '/home/ubuntu/Dawood/activity_dataset/20231004-activity-all-annotated-tasks'
base_path_augmented = '/home/ubuntu/Dawood/Augmentation/activity_dataset/frames'


unique_video_names = get_unique_video_names(original_annotation_file)

print("Total Videos Sampled: " + str(len(unique_video_names)))

merge_ava_annotations(original_annotation_file, augmented_annotation_file, merged_annotation_file)

print("Annotation merging completed")

#sync_frames(unique_video_names, base_path_original, base_path_augmented)

print("Frame Synchronization completed")
