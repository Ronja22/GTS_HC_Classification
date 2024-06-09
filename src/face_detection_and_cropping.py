import os
import logging
import pandas as pd
import numpy as np

import src.utils.directories as dir
import src.utils.videos as vid
import src.face_mesh_utils.face_detection as face_detect

def detect_face_and_crop(input_folder,
                         output_folder,
                         specific_file=None,
                         override = False):
    """
    Detect faces and crop videos from the specified folder.

    Args:
        input_folder (str): Folder containing the input videos.
        output_folder (str): Folder for saving the cropping information
        specific_file (str or None): Specify the video file name to process a specific video from the folder.
                                     Set to None to process all videos in the input folder.

    Returns:
        None
    """
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create the output directory if it doesn't exist
    dir.create_directory_if_not_exists(output_folder)
    logging.info(f"Output folder set to: {output_folder}")
    
    # Get the list of files in the input folder
    video_files = os.listdir(input_folder)
    
    if specific_file:
        logging.info(f"Processing specific file: {specific_file}")
    
    for video_file in video_files:
        # Check if we need to process only a specific file
        if specific_file and specific_file not in video_file:
            continue
        
        # Check if the file is a valid video file
        if not vid.is_video_file(video_file):
            logging.warning(f"Skipping non-video file: {video_file}")
            continue
        
        input_video_path = os.path.join(input_folder, video_file)
        # Check if the cropping CSV file exists
        video_file_without_extension, _ = os.path.splitext(video_file)
        cropping_csv_path = os.path.join(output_folder, video_file_without_extension + "_cropping.csv")
        if os.path.exists(cropping_csv_path) and not override:
            logging.info(f"Skipping {cropping_csv_path}, file already exists.")
            continue

        # Step 1: Face Detection and Cropping
        logging.info(f"Face Detection and Cropping for {video_file}")
        
        try:
            face_detect.face_detection_and_cropping(input_video_path, output_folder)
            logging.info(f"Successfully processed face detection and cropping for {video_file}")
        except Exception as e:
            logging.error(f"Error during face detection and cropping for {video_file}: {e}")
            continue

    logging.info("Face detection and cropping complete.")

