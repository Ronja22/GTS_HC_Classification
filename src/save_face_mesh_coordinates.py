import os
import pandas as pd
import numpy as np
import logging
import traceback

import src.utils.directories as dir
import src.utils.videos as vid
import src.face_mesh_utils.face_mesh as face_mesh

def save_facemesh_coordinates(input_folder,
                              input_cropping_folder,
                              output_coordinate_path,
                              specific_file=None):
    """
    Save face mesh coordinates from videos based on specified parameters.

    Args:
        input_folder (str): Folder containing the input videos.
        input_cropping_folder (str): Folder containing the cropping parameters.
        output_coordinate_path (str): Folder for saving face mesh coordinates.
        specific_file (str or None): Specify the video file name to process a specific video from the folder.
                                     Set to None to process all videos in the input folder.

    """
   
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Iterate through each video in the input folder
        for video_file in os.listdir(input_folder):
            # Check if a specific video file is provided
            if specific_file and specific_file not in video_file:
                continue
            
            # Check if video_file is indeed a video file
            if vid.is_video_file(video_file):
                logging.info(f"Processing {video_file}")

                # Set paths
                input_video_path = os.path.join(input_folder, video_file)
                root, extension = os.path.splitext(input_video_path)
                video_file_without_extension, _ = os.path.splitext(video_file)
                
                # Check if the cropping CSV file exists
                cropping_csv_path = os.path.join(input_cropping_folder, video_file_without_extension + "_cropping.csv")
                
                if not os.path.exists(cropping_csv_path):
                    logging.error(f"No cropping parameters for {video_file}. "
                                  "Please check that the coordinate path is correct or run "
                                  "00_main_face_detection_and_cropping.py to create the cropping parameters.")
                    continue

                # Load cropping parameters
                cropping = pd.read_csv(cropping_csv_path)
                cropping_values = cropping[['origin_y', 'end_y', 'origin_x', 'end_x']].values[0]

                # Save eye coordinates based on face mesh
                try:
                    face_mesh.save_coordinates(input_video_path, output_coordinate_path, cropping_values)
                    logging.info(f"Saved face mesh coordinates for {video_file}")
                except Exception as e:
                    logging.error(f"Error saving face mesh coordinates for {video_file}: {e}")
                    raise  # Raise the exception to stop execution

    except Exception as e:
        logging.error("An error occurred during processing.")
        logging.error(traceback.format_exc())  # Print the full traceback

