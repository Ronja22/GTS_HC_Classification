import numpy as np
import pandas as pd
import os

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

import src.utils.directories as dir

def set_start_and_endpoint(filename, video_duration):
    """
    Returns the start and end points of the subclip in seconds based on the filename and video duration.

    Parameters:
        filename (str): The name of the video file.
        video_duration (float): The duration of the video in seconds.

    Returns:
        tuple: A tuple containing the start and end points in seconds.
    """
    
    # Default values for start and end points
    startpoint = 450
    endpoint = 600

    # Define segments for specific video types
    if "P2a" in filename:
        # For most P2a videos, the desired segment is in the last 2.5 minutes
        startpoint = 450
        endpoint = 600
    elif "P3a" in filename:
        # For most P3a videos, the desired segment is in the second 2.5 minutes
        startpoint = 150
        endpoint = 300
    elif "P3b" in filename:
        # For most P3b videos, the desired segment is in the last 2.5 minutes
        startpoint = 450
        endpoint = 600
    elif "HYPE" in filename:
        if video_duration >= 550:
            # For most HYPE videos longer than 9 minutes, the desired segment is in the last 2.5 minutes
            startpoint = 450
            endpoint = 600
        elif video_duration >= 250:
            # For shorter HYPE videos longer than 4 minutes but less than 9 minutes, the desired segment is in the second 2.5 minutes
            startpoint = 150
            endpoint = 300

    # Handle exceptions to the general rules
    if "GTS_P2a_007_T2" in filename:
        # Specific case for GTS_P2a_007_T2, desired segment is in the second 2.5 minutes
        startpoint = 150
        endpoint = 300
    elif "HC_P3a_001" in filename:
        # Specific case for HC_P3a_001, desired segment is the first 2.5 minutes
        startpoint = 0
        endpoint = 150
    elif "HC_P3b_002_RUSH2_T2" in filename or "HC_P3b_002_RUSH1_T2" in filename:
        # Specific cases for HC_P3b_002_RUSH2_T2 and HC_P3b_002_RUSH1_T2, desired segment is in the second 2.5 minutes
        startpoint = 150
        endpoint = 300

    # Ensure the endpoint does not exceed the video duration
    if endpoint > video_duration:
        endpoint = video_duration

    return startpoint, endpoint

def cut_single_video(video_path, savepath, start_time=None, end_time=None):
    """
    Cuts a segment from a video based on predefined durations and saves it.

    Parameters:
        videopath (str): Path to the original video file.
        savepath (str): Directory where the cut video will be saved.
        start_time (int, optional): Start time in seconds for cutting the video. Defaults to None.
        end_time (int, optional): End time in seconds for cutting the video. Defaults to None.
    
    Returns:
        None
    """

    # Load the original video clip
    original_clip = VideoFileClip(video_path)

    # Get the duration of the video in seconds
    video_duration = original_clip.duration

    # Define the start and end times based on video duration if not provided
    if start_time is None or end_time is None:
        start_time, end_time = set_start_and_endpoint(video_path, video_duration)

    # Print the cutting range
    print(f"Cutting {os.path.basename(video_path).split('.')[0]} from {start_time} to {end_time}")


    # Create the target file name
    targetname = os.path.join(savepath, os.path.basename(video_path).split(".")[0] + "_cut.mp4")

    # Create the subclip with specified start and end times
    subclip = original_clip.subclip(start_time, end_time)

    # Save the subclip
    subclip.write_videofile(targetname, codec="libx264", verbose=False)

def cut_videos_from_folders(video_folders, save_path, certain_subjects, override=False):
    """
    Cut videos from multiple folders based on certain subjects.

    Args:
        video_folders (list): List of folders containing videos to be cut.
        save_path (str): Path where the cut videos will be saved.
        certain_subjects (list): List of strings representing certain subjects to filter videos.
        override (bool, optional): If False, skip videos already cut in the save folder. Default is False.
    """
    # Create the save path directory if it doesn't exist
    dir.create_directory_if_not_exists(save_path)
    
    # Iterate through each folder of videos
    for folder in video_folders:
        # Iterate through each file in the folder
        for filename in os.listdir(folder):
            print(f"Processing video from {folder}: {filename}")
            print("Selected subjects:", certain_subjects)
            
            # If no certain subjects provided, set a default value
            if not certain_subjects:
                certain_subjects = ["_"]
                
            # Check if the filename contains any of the specified subjects
            if any(subject in filename for subject in certain_subjects):
                print("Found video for processing:", filename)
                
                # If override is False and the file already exists in the save path, skip it
                if not override and os.path.exists(os.path.join(save_path, os.path.basename(filename).split(".")[0] + "_cut.mp4")):
                    print("File already exists, skipping:", filename)
                    continue

                # Call the function to cut the video
                cut_single_video(os.path.join(folder, filename), save_path)

