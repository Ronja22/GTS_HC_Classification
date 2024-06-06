import cv2
import os
import matplotlib.pyplot as plt
import src.utils.directories as dir
from moviepy.editor import VideoFileClip

import src.utils.directories as dir

def downsample_video(video_path, output_folder, target_fps, certain_subjects=None, override=False):
    """
    Downsample video(s) to a target frame rate (FPS) and save them in the output folder.
    
    Parameters:
    video_path (str): Path to a single video file or a folder containing multiple video files.
    output_folder (str): Path to the folder where downsampled videos will be saved.
    target_fps (int): The target frame rate to downsample videos to.
    certain_subjects (list of str, optional): List of strings to filter files by their names. Only files containing any of these strings will be processed. Defaults to None.
    override (bool, optional): If False, the function will skip processing files that already exist in the output folder. Defaults to False.
    """
    
    valid_extensions = ['.mp4', '.avi', '.mts', '.m4v']
    
    # Create the output folder if it doesn't exist
    dir.create_directory_if_not_exists(output_folder)
    
    # Determine if video_path is a single file or a directory
    if os.path.isfile(video_path):
        files = [video_path]  # Single file
    else:
        # List of files with valid video extensions in the directory
        files = [os.path.join(video_path, f) for f in os.listdir(video_path) if any(f.lower().endswith(ext) for ext in valid_extensions)]
    
    file_count = len(files)
    print(f"Total files to process: {file_count}")

    for file in files:
        filename = os.path.basename(file)  # Extract the filename from the path
        print(f"Processing: {filename}")

        # Check if certain_subjects is specified and if the filename contains any of the specified subjects
        if certain_subjects:
            if not any(subject in filename for subject in certain_subjects):
                print(f"Skipping {filename}, does not match certain_subjects filter.")
                continue
        
        output_path = os.path.join(output_folder, filename)
        
        # Check if the file already exists in the output folder and if override is False
        if not override and os.path.exists(output_path):
            print(f"File {filename} exists in the output folder, skipping.")
            continue
        
        # Load the input video
        video_clip = VideoFileClip(file)
        
        original_fps = video_clip.fps
        print(f"Original FPS for '{filename}': {original_fps}")
        
        # Downsample the video only if the original FPS is different from the target FPS
        if original_fps != target_fps:
            downsampled_clip = video_clip.set_fps(target_fps)
        else:
            downsampled_clip = video_clip
        
        # Write the video to the output file
        downsampled_clip.write_videofile(output_path, codec='libx264', logger=None)
        
        new_fps = downsampled_clip.fps
        print(f"New FPS for '{filename}': {new_fps}")
        
        # Close the video clips to release resources
        video_clip.close()
        downsampled_clip.close()
    
    print("Downsampling complete.")


def save_frame_from_each_video(input_path, save_path, print_frame=True, print_frame_number=500, override=False):
    """
    Save and optionally print the frame at a specified frame number from video files.

    Args:
        input_path (str or list): Path to a video file, a folder containing video files, or a list of folders containing video files.
        save_path (str): Path where the frames will be saved.
        print_frame (bool, optional): Whether to display the frame. Default is True.
        print_frame_number (int, optional): Frame number to be printed. Default is 500.
        override (bool, optional): If False, existing frames will not be overwritten. Default is False.
    """
    def process_video(video_path):
        """
        Process a single video file to save a specified frame.

        Args:
            video_path (str): Path to the video file.
        """
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        frame_number = 0
        
        # Define the output path for the frame
        output_path = os.path.join(save_path, f"{os.path.splitext(os.path.basename(video_path))[0]}_{print_frame_number}th_frame.jpg")
        
        # Check if the file exists and skip if override is False
        if not override and os.path.exists(output_path):
            print(f"Frame {print_frame_number} of {os.path.basename(video_path)} already exists, skipping.")
            cap.release()
            return
        
        # Read frames from the video
        while True:
            success, frame = cap.read()
            if not success:
                break
            if frame_number == print_frame_number:
                dir.create_directory_if_not_exists(save_path)
                cv2.imwrite(output_path, frame)
                if print_frame:
                    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    plt.axis('off')  # Hide axes
                    plt.show()
                break
            frame_number += 1
        cap.release()

    # Check if input_path is a list
    if isinstance(input_path, list):
        for folder in input_path:
            if os.path.isdir(folder):
                for filename in os.listdir(folder):
                    video_file = os.path.join(folder, filename)
                    if os.path.isfile(video_file):
                        process_video(video_file)
    # Check if input_path is a folder
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            video_file = os.path.join(input_path, filename)
            if os.path.isfile(video_file):
                process_video(video_file)
    # Assume input_path is a single video file
    elif os.path.isfile(input_path):
        process_video(input_path)
    else:
        print("Invalid input path")