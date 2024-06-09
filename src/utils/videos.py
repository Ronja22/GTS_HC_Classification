import cv2
import os
import matplotlib.pyplot as plt
import src.utils.directories as dir
from moviepy.editor import VideoFileClip

import src.utils.directories as dir

def get_video_dimensions(input_video_path):
    """
    Retrieves the dimensions (width and height) of a video file.

    Parameters:
        input_video_path (str): The path to the input video file.

    Returns:
        tuple: A tuple containing the width and height of the video.
            - Width (int): The width of the video in pixels.
            - Height (int): The height of the video in pixels.
        If the video file cannot be opened or an error occurs, returns None.
    """
    # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)

    # Check if the video is successfully opened
    if not video_capture.isOpened():
        print("Error: Unable to open the video file.")
        return None

    # Get the video width and height
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Release the video capture object
    video_capture.release()

    return width, height


def get_frame_from_video(filepath, frame_number):
    """
    Retrieves a specific frame from a video file.

    Parameters:
        filepath (str): The path to the video file.
        frame_number (int): The index of the desired frame to retrieve.

    Returns:
        numpy.ndarray or None: The image frame as a NumPy array if successful, or None if there was an error.

    Raises:
        FrameLoadError: If the desired frame cannot be read from the video file.

    Note:
        This function uses the OpenCV library to read and extract frames from the video file.
    """

    # Open the video file
    video = cv2.VideoCapture(filepath)
    class FrameLoadError(Exception):
        pass

    # Check if the video was opened successfully
    if not video.isOpened():
        print("Fehler beim Öffnen des Videos.")
        return None

    # Set the desired frame index
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame from the video
    ret, frame = video.read()

    # Release the video object
    video.release()

    # Check if the frame was successfully read
    if not ret:
        raise FrameLoadError(f"Konnte Frame {frame_number} nicht lesen.")

    return frame

def is_video_file(filename):
    """
    Checks if the given filename corresponds to a video file based on its extension.
    
    Parameters:
        filename (str): The name of the file to check.
        
    Returns:
        bool: True if the file is a video file, False otherwise.
    """
    # Get the file extension
    _, extension = os.path.splitext(filename)
    # Check if the file extension corresponds to a video format
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg', '.webm', ".mts"]
    return extension.lower() in video_extensions

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