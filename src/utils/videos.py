import cv2
import os
import matplotlib.pyplot as plt
import src.utils.directories as dir

def save_frame_from_each_video(input_path, save_path, print_frame=True, print_frame_number=500):
    """
    Save and optionally print the frame at a specified frame number from video files.

    Args:
        input_path (str or list): Path to a video file, a folder containing video files, or a list of folders containing video files.
        save_path (str): Path where the frames will be saved.
        print_frame (bool, optional): Whether to display the frame. Default is True.
        print_frame_number (int, optional): Frame number to be printed. Default is 500.
    """
    # Function to process a single video file
    def process_video(video_path):
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        frame_number = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            if frame_number == print_frame_number:
                output_path = os.path.join(save_path, f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_number}th_frame.jpg")
                
                dir.create_directory_if_not_exists(save_path)
                cv2.imwrite(output_path, frame)
                if print_frame:
                    plt.imshow(frame)
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