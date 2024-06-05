import numpy as np
import helper_functions as hf
import pandas as pd
import os
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

def cut_videos(videopath,savepath,start_time, end_time):

    # Load the original video clip
    original_clip = VideoFileClip(videopath)

    # Get the duration of the video in seconds
    video_duration = original_clip.duration

    # Define the start and end times based on video duration
    if video_duration >= 550:
        start_time = 450
        end_time = 600
    elif video_duration >= 250:
        start_time = 150
        end_time = 300
    else:
        print("Video is too short to cut.")
        return
    if end_time> video_duration:
        end_time = video_duration
    print("cutting from to",start_time, end_time)
    targetname = os.path.join(savepath, os.path.basename(videopath).split(".")[0] + ".mp4")

    # Create the subclip with specified start and end times
    subclip = original_clip.subclip(start_time, end_time)

    # Save the subclip
    subclip.write_videofile(targetname, codec="libx264", verbose=False)


    
def save_frame_from_each_video(video_path, print_frame = True, print_frame_number = 500):
    # prints the print_frame_numbe-th frame for each video in folder

    
    if video_path.endswith(".mp4") or video_path.endswith(".avi"):

        cap = cv2.VideoCapture(video_path)
        frame_number = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            if frame_number == print_frame_number:
                output_path = video_path.split(".")[0] + str(frame_number) + "th_frame.jpg"
                cv2.imwrite(output_path, frame)

                # if print_frame = True, plot frames directely
                if print_frame == True:
                    plt.imshow(frame)
                    plt.show()

                break
            frame_number += 1
        cap.release()    
        
import shutil
def save_video_to_path(folder, filename, savepath):
    # Create the full source file path
    source_file_path = os.path.join(folder, filename)
    
    # Create the full destination file path
    destination_file_path = os.path.join(savepath, filename)
    
    # Check if the source file exists
    if os.path.exists(source_file_path):
        try:
            # Copy the file to the destination path
            shutil.copy(source_file_path, destination_file_path)
            print(f"Video saved to {destination_file_path}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Source file {source_file_path} does not exist.")





def cut_all_hype_videos(video_folder, save_path, certain_subjects, override = False):
    # This function checks from where to where the RUSH videos have to be cut
    # it also calls the cut function and cuts the videos. if override == false it skips videos wwhich are already cut under the save folder.
    # it also saves one frame for each file to check that they are cut correctly.
    hf.create_path_if_not_existent(save_path)
    
    # for each folder of videos
    for folder in video_folder:
        for filename in os.listdir(folder):
            
            print("video from", folder + os.path.basename(filename).split(".")[0]  + ".mp4")
            #print(certain_subjects)
            if certain_subjects == []:
                certain_subjects = ["_"]
            if any(string in filename for string in certain_subjects):
                print("Filename:", filename)
                if override == False:
                    if os.path.exists(save_path + os.path.basename(filename).split(".")[0]  + ".mp4"):
                        print("file " + filename  + " exists, skipping")
                        continue

                #get start and endpoints
                startpoint = 450
                endpoint = 600
                
                
                if "BUD" in filename:
                    save_video_to_path(folder,filename,save_path)
                else:
                    #call cut rush function
                    cut_videos(folder + filename, save_path, startpoint, endpoint)

                #call saveframe and plot frame also
                save_frame_from_each_video(save_path + filename.split(".")[0] + ".mp4")

                
                

def save_as_mp4(folder_name, save_folder):
    # Check if the output folder exists. If not, create it.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Iterate through the input folder and convert MTS videos to MP4.
    for root, _, files in os.walk(folder_name):
        for file in files:
            if file.lower().endswith(".mts"):
                input_path = os.path.join(root, file)
                output_filename = os.path.splitext(file)[0] + ".mp4"
                output_path = os.path.join(save_folder, output_filename)

                try:
                    original_clip = VideoFileClip(input_path)
                    original_clip.write_videofile(output_path, codec="libx265", verbose=True)
                    original_clip.close()
                    print(f"Converted {input_path} to {output_path}")
                except Exception as e:
                    print(f"Error converting {input_path}: {e}")


