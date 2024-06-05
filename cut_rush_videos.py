import numpy as np
import helper_functions as hf
import pandas as pd
import os
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


def set_start_and_endpoint(filename):
    
  
    if "P2a" in filename:
        startpoint = 450
        endpoint = 600
    elif "P3a" in filename:
        startpoint = 150
        endpoint = 300
    elif "P3b" in filename:
        startpoint = 450
        endpoint = 600
    else:
        startpoint = 450
        endpoint = 600  # or any default values you want to set

    # Check for exceptions
    if "GTS_P2a_007_T2" in filename:
        startpoint = 150
        endpoint = 300
    elif "HC_P3a_001" in filename:
        startpoint = 0
        endpoint = 150
    elif "HC_P3b_002_RUSH2_T2" in filename:
        startpoint = 150
        endpoint = 300
    elif "HC_P3b_002_RUSH1_T2" in filename:
        startpoint = 150
        endpoint = 300


    return startpoint, endpoint

def cut_videos(videopath,savepath,start_time, end_time):
    # This function loads the video from videopath, cuts it from starttime to endtime (in seconds) and saves it to savepath
    
    
    print("saving under", savepath + "/" + os.path.basename(videopath).split(".")[0] + ".mp4")
    #ffmpeg_extract_subclip(videopath , starttime,endtime,
    #targetname=savepath + "/" + os.path.basename(videopath).split(".")[0] + ".mp4")

    targetname = os.path.join(savepath, os.path.basename(videopath).split(".")[0] + ".mp4")

    # Load the original video clip
    original_clip = VideoFileClip(videopath)

    # Create the subclip with exact 150 seconds duration
    subclip = original_clip.subclip(start_time, end_time)

    # Save the subclip
    subclip.write_videofile(targetname, codec="libx264",verbose=False)
    
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

def cut_all_rush_videos(video_folder, save_path, certain_subjects, override = False):
    # This function checks from where to where the RUSH videos have to be cut
    # it also calls the cut function and cuts the videos. if override == false it skips videos wwhich are already cut under the save folder.
    # it also saves one frame for each file to check that they are cut correctly.
    hf.create_path_if_not_existent(save_path)
    
    # for each folder of videos
    for folder in video_folder:
        for filename in os.listdir(folder):
            print("video from", folder + os.path.basename(filename).split(".")[0]  + ".mp4")
            print(certain_subjects)
            if certain_subjects == []:
                certain_subjects = ["_"]
            if any(string in filename for string in certain_subjects):
                print("Filename:", filename)
                if override == False:
                    if os.path.exists(save_path + os.path.basename(filename).split(".")[0]  + ".mp4"):
                        print("file " + filename  + " exists, skipping")
                        continue

                #get start and endpoints
                startpoint, endpoint = set_start_and_endpoint(filename)
                print("cutting from to",startpoint, endpoint)


                #call cut rush function
                cut_videos(folder + filename, save_path, startpoint, endpoint)

                #call saveframe and plot frame also
                save_frame_from_each_video(save_path + filename.split(".")[0] + ".mp4")
