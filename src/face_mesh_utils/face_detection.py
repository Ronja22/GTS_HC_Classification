import cv2 
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union
import math
import pandas as pd
import csv

from ..utils import videos as vid



def get_face_bounding_box(video_file, certainty=0.8, debug = False):
    """
    Retrieves the bounding box coordinates of the detected face in a video.

    Parameters:
        video_file (str): The path to the video file.
        certainty (float, optional): The minimum confidence score required for accepting the face detection result.
            Faces with scores lower than this value will be ignored. Default is 0.9.

    Returns:
        list: A list representing the bounding box coordinates of the detected face in the format [start_y, end_y, start_x, end_x].
            start_y (int): The Y-coordinate of the top-left corner of the bounding box.
            end_y (int): The Y-coordinate of the bottom-right corner of the bounding box.
            start_x (int): The X-coordinate of the top-left corner of the bounding box.
            end_x (int): The X-coordinate of the bottom-right corner of the bounding box.

    Note:
        This function uses a face detection algorithm to find faces in the video. It relies on the OpenCV library
        for video processing and the face detection model used should be accurate enough to meet the desired certainty.
        The function returns None if no face is detected with a confidence score greater than or equal to the specified certainty.
        """
    

    # Initialize the face detection model
    base_options = python.BaseOptions(model_asset_path= os.path.join('src/face_mesh_utils/mediapipe_models','blaze_face_short_range.tflite'))
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.3)
    detector = vision.FaceDetector.create_from_options(options)

    # Open the video file and get the total number of frames
    cap = cv2.VideoCapture(video_file)
    print(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   

    # Check if frames are found
    if total_frames == 0:
        raise ValueError("No frames found in the video: " + video_file)

    # Initialize variables for cropping in case no face can be detected
    crop_iteration = 0
    found_valid_boxes = False
    detection_results = []
    origins_x = []
    origins_y = [] 
    widths = []
    heights =  []
    best_scores = [] 
    best_frames = []
    middle_points = []
    
    # Loop until the best score is greater than or equal to the defined certainty
    while found_valid_boxes == False:
        print(crop_iteration)
        crop_iteration += 1

        # Adjust crop_y and crop_x based on the iteration, if no face is detected in one iteration we start again with a smaller frame
        if crop_iteration == 1:
            crop_y = 0
            crop_x = 0
        else:
            crop_y = 50 + (crop_iteration - 1) * 5
            crop_x += 100 + (crop_iteration - 1) * 10

        # Print crop_y and crop_x for debugging purposes
        if debug:   
            print("face_detection.get_face_bounding_box")
            print("crop y", crop_y)
            print("crop x", crop_x)

        # Create a crop region based on the adjusted crop_y and crop_x
        crop = [0 + int(0.2 * crop_y), -1 - crop_y, 0 + crop_x, -1 - crop_x]
        print("cropping with following parameters [y_start, y_end, x_start, x_end]:", crop)
        
        # Process video frames in steps of (crop_iteration*100 + (crop_iteration-1)*20)
        # check every 25th frame ergo every first frame of every second
        for i in range(0+crop_iteration-1,total_frames,25):
            #print("i", i)
            try:
                # Get a frame from the video
                img = vid.get_frame_from_video(video_file, i)
                #print(i)
            except:
                # If an exception occurs, skip to the next frame
                continue
           


            # Crop the frame based on the defined crop region
            img = img[crop[0]:crop[1], crop[2]:crop[3], :]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
           
            # Detect faces in the cropped frame
            detection_result = detector.detect(img)

            # Iterate through the detected faces and update the best_box, bestscore, and bestframe if needed
            for detection in detection_result.detections:
                #print("detection")
                bounding_box = detection.bounding_box
                cat = detection.categories[0]

               
                if cat.score >= certainty:
                    found_valid_boxes = True
                    detection_results.append(detection_result)
                    origins_y.append(bounding_box.origin_y)
                    origins_x.append(bounding_box.origin_x)
                    widths.append(bounding_box.width)
                    heights.append(bounding_box.height)
                    best_scores.append(cat.score)
                    best_frames.append(i)
                    middle_points.append([bounding_box.origin_x + bounding_box.width/2, bounding_box.origin_y + bounding_box.height/2])

    print(f"{len(best_scores)} frames found with certainty > {certainty}")
    
    # get the median of bounding boxes above certainty
    final_height = int(np.median(heights))
    final_width = int(np.median(widths))
    final_origin_x =int(np.median(origins_x))
    final_origin_y = int(np.median(origins_y))

    # Get a frame from the video using the first frame where the face was detected with a certainty above the requested certainty
    img = vid.get_frame_from_video(video_file, best_frames[0])

    # Crop the frame based on the defined crop region, for visualization this has to be done because bounding box is based on this cropped frame
    img = img[crop[0]:crop[1], crop[2]:crop[3], :]

    # Print the bestscore, bestframe, and best_box for debugging purposes
    data = [
        {
            "best bounding box": "",
            "origin_x": final_origin_x,
            "origin_y": final_origin_y,
            "height": final_height,
            "width": final_width,
        }
    ]
    if debug: 
        print("face_detection.get_face_bounding_box")
        print(data)

    # Visualize the image with bounding box annotations
    annotated_image = visualize_bounding_box(img, final_origin_x, final_origin_y, final_height, final_width)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    plt.imshow(cv2.cvtColor(rgb_annotated_image, cv2.COLOR_RGB2BGR))
    plt.axis('off')
    plt.show(block=False)

    # Display the image for 3 seconds
    plt.pause(3)
    plt.close('all')  # Close all open figures
    root, extension = os.path.splitext(video_file)

    # Write the data to a CSV file
    with open(root + "_bounding_box.csv", mode="w", newline="") as file:
        fieldnames = ["best bounding box", "origin_x", "origin_y", "height", "width"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data
        for row in data:
            writer.writerow(row)
   
    # Calculate the final_crop based on the best_box and the applied crop region
    final_crop = [final_origin_y + crop[0], final_origin_y+ crop[0] + final_height, 
                  final_origin_x + crop[2], final_origin_x + crop[2] + final_width]
    cap.release()
    # Return the final_crop
    return final_crop


def visualize_bounding_box(
    image,
    origin_x, origin_y,height1,width1
    ) -> np.ndarray:
    MARGIN = 10  # pixels
    ROW_SIZE = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 0, 0)  # red
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
    Returns:
    Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape


    start_point = origin_x, origin_y
    end_point = origin_x + width1, origin_y + height1
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

  

    return annotated_image


def save_cropping_parameters(bounding_box, path_to_video_file, output_folder,video_width=1920, video_height=1080):
    """
    Saves the cropping parameters based on a face bounding box.

    Parameters:
        bounding_box (list): A list representing the bounding box coordinates of the detected face.
            It should be in the format [start_y, end_y, start_x, end_x].
        path_to_video_file (str): The path to the video file.
        video_width (int, optional): The width of the video in pixels. Default is 1920.
        video_height (int, optional): The height of the video in pixels. Default is 1080.

    Note:
        This function calculates the cropping parameters based on the provided bounding box and saves
        them to a CSV file with the same name as the video file but with "_cropping.csv" appended.
    """
    # Calculate height and width of bounding box
    height = bounding_box[1] - bounding_box[0]
    width = bounding_box[3] - bounding_box[2]

    # Calculate cropping parameters
    y_start = int(max(0, bounding_box[0] - 0.5 * height))
    y_end = int(min(bounding_box[1] + height, video_height))
    x_start = int(max(0, bounding_box[2] - 1.5 * width))
    x_end = int(min(bounding_box[3] + 1.5 * width, video_width))
    crop = [y_start, y_end, x_start, x_end]

    # Create DataFrame
    df = pd.DataFrame({'origin_y': [y_start],
                       'origin_x': [x_start],
                       'end_x': [x_end],
                       'end_y': [y_end],
                       'width': [x_end - x_start],
                       'height': [y_end - y_start]})

    # Get the root path of the video file
    root, _ = os.path.splitext(path_to_video_file)
    # Get the base name (filename with path removed)
    filename = os.path.basename(root)

    # Save the DataFrame to a CSV file
    print(f"saving under {os.path.join(output_folder, filename)}_cropping.csv")
    df.to_csv(os.path.join(output_folder, filename + "_cropping.csv"))


def face_detection_and_cropping(input_video_path, output_folder):
    """
    Detects the face bounding box in a video and saves cropping parameters based on the detected bounding box.

    Parameters:
        input_video_path (str): The path to the input video file.

    Note:
        This function detects the face bounding box using the face detection algorithm from the 'face_detect' module,
        retrieves the dimensions of the video using the 'vid' module, and then saves the cropping parameters based on
        the detected bounding box to a CSV file using the 'face_detect' module.
    """
    # Detect face bounding box
    bounding_box = get_face_bounding_box(input_video_path)
    
    # Get video dimensions
    video_width, video_height = vid.get_video_dimensions(input_video_path)
    
    # Save cropping parameters
    save_cropping_parameters(bounding_box, input_video_path,output_folder, video_width, video_height)