import cv2 
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union
import math
import helper_functions as hf
import pandas as pd

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


def visualize_bounding_box1(
    image,
    detection_result
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

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                         width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image

def get_face_bounding_box(filepath, certainty=0.9):
    """
    Retrieves the bounding box coordinates of the detected face in a video.

    Parameters:
        filepath (str): The path to the video file.
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
    import csv
    # Initialize the face detection model
    base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.3)
    detector = vision.FaceDetector.create_from_options(options)

    # Open the video file and get the total number of frames
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize variables to store the best face detection result
    best_score = 0
    best_frame = -1
    best_box = []
    
    # Initialize variables for cropping in case no face can be detected
    j = 0
    found_valid_boxes = False
    results = []
    origins_x = []
    origins_y = [] 
    widths = []
    heights =  []
    best_scores = [] 
    best_frames = []
    middlepoints = []
    
    # Loop until the best score is greater than or equal to the defined certainty
    while found_valid_boxes == False:
        j += 1

        # Adjust crop_y and crop_x based on the iteration, if no face is detected in one iteration we start again with a smaller frame
        if j == 1:
            crop_y = 0
            crop_x = 0
        else:
            crop_y = 50 + (j - 1) * 5
            crop_x += 100 + (j - 1) * 10

        # Print crop_y and crop_x for debugging purposes
        print("crop y", crop_y)
        print("crop x", crop_x)

        # Create a crop region based on the adjusted crop_y and crop_x
        crop = [0 + int(0.2 * crop_y), -1 - crop_y, 0 + crop_x, -1 - crop_x]
        print("cropping with following parameters [y_start, y_end, x_start, x_end]:", crop)

        # Process video frames in steps of (j*100 + (j-1)*20)
        # check every 25th frame ergo every first frame of every second
        for i in range(0+j-1,total_frames,25):
            try:
                # Get a frame from the video
                img = get_frame_from_video(filepath, i)
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
                bbox = detection.bounding_box
                cat = detection.categories[0]

                #print(cat.score)

                if cat.score >= certainty:
                    found_valid_boxes = True
                    results.append(detection_result)
                    origins_y.append(bbox.origin_y)
                    origins_x.append(bbox.origin_x)
                    widths.append(bbox.width)
                    heights.append(bbox.height)
                    best_scores.append(cat.score)
                    best_frames.append(i)
                    middlepoints.append([bbox.origin_x + bbox.width/2, bbox.origin_y + bbox.height/2])
                        
    print(len(best_scores), " many frames with certainty >0.9 found")
    
    ### GET THE MEDIAN BOX OF BEST BOXES, save  also the values above 0.9 and also especially the with and hight of the median box (check if its at least 256       
    final_height = int(np.median(heights))
    final_width = int(np.median(widths))
    final_origin_x =int(np.median(origins_x))
    final_origin_y = int(np.median(origins_y))
    
    
    
    # Get the frame from the video using the bestframe index
    img = get_frame_from_video(filepath, 10)

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
    print(data)

    # Visualize the image with bounding box annotations

    annotated_image = visualize_bounding_box(img, final_origin_x, final_origin_y, final_height, final_width)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    plt.imshow(cv2.cvtColor(rgb_annotated_image, cv2.COLOR_RGB2BGR))
    plt.axis('off')
    plt.show()

    # Write the data to a CSV file
    with open(filepath.split(".")[0] + "bounding_box.csv", mode="w", newline="") as file:
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


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    
    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw_landmarks_on_image(rgb_image, face_landmarker_result):
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe import solutions
    face_landmarks_list = face_landmarker_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp.solutions.drawing_styles
              .get_default_face_mesh_iris_connections_style())
    return annotated_image



def save_face_mesh_coordinates(video_folder, save_folder,certain_subjects = [], override = False,certainty = 0.9):
    """
    Save face mesh coordinates for frames of videos.

    This function first calls 'get_face_bounding_box' to obtain the cropping coordinates
    for each frame in the videos. It then uses 'save_face_mesh_for_videos' to calculate
    face meshes for all frames based on the obtained bounding box coordinates.

    Parameters:
        video_folder (list of str): The paths to the folders containing video files.
        save_folder (str): The directory where the face mesh coordinates will be saved.
        certain_subjects (list, optional): A list of subject identifiers. If provided,
            the function will only calculate face mesh coordinates for the specified
            subjects. Default is an empty list, meaning it processes all subjects.
        override (bool, optional): If False, the function will skip calculating face
            mesh coordinates for a video if the coordinates for that video already exist
            in the 'save_folder'. If True, it will override any existing coordinates.
            Default is False.

    Notes:
        - The function assumes that the video files are stored in 'video_folder' and have
          filenames that uniquely identify each subject (e.g., subject_1.mp4,
          subject_2.mp4, etc.).
        - The function saves face mesh coordinates for each video in separate .npy files
          (e.g., subject_1_coordinates..npy, subject_2_coordinates.npy, etc.), using the
          MediaPipe's FaceMesh model to detect facial landmarks.
        - If 'certain_subjects' is not provided (empty list), the function will process
          all subjects in 'video_folder'. Otherwise, it will only calculate face mesh
          coordinates for the specified subjects.
        - The function uses 'override' to control whether to overwrite existing
          coordinate files for each video or skip processing videos with existing
          coordinate files in the 'save_folder'.

    Example:
        video_folder = ["path/to/videos1", "paht/to/videos2"]
        save_folder = "path/to/save_coordinates"
        certain_subjects = ["subject_1", "subject_3"]
        save_face_mesh_coordinates(video_folder, save_folder,
                                   certain_subjects=certain_subjects, override=True)
        """

 
    # for each folder of videos
    for folder in video_folder:
        for filename in os.listdir(folder):
            
        
            # if there are no certain subjects for which the face mesh should be applied, certain_subjects will be set to "_" so all subjects will be used
            if certain_subjects == []:
                certain_subjects = [""]
            if any(string in filename for string in certain_subjects):
                print("filename",filename)
                
                # if override is false, check if coordinates already exist and skip this video
                if override == False:
                    if os.path.exists(save_folder + "Coordinates_X/" + filename[:-3] + "npy"):
                        print("file " + filename  + " exists, skipping")
                        continue
            
                filename = folder + filename
                # if file is a video
                if filename.endswith(".mp4") or filename.endswith(".avi"):
                    print("filename 3", filename)
                    bounding_box = get_face_bounding_box(filename,certainty)
                    
                    #calculate height and width of bounding box
                    height = bounding_box[1] - bounding_box[0]
                    width = bounding_box[3]-bounding_box[2]
                    
                    #calculate cropping parameters:
                    y_start = int(max(0, bounding_box[0] - 0.5 * height))
                    y_end = int(bounding_box[1] + height)
                    x_start = int(max(0, bounding_box[2] - 1.5 * width))
                    x_end = int(bounding_box[3] + 1.5 * width)
                    crop = [y_start, y_end, x_start, x_end]
                    
                    # DataFrame erstellen
                    df = pd.DataFrame({'origin_y': [y_start],
                                       'origin_x': [x_start],
                                       'width': [x_end - x_start],
                                       'height': [y_end - y_start]})
                    
                    df.to_csv(filename.split(".")[0] + "cropping.csv")
                    
                    save_face_mesh_for_videos(filename,save_folder,crop = crop)
             




def save_face_mesh_for_videos(videopath, savepath, crop):
    """
    Save face mesh for each frame of a video.

    Parameters:
        videopath (str): The path to the video file.
        savepath (str): The directory where the face mesh images will be saved.
        crop: [start_y, end_y, start_x, end_x] parameters to crop each frame
    """
    print(savepath)
    
    # Check if folder for Coordinates_X, Y, Z, Transformation matrix and Blendshapes exist and if not create them
    # Check whether the specified path exists or not
    path = savepath + "Coordinates_X/"
    hf.create_path_if_not_existent(path)
    
    path = savepath + "Coordinates_Y/"
    hf.create_path_if_not_existent(path)
    
    path = savepath + "Coordinates_Z/"
    hf.create_path_if_not_existent(path)
    
    path = savepath + "Transformation_matrix/"
    hf.create_path_if_not_existent(path)
    
    path = savepath + "Blendshape/"
    hf.create_path_if_not_existent(path)

    
    # capture video and get the total frame number
    cap = cv2.VideoCapture(videopath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("number of frames:", total_frames)
    
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) 
    print("framerate ", frame_rate)


    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a face landmarker instance with the video mode:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
        min_face_detection_confidence = 0.05,
                                           min_face_presence_confidence = 0.05,
                                           min_tracking_confidence = 0.05,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1,
        running_mode=VisionRunningMode.VIDEO)
    
    # arrays to store the coordinates
    X = np.empty((0, 478))
    Y = np.empty((0, 478))
    Z = np.empty((0, 478))
    T = []
    B = []
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        frame_number =0
        while True:
            #print("frame:", frame_number)
            if frame_number == total_frames:
                break
            success, img = cap.read()

            if success == False:
                print("break", frame_number)
                
            if success == True:
                img = img[crop[0]:crop[1], crop[2]:crop[3]]
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
                frame_timestamp_ms = int((frame_number / frame_rate) * 1000)
                face_landmarker_result = landmarker.detect_for_video(img, frame_timestamp_ms)
                #print(face_landmarker_result)
                
                xcoord=[]
                ycoord=[]
                zcoord=[]
                
                # if we cant find landmarks, at the beginning we use a row of 0s, if in the middle we copy the data from the frame before
                if face_landmarker_result.face_landmarks ==[]:
                    print("not detected" , frame_number)
                    if frame_number == 0:
                        X=np.zeros((1,478))#468
                        Y=np.zeros((1,478))
                        Z=np.zeros((1,478))
                    else:
                        X=np.concatenate([X,[X[-1,:]]])
                        Y=np.concatenate([Y,[Y[-1,:]]])
                        Z=np.concatenate([Z,[Z[-1,:]]])
                    T.append([])
                    B.append([])
                    
                # get the face mesh coordinates
                else:
                    xcoord = [face_landmarker_result.face_landmarks[0][j].x for j in range(0,len(face_landmarker_result.face_landmarks[0]))]
                    ycoord = [face_landmarker_result.face_landmarks[0][j].y for j in range(0,len(face_landmarker_result.face_landmarks[0]))]
                    zcoord = [face_landmarker_result.face_landmarks[0][j].z for j in range(0,len(face_landmarker_result.face_landmarks[0]))]

                    #concenate them to the coordinates before
                    X=np.concatenate([X,[xcoord]])
                    Y=np.concatenate([Y,[ycoord]])
                    Z=np.concatenate([Z,[zcoord]])
                    #print(face_landmarker_result)
                    #print(face_landmarker_result.facial_transformation_matrixes)
                    T.append(face_landmarker_result.facial_transformation_matrixes)
                    #print(face_landmarker_result.face_blendshapes)
                    B.append(face_landmarker_result.face_blendshapes[0])
                
            
                if frame_number % 200 == 0:
                    print("timestamp_ms", frame_timestamp_ms)
                    print("frame:", frame_number)
                    annotated_image = draw_landmarks_on_image(imgRGB, face_landmarker_result)
                    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    plt.pause(0.001)  # Pause briefly to show the current frame
                    plt.show()
                
                
                frame_number += 1
                
    cap.release()
    print(videopath)   
    filename = os.path.splitext(videopath.split(os.sep)[-1])[0]
    print(filename)
    print(videopath)
    print(videopath.split(os.sep))

 
    print(X.shape[0]==frame_number)
    
    
    
    print(len(X))
    print(len(Y))
    print(len(Z))
    np.save(savepath + "Coordinates_X/"+filename+".npy",X)
    np.save(savepath + "Coordinates_Y/"+filename+".npy",Y)
    np.save(savepath + "Coordinates_Z/"+filename+".npy",Z)
    T = pd.DataFrame(T)
    T.to_csv(savepath + "Transformation_matrix/" + filename + ".csv", index=False)
    B = pd.DataFrame(B)
    B.to_csv(savepath + "Blendshape/" + filename + ".csv", index=False)
            
            