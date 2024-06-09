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

from ..utils import directories as dir

def create_face_landmarker_options():
    """
    Create and configure options for the FaceLandmarker.

    Returns:
        mp.tasks.vision.FaceLandmarkerOptions: Configured options for the FaceLandmarker.
    """
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    cwd = os.getcwd()
    # Create a face landmarker instance with the video mode:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path= os.path.join(cwd ,'src/face_mesh_utils/mediapipe_models','face_landmarker_v2_with_blendshapes.task')),
        min_face_detection_confidence = 0.05,
                                           min_face_presence_confidence = 0.05,
                                           min_tracking_confidence = 0.05,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1,
        running_mode=VisionRunningMode.VIDEO)

    return options

def save_coordinates(video_path, save_path, crop, debug = False):
    """
    Detects landmarks in each frame of a video, extracts coordinates, and saves them to files.

    Parameters:
        video_path (str): The path to the input video file.
        save_path (str): The path to the directory where the coordinates will be saved.
        crop (list): A list containing the cropping parameters [y_start, y_end, x_start, x_end].
        debug (bool, optional): Whether to display debug information. Default is False.
    """
    if debug:
        print("faace_mesh.save_coordinates")
        print("Output directory:", save_path)

    # Check if directories for coordinates exist, if not, create them
    subdirectories = ["Coordinates_X", "Coordinates_Y", "Coordinates_Z", "Transformation_matrix", "Blendshape"]
    for subdir in subdirectories:
        path = os.path.join(save_path, subdir)
        dir.create_directory_if_not_exists(path)
    

    num_coords = 468
    print(num_coords)
    
    # capture video and get the total frame number
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) 
    if debug:
        print("number of frames:", total_frames)
        print("framerate ", frame_rate)

    # Create a face landmarker instance with the video mode:
    options = create_face_landmarker_options()
    
    # Initialize arrays to store the coordinates
    X = np.empty((0, num_coords))
    Y = np.empty((0, num_coords))
    Z = np.empty((0, num_coords))

    # Initialize lists for transformation matrices and blendshapes
    T = []
    B = []



    FaceLandmarker = mp.tasks.vision.FaceLandmarker

    with FaceLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        frame_number = 0
        while frame_number < total_frames:
            success, img = cap.read()

            if not success:
                # If frame could not be read, skip to the next frame
                print("Frame could not be read:", frame_number)
                frame_number += 1
                continue

            img = img[crop[0]:crop[1], crop[2]:crop[3]]
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
            frame_timestamp_ms = int((frame_number / frame_rate) * 1000)
            face_landmarker_result = landmarker.detect_for_video(img, frame_timestamp_ms)
            
            xcoord=[]
            ycoord=[]
            zcoord=[]
            
            xcoord = []
            ycoord = []
            zcoord = []

            # If no landmarks are found
            if not face_landmarker_result.face_landmarks:
                print("Landmarks not detected in frame:", frame_number)
                if frame_number == 0:
                    # Initialize arrays with zeros for the first frame
                    X = np.zeros((1, num_coords))
                    Y = np.zeros((1, num_coords))
                    Z = np.zeros((1, num_coords))
                else:
                    # Concatenate the last row to maintain continuity
                    X = np.concatenate([X, [X[-1, :]]])
                    Y = np.concatenate([Y, [Y[-1, :]]])
                    Z = np.concatenate([Z, [Z[-1, :]]])
                # Append empty lists for transformation matrices and blendshapes
                T.append([])
                B.append([])
                
            # get the face mesh coordinates
            else:
                xcoord = [face_landmarker_result.face_landmarks[0][j].x for j in range(0,len(face_landmarker_result.face_landmarks[0]))]
                ycoord = [face_landmarker_result.face_landmarks[0][j].y for j in range(0,len(face_landmarker_result.face_landmarks[0]))]
                zcoord = [face_landmarker_result.face_landmarks[0][j].z for j in range(0,len(face_landmarker_result.face_landmarks[0]))]

           
                #concenate them to the coordinates before
                X=np.concatenate([X,[xcoord[0:468]]])
                Y=np.concatenate([Y,[ycoord[0:468]]])
                Z=np.concatenate([Z,[zcoord[0:468]]])
                
                T.append(face_landmarker_result.facial_transformation_matrixes)
                B.append(face_landmarker_result.face_blendshapes[0])


            if debug:
                if frame_number % 200 == 0:
                    print("timestamp_ms", frame_timestamp_ms)
                    print("frame:", frame_number)
                    annotated_image = draw_landmarks_on_image(imgRGB, face_landmarker_result)
                    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    plt.pause(0.001)  # Pause briefly to show the current frame
                    plt.show()
                
                    
            frame_number += 1
                
    cap.release()
    filename = os.path.splitext(video_path.split(os.sep)[-1])[0]

    #save x and y and z
    np.save(os.path.join(save_path, "Coordinates_X", filename+".npy"),X)
    np.save(os.path.join(save_path, "Coordinates_Y", filename+".npy"),Y)
    np.save(os.path.join(save_path, "Coordinates_Z", filename+".npy"),Z)
    T = pd.DataFrame(T)
    T.to_csv(os.path.join(save_path, "Transformation_matrix" , filename + ".csv"), index=False)
    B = pd.DataFrame(B)
    B.to_csv(os.path.join(save_path, "Blendshape" , filename + ".csv"), index=False)


    
def draw_specific_landmarks_on_image(rgb_image, face_landmarker_result, selected_landmarks):
    """
    Draws specific landmarks on the input RGB image.

    Parameters:
        rgb_image (numpy.ndarray): The input RGB image.
        face_landmarker_result (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
            The result of face landmark detection containing landmark points.
        selected_landmarks (list): A list of landmark indices to be drawn on the image.

    Returns:
        numpy.ndarray: Annotated image with specific landmarks drawn.

    Note:
        This function draws specific landmarks defined by their indices on the input image and overlays them
        with the face mesh landmarks and connections.

    Example:
        annotated_img = draw_specific_landmarks_on_image(rgb_img, face_landmarks, [0, 1, 2, 3, 4])
    """
    from mediapipe import solutions
    annotated_image = np.copy(rgb_image)

    # Draw only specific landmarks
    for idx in selected_landmarks:
        landmark_point = face_landmarker_result[idx]
        height, width, _ = annotated_image.shape
        cx, cy = int(landmark_point.x * width), int(landmark_point.y * height)
        cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)  # Example: Draw a green circle at the landmark position
    
    # Overlay face mesh landmarks and connections
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarker_result,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarker_result,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarker_result,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    
    return annotated_image



def draw_landmarks_on_image(rgb_image, face_landmarker_result):
    """
    Draws face landmarks on the input RGB image.

    Parameters:
        rgb_image (numpy.ndarray): The input RGB image.
        face_landmarker_result (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
            The result of face landmark detection containing landmark points.

    Returns:
        numpy.ndarray: Annotated image with face landmarks drawn.

    Note:
        This function draws face landmarks detected by the face landmark detector on the input image.
        It overlays the landmarks with the face mesh landmarks and connections.

    Example:
        annotated_img = draw_landmarks_on_image(rgb_img, face_landmarks)
    """
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

        # Overlay face mesh landmarks and connections
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    return annotated_image

