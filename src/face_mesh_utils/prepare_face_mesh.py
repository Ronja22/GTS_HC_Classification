from multiprocessing.spawn import prepare
import pandas as pd
import numpy as np
import os
import math


import src.utils.directories as dir

def prepare_face_mesh_urge(filepath):
    
    filename = os.path.basename(filepath).split(".")[0]
    print("filename", filename)
    folder_name = os.path.dirname(os.path.dirname(filepath)) + "/"
    print("folder name", folder_name)
    # get path of transformation matrix and blendshapes
    transformation_path = folder_name + "Transformation_matrix/" + filename + ".csv"
    blendshape_path = folder_name + "Blendshape/" + filename + ".csv"
    
    print("filepath", filepath)
    axis = hf.get_axis(filepath)
    print("current axis",axis)
    
    # load facemesh
    mesh=np.load(filepath)
    
    #get number of features and number of frames
    num_feat=mesh.shape[1]
    num_frames=mesh.shape[0]
    print("number of features", num_feat)
    print("number of frames", num_frames)
    
    # put mesh data into a dataframe
    mesh=pd.DataFrame(mesh,columns=range(0,num_feat))
    print(mesh)
    
    # create column for number of frames
    mesh["frame"]=range(0,num_frames)
    
    # for p3a, ratings are saved under different names, instead of "UZL_E_GTS_P3a_001_UF_T1" it is "GTS_P3a_001_UF"
    savename=filename
    if "P3a" in filename:
        filename = filename.replace("UZL_E_","")
        filename = filename.replace("_T1","")
        #print("yes")
        print(filename)
    
    #load rating
    rating = pd.read_csv("/home/ronjaschappert/Documents/UrgeTicAnalysis_2021/DeepTic/Tic_Rating_analysis/RUSH/final/data/URGE/Ratings/" + filename+".txt",delimiter = ';')
    
    print("length rating", len(rating))
    # calculate first and last frame number:
    start_msec = rating.onset.iloc[0]
    end_msec = rating.onset.iloc[-1]

    # 30 fps -> 0.03 frames per millisecond
    start_frame = round(start_msec * 0.025)
    end_frame = round(end_msec * 0.025) + 24  # First frame of last second + 24 more frames


    print("Length:", end_frame - start_frame + 1)

    # transform frame to numeric
    mesh["frame"] = pd.to_numeric(mesh["frame"])
    
    # cut also transformation matrix and blendshapes
    if axis == "X":
        transformation = pd.read_csv(transformation_path)
        transformation["frame"] = mesh["frame"].copy()
        transformation_cut = transformation.loc[transformation.frame.between(start_frame,end_frame)]
        
        blendshape = pd.read_csv(blendshape_path)
        blendshape["frame"] = mesh["frame"].copy()
        blendshape_cut = blendshape.loc[blendshape.frame.between(start_frame,end_frame)]
        
    #cut facemesh
    mesh_cut = mesh.loc[mesh.frame.between(start_frame,end_frame)]
    

    print("len mesh_cut",len(mesh_cut))
    savepath = folder_name  + "Coordinates_" + axis  + "_prepared/"
    
    dir.create_directory_if_not_exists(savepath)
    
    if axis == "X":
        transformationsavepath = folder_name + "transformation_cut"
        dir.create_directory_if_not_exists(transformationsavepath)
        transformation_cut.to_csv(transformationsavepath +"/"+ savename + ".csv")
        
        blendshapesavepath = folder_name + "blendshape_cut"
        dir.create_directory_if_not_exists(blendshapesavepath)
        blendshape_cut.to_csv(blendshapesavepath +"/"+ savename + ".csv")
        
    #save face mesh prepared
    mesh_cut.to_csv(savepath + savename+".csv",index=False)
    
    

def prepare_face_mesh_all_urge(folder_name, certain_subjects, override = False):
    # this function takes the face mesh coordinates from folder_name/Coordinates_X (Y,Z) and saves them under folder_name/Coordinates_X_prepared
    
    for axis in ["X","Y","Z"]:
        print("folder_name", folder_name)
        for filename in os.listdir(folder_name + "Coordinates_" + axis):
            # if there are no certain subjects for which the face mesh should be applied, certain_subjects will be set to "_" so all subjects will be used
            if certain_subjects == []:
                certain_subjects = ["_"]
            if any(string in filename for string in certain_subjects):


                filepath = folder_name + "Coordinates_" + axis + "/" + filename
                savepath = folder_name + "Coordinates_" + axis + "_prepared/"
                save = savepath + filename[:-4] + ".csv"
                print("savename", save)
                # if override is false, check if coordinates already exist and skip this video
                if override == False:
                    if os.path.exists(save):
                        print("file " + filename  + " exists, skipping")
                        continue

                
                hf.create_path_if_not_existent(savepath)
                
                
                #prepare urge coordinates for single file
                prepare_face_mesh_urge(filepath) 
                
                
                
                
                

def prepare_face_mesh_rush_hype(input_folder, certain_subjects=[], override=False):
    """
    Prepare face mesh data by converting and saving it in a structured format.

    Args:
        input_folder (str): The path to the main folder containing raw face mesh data.
        certain_subjects (list, optional): List of specific subjects to process. Default is an empty list.
        override (bool, optional): If False, skip processing for existing files. Default is False.

    Returns:
        None
    """
    
    # Iterate over the X, Y, Z coordinate directories
    for ax in ["X", "Y", "Z"]:
        startfolder = os.path.join(input_folder, "Coordinates_" + ax)
        savefolder = os.path.join(input_folder, "Coordinates_" + ax + "_prepared")
        
        # Create the output directory if it does not exist
        dir.create_directory_if_not_exists(savefolder)

        # Iterate over all files in the start folder
        for filename in os.listdir(startfolder):
            # If certain_subjects is empty, default to processing all files
            if not certain_subjects:
                certain_subjects = ["_"]
            
            # Process files that match any of the certain_subjects strings
            if any(subject in filename for subject in certain_subjects):
                print("Processing filename:", filename)

                
                output_file = os.path.join(savefolder, filename.split(".")[0] + ".csv")
                
                # If override is False, skip files that already exist in the save folder
                if not override and os.path.exists(output_file):
                    print("File " + filename + " exists, skipping")
                    continue

                # Load the raw face mesh data from the .npy file
                mesh = np.load(os.path.join(startfolder, filename))

                # Get the dimensions of the mesh data (num_frames x num_features)
                num_frames, num_features = mesh.shape

                # Convert mesh data to a DataFrame with frame numbers
                mesh_df = pd.DataFrame(mesh, columns=range(num_features))
                mesh_df["frame"] = range(num_frames)
                mesh_df["frame"] = pd.to_numeric(mesh_df["frame"])  # Ensure frame column is numeric

                # Save the prepared face mesh data to a CSV file in the save folder
                mesh_df.to_csv(output_file, index=False)
                print(f"Saved prepared data to {output_file}")


