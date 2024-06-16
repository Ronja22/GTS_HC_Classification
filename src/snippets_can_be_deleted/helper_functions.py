import os
import pandas as pd
import numpy as np


def create_path_if_not_existent(path):
    # this function creates a directory if it does not exist
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
        
        
def get_axis(filepath):
    """
    Extract the axis from the filepath

    This function takes a filename as input and extracts the axis letter ('X', 'Y', or 'Z')
    from the basename of the file. The axis letter is assumed to be a single capital letter,
    and the function returns the axis as a string.

    Parameters:
    filename (str): The input filename from which to extract the axis information.

    Returns:
    axis (str): The axis letter ('X', 'Y', or 'Z') extracted from the filepath
                If no axis letter is found, returns an empty string ('').

    Example:
    filename = "data_XYZ.npy"
    axis = get_axis(filename)
    # axis will be 'X'
    """

    # Extract the basename of the file without the directory path and extension
    
    name_without_extension = os.path.splitext(filepath)[0]

    # Initialize the axis to an empty string
    axis = ''

    # Check if 'X', 'Y', or 'Z' is present in the basename (ignoring case)
    if '_X' in name_without_extension.upper():
        axis = 'X'
    elif '_Y' in name_without_extension.upper():
        axis = 'Y'
    elif '_Z' in name_without_extension.upper():
        axis = 'Z'

    # Return the extracted axis letter
    return axis



def print_all_shapes_from_folder(folder_path,certain_subjects):
    # Liste zum Speichern der DataFrame-Formen 
    shapes = [] 
    # Durchsuche den Ordner nach Dateien 
    for filename in os.listdir(folder_path): 
        if filename.endswith('.csv'): 
            if certain_subjects == []:
                certain_subjects = ["_"]
            if any(string in filename for string in certain_subjects):

                # Prüfe auf das gewünschte Dateiformat, z.B. '.csv' 
                file_path = os.path.join(folder_path, filename) 
                # Lade den DataFrame 
                df = pd.read_csv(file_path) 
                # Speichere die Form des DataFrames in der Liste 
                #shapes.append((filename, df.shape)) 
                print(filename, df.shape)
                
                
def get_rush_excludes():
    
    # excluded because in other studies it was being questioned if they are really HC
    exclude = ["HC_P2a_007", "HC_P3a_003"]
   
    
    return exclude

def get_urge_excludes():
    # excluded because subject is chewing gum during video
    return ["GTS_P3a_023", "P2a_015_UF_T1", "P2a_015_US_T1"]


def get_hype_excludes():
    
    return ["UKD_HYPE_K_HC_012"]


def relabel_HCs(df):
    # this functions should relabel all HC videos which I checked with alexander and he found to be GTS subjects.
    # to make sure I dont get the same name (for example HC_001 turns out to be GTS, but we already have a GTS_001 video) I am gonna rename them in the following way:
    # HC_P3a_001 will become GTS_P3a_001a
    #folgende Änderungen sollen gemacht werden:
    # HC_P3a_011 -> GTS_P3a_011a
    # HC_P3b_025 -> GTS_P3b_025a
    # HC_P3b_028 -> GTS_P3b_028a
    
    # HC_P3b_003 must be excluded because he was not following the instructions
    
    #print(df["ID"])
    # Define the replacement mapping
    replacement_mapping = {
        'HC_P3a_011': 'GTS_P3a_011a',
        'HC_P3b_025': 'GTS_P3b_025a',
        'HC_P3b_028': 'GTS_P3b_028a'
    }

    # Apply the replacements
    df['ID'] = df['ID'].replace(replacement_mapping, regex = True)
    
    df = df[~df["ID"].str.contains("HC_P3a_003")]
    
    df["class"] = np.zeros(len(df))
    df["class"][df["ID"].str.contains("GTS")] = 1
    return(df)


