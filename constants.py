class PreprocessHYPE:
    """
    Class to preprocess HYPE videos.
    """
    # List of input video folders
    INPUT_VIDEO_FOLDERS = ["../Data/Videos/HYPE_UZL"]

    # Path in which the cut videos should be saved
    OUTPUT_CUT_VIDEO_FOLDER = "../Data/Videos_cut/HYPE_UZL"

    # List of specific files to process, or None to process all
    SPECIFIC_FILES = None

    # If False, it will not override data that already exists
    OVERRIDE = False

    #follder in which one frame of each video should be saved for sanity checks
    OUTPUT_FRAME_FOLDER = "../Data/Video_frames/HYPE_UZL"

class PreprocessRUSH:
    """
    Class to preprocess RUSH videos.
    """


    # List of input video folders
    INPUT_VIDEO_FOLDERS = ["../Data/Videos/RUSH/P1", "../Data/Videos/RUSH/P2a", "../Data/Videos/RUSH/Q1", "../Data/Videos/RUSH/Q2", "../Data/Videos/RUSH/P3a"]

    # Path in which the cut videos should be saved
    OUTPUT_CUT_VIDEO_FOLDER = "../Data/Videos_cut/RUSH"

    # List of specific files to process, or None to process all
    SPECIFIC_FILES = None

    # If False, it will not override data that already exists
    OVERRIDE = False

    #follder in which one frame of each video should be saved for sanity checks
    OUTPUT_FRAME_FOLDER = "../Data/Video_frames/RUSH"