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