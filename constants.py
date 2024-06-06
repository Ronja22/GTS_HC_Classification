class PreprocessHYPE:
    """
    Class to preprocess HYPE videos.
    """
    # List of input video folders
    INPUT_VIDEO_FOLDERS = ["../Data/Videos/HYPE_UZL"]

    # Path in which the cut videos should be saved
    CUT_VIDEO_FOLDER = "../Data/Videos_cut/HYPE_UZL"

    # List of specific files to process, or None to process all
    SPECIFIC_SUBJECTS = None

    # If False, it will not override data that already exists
    OVERRIDE = False

    # Folder in which one frame of each video should be saved for sanity checks
    FRAME_FOLDER = "../Data/Video_frames/HYPE_UZL"

    # Target frames per second for the downsampled videos
    TARGET_FPS = 25

    # Path in which the downsampled videos should be saved
    DOWNSAMPLED_VIDEO_FOLDER = "../Data/Videos_downsampled/HYPE"


class PreprocessRUSH:
    """
    Class to preprocess RUSH videos.
    """
    # List of input video folders
    INPUT_VIDEO_FOLDERS = ["../Data/Videos/RUSH/P1", "../Data/Videos/RUSH/P2a", "../Data/Videos/RUSH/Q1", "../Data/Videos/RUSH/Q2", "../Data/Videos/RUSH/P3a"]

    # Path in which the cut videos should be saved
    CUT_VIDEO_FOLDER = "../Data/Videos_cut/RUSH"

    # List of specific files to process, or None to process all
    SPECIFIC_SUBJECTS = None

    # If False, it will not override data that already exists
    OVERRIDE = False

    # Folder in which one frame of each video should be saved for sanity checks
    FRAME_FOLDER = "../Data/Video_frames/RUSH"

    # Target frames per second for the downsampled videos
    TARGET_FPS = 25

    # Path in which the downsampled videos should be saved
    DOWNSAMPLED_VIDEO_FOLDER = "../Data/Videos_downsampled/RUSH"