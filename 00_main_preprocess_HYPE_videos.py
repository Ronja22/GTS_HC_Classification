from constants import PreprocessHYPE as const
from src.utils.prints import print_variables_of_class
from src.preprocessing import cut_videos
from src.utils import videos

def main():
    """
    Main function to preprocess HYPE videos by cutting, saving frames, and downsampling.
    """
    # Print information about the preprocessing parameters
    print("Preprocessing HYPE videos with the following parameters:")
    print_variables_of_class(const)
    print()

    # Cut HYPE videos to extract segments where only the upper body is visible and the experimenter is out of the room
    cut_videos.cut_videos_from_folders(
       const.INPUT_VIDEO_FOLDERS,
       const.CUT_VIDEO_FOLDER,
       const.SPECIFIC_SUBJECTS,
        const.OVERRIDE
    )

    # Save one frame from each cut video to ensure the correct segments were cut
    videos.save_frame_from_each_video(
        const.CUT_VIDEO_FOLDER,
        const.FRAME_FOLDER,
        print_frame=False
    )

    # Downsample the videos to the target frame rate (FPS)
    videos.downsample_video(
        const.CUT_VIDEO_FOLDER,
        const.DOWNSAMPLED_VIDEO_FOLDER,
        const.TARGET_FPS,
        const.SPECIFIC_SUBJECTS,
        const.OVERRIDE
    )

if __name__ == "__main__":
    main()