from constants import PreprocessHYPE as const
from src.utils.prints import print_variables_of_class
from src.preprocessing import cut_videos
from src.utils import videos


def main():
    # Print information about the parameters
    print("Preprocessing HYPE videos with the following parameters:")
    print_variables_of_class(const)
    print()

    # Cut HYPE-Videos since we will just use the part of the video with upper body only where the experimenter is out of the room
    cut_videos.cut_videos_from_folders(const.INPUT_VIDEO_FOLDERS,
               const.OUTPUT_CUT_VIDEO_FOLDER,
               const.SPECIFIC_FILES,
               const.OVERRIDE)
    
    # to make sure the right segments are cut, save one frame of each cutted video 
    videos.save_frame_from_each_video(const.INPUT_VIDEO_FOLDERS,
                                      const.OUTPUT_FRAME_FOLDER,
                                      print_frame = False)
    #asdfasdfdsdf

if __name__ == "__main__":
    main()