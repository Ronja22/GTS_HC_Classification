from constants import PreprocessRUSH as const
from src.utils.prints import print_variables_of_class
from src.preprocessing import cut_videos
from src.utils import videos
import src.face_detection_and_cropping as face_detect
import src.save_face_mesh_coordinates as save_mesh
import src.face_mesh_utils.prepare_face_mesh as prep_mesh
import src.preprocessing.transform_to_standardposition as standard_position
import src.preprocessing.center_and_normalize as center_norm
import src.preprocessing.medianfilter as filter
import src.preprocessing.calculate_features as feat

def main():
    """
    Main function to preprocess RUSH videos by cutting, saving frames, and downsampling.
    """
    # Print information about the preprocessing parameters
    print("Preprocessing RUSH videos with the following parameters:")
    print_variables_of_class(const)
    print()

    # Cut RUSH videos to extract segments where only the upper body is visible and the experimenter is out of the room
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

    # Face detection and cropping
    face_detect.detect_face_and_crop(
        const.DOWNSAMPLED_VIDEO_FOLDER,
        const.CROPPING_FOLDER,
        const.SPECIFIC_SUBJECTS
    )

    # save face mesh coordinates
    save_mesh.save_facemesh_coordinates(
        const.DOWNSAMPLED_VIDEO_FOLDER,
        const.CROPPING_FOLDER,
        const.COORDINATE_FOLDER,
        const.SPECIFIC_SUBJECTS
    )

    # save face mesh coordinates
    prep_mesh.prepare_face_mesh_rush_hype(
        const.COORDINATE_FOLDER,
        const.SPECIFIC_SUBJECTS,
        const.OVERRIDE
    )

    # transform mesh to standard position
    standard_position.transform_to_standardposition(
        const.COORDINATE_FOLDER,
        const.SPECIFIC_SUBJECTS,
        const.OVERRIDE                                                                                     
    )

    # center and normalize face-mesh data
    center_norm.center_and_normalize_data(
        const.COORDINATE_FOLDER,
        const.SPECIFIC_SUBJECTS,
        const.OVERRIDE
    )

    # apply median filter 
    filter.medianfilter(
        const.COORDINATE_FOLDER,
        const.SPECIFIC_SUBJECTS,
        override = const.OVERRIDE
    )

    # calculate features
    feat.calc_all_features(
        const.COORDINATE_FOLDER,
        const.FEATURE_FOLDER,
        const.SPECIFIC_SUBJECTS,
        const.OVERRIDE,
        const.TARGET_FPS
    )

if __name__ == "__main__":
    main()