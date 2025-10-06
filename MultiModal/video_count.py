import os
import pandas as pd
def count_videos(parent_folder, ):
    extensions=(".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv" )
    # extensions=(".wav")
    count = 0
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.lower().endswith(extensions):
                count += 1
    return count
import os

def get_unprocessed_videos(videos_dir, preprocessed_dir, video_ext=(".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv" ),csv=False):
    """
    Returns a list of video files that are in `videos_dir` but not yet in `preprocessed_dir`.
    
    Args:
        videos_dir (str): Path to folder containing raw videos.
        preprocessed_dir (str): Path to folder containing preprocessed videos.
        ext (tuple): Allowed video extensions.
    
    Returns:
        list: Paths of videos that need to be preprocessed.
    """
    unprocessed = []
    


    for root, _, files in os.walk(videos_dir):
        for f in files:
            if not f.lower().endswith(video_ext):
                continue

            rel_path = os.path.relpath(os.path.join(root, f), videos_dir)  # relative path inside "videos/"
            rel_base, _ = os.path.splitext(rel_path)

            # Check if corresponding preprocessed file exists (any of allowed processed_ext)
            found = False
            for ext in video_ext:
                if os.path.exists(os.path.join(preprocessed_dir, rel_base + ext)):
                    found = True
                    break

            if not found:
                input_path = os.path.join(root, f)
                save_path = input_path.replace(videos_dir, preprocessed_dir)
                unprocessed.append((input_path, save_path))

    return unprocessed




if __name__=="__main__":
    dataset_root   = "/home/manik/Downloads/FakeAVCeleb_v1.2/"
    processed_root = "/home/manik/Downloads/FakeAvCelebPreprocessed/"


    print(f"number of videos in original dataset: {count_videos(dataset_root)}")
    print(f"number of videos in processed dataset: {count_videos(processed_root)}")



    

    unprocessed_list = get_unprocessed_videos(dataset_root, processed_root)

    print(f"Videos left to preprocess: {len(unprocessed_list)}")
  
    # print(len(unprocessed_list)+count_videos(processed_root))
    print(unprocessed_list[:][1])  # print first 5 unprocessed videos
    





