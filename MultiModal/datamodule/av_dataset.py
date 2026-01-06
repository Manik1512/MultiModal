"""
This file reads the csv file present in the dataset directory 
and read a single video from dataset and return all video frames and its corresponding audio 
"""
# import sys
# sys.path.insert(0, "../")

import os
import torch
import torchvision
import pandas as pd 
import numpy as np 
import random
# import sampler   #uncomment this when running independently
# import transforms  #uncomment this when running independently

import torchaudio
from . import transforms
from . import sampler
def dataset_name():
    pass

def load_video(path, processed_dir=None,debug=True,audio_path=None):
    audio,vid=None,None
    # audio_path = os.path.splitext(path)[0] + ".wav"
    try:
        vid, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")
        vid = vid.permute((0, 3, 1, 2))
    except Exception as e:
        print(f"[Warning] Could not load video: {path}. Error: {e}")
        vid = torch.zeros((25, 3, 96, 96))
    

    try:
        audio, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            audio = resampler(audio)
            # print(sample_rate)
            sample_rate = 16000
        
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)

    except Exception as e:
        print(f"[Warning] Could not load audio: {audio_path}. Error: {e}")
        audio = torch.zeros((1, 16000))

    if debug:
        print("audio ka size iin load video function",audio.shape)
        print("video ka size iin load video function",vid.shape)
    return vid,audio 




class CELEB_AV(torch.utils.data.Dataset):

    """rate ratio-how many audio samples correspond to one video frame
        For 16 kHz audio and 25 fps video: rate_ratio = 16000 / 25 = 640

        Expected audio length = 32 * 640 = 20,480 samples
    """
    def __init__(
        self,
        csv_file,
        rate_ratio=640,
        debug=True,
        num_frames=32,
        unprocessed_dir=None,
        preprocessed_dir=None,
        subset="train",
        modality_drop_rate=0.0,
        deepfake_type=None,
        dataset_type="directory_based",
        a_drop=False,
        v_drop=False
    
        
    ):

        df = pd.read_csv(csv_file)
        self.root_dir = unprocessed_dir if unprocessed_dir else None
        self.rate_ratio = rate_ratio
        self.num_frames = num_frames  
        self.debug = debug  
        self.processed_dir=preprocessed_dir if preprocessed_dir else None
        self.subset=subset
        self.modality_drop_rate=modality_drop_rate
        self.audio_transform=transforms.AudioTransform(self.subset,modality_drop_rate=self.modality_drop_rate,audio_drop=a_drop)
        self.video_transform=transforms.VideoTransform(self.subset,video_drop=v_drop)
        self.dataset_type=dataset_type

        if self.subset == "train" and dataset_type=="FakeAvCeleb":
            self.df= df[df["split_by_source"] == "train"].reset_index(drop=True)
        elif self.subset == "val" and dataset_type=="FakeAvCeleb":
            self.df= df[df["split_by_source"] == "val"].reset_index(drop=True)
        elif self.subset == "test" and deepfake_type is None and dataset_type=="FakeAvCeleb":
            self.df= df[df["split_by_source"] == "test"].reset_index(drop=True)

        elif self.subset == "test" and deepfake_type is not None and dataset_type=="FakeAvCeleb" :
            self.df = df[(df["split_by_source"] == "test") & (df['type'] == deepfake_type)].reset_index(drop=True)
            # self.df=self.df[self.df['method']=='faceswap'].reset_index(drop=True)  # to limit to only faceswap method for specific deepfake type

        elif dataset_type=="directory_based":
            self.df=df.reset_index(drop=True)
        else:   
            raise ValueError("subset must be one of 'train', 'val', or 'test'")

        self.dropout=transforms.ModalityDropout(p=self.modality_drop_rate)
        self.a_drop=a_drop
        self.v_drop=v_drop

    def sample_frames(self, video, audio):
        """
        Sample exactly `self.num_frames` consecutive frames 
        and their corresponding audio segment.
        """
        total_frames = video.shape[0]        
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.transpose(0, 1)  # [L, 1]

        if total_frames >= self.num_frames:
            if self.subset=="train":
                frame_start = np.random.randint(0, total_frames - self.num_frames + 1)  
            elif self.subset=="val" or self.subset=="test":
                frame_start=0

            frame_end = frame_start + self.num_frames
            video = video[frame_start:frame_end]
            # corresponding audio slice
            audio_start_expected = int(round(frame_start * self.rate_ratio))
            audio_end_expected = int(round(frame_end * self.rate_ratio))
            audio_end_real=int(audio.shape[0])

            if  audio_end_real< audio_end_expected :   #if audio slice is smaller than expected, pad it
                pad_len = audio_end_expected - audio_end_real 
                audio = torch.nn.functional.pad(audio, (0,0,0,pad_len))  # pad last dim
            audio = audio[audio_start_expected:audio_end_expected]
            

        elif total_frames < self.num_frames:
            audio_end_real = int(audio.shape[0])
            audio_end_expected = int(round(self.num_frames * self.rate_ratio))

            pad_count = self.num_frames - total_frames
            video_pad = video[-1:].repeat(pad_count, 1, 1, 1)
            video = torch.cat([video, video_pad], dim=0)
            if audio_end_real < audio_end_expected:
                pad_len = audio_end_expected - audio_end_real
                audio = torch.nn.functional.pad(audio, (0,0,0,pad_len))  # pad last dim
            
            audio = audio[:audio_end_expected]    
        
        return video, audio

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.dataset_type=="FakeAvCeleb":
            
            video_path = row["path"]
            type=row["type"]
            race=row["race"]
            gender=row["gender"]
            id=row["source"]
            method = str(row["method"]).lower()
            type=str(row["type"])
            if self.processed_dir:
                video_path = os.path.join(self.processed_dir,type,race,gender,id,video_path)
                audio_path = os.path.splitext(video_path)[0] + ".wav"
                base, ext = os.path.splitext(video_path)
                video_path = f"{base}_roi{ext}"
            
            # print(video_path)

        elif self.dataset_type=="directory_based":
            video_path=str(row["video_path"])
            audio_path=str(row["audio_path"])
            method = str(row["method"]).lower()
            type="other"


        if self.debug:
            print(video_path)

        # modality_to_drop = random.choice([0, 1])  #if 0 then video , if 1 then drop audio 

        video,audio = load_video(video_path,self.processed_dir,self.debug,audio_path)
        video, audio = self.sample_frames(video, audio)
        video = self.video_transform(video,type=type)
        audio = self.audio_transform(audio,type=type)

        # if modality_to_drop==0:
        #     video,video_flag=self.dropout(video)
        # else :
        #     audio,audio_flag=self.dropout(audio)

        audio,video,flag=self.dropout(audio=audio,video=video,type=type)
        if self.debug:
            print("flag value",flag)

        

        target = torch.tensor(0) if method == "real" else torch.tensor(1)

        if self.debug:
            print("video shape",video.shape)
            print("audio shape",audio.shape)


        video = video.contiguous()
        audio = audio.contiguous()
        return {
            "video": video,    # [T, C, H, W]
            "audio": audio,    # [T, 1]
            "target": target ,  # scalar label
            "flag": flag  # audio dropped or not
        }
        

    def __len__(self):
        return len(self.df)
    




if __name__ == "__main__":

    from torch.utils.data import  WeightedRandomSampler,DataLoader
    from collections import Counter
    train_dataset = CELEB_AV(
        # root_dir="/home/manik/Downloads/FakeAVCeleb_v1.2",
        unprocessed_dir=None,
        csv_file="/home/manik/Downloads/FakeAvCelebPreprocessed/df_matching_frames.csv",
        subset="train",
        modality_drop_rate=0.4,
        preprocessed_dir="/home/manik/Downloads/FakeAvCelebPreprocessed",
        num_frames=25,
        debug=False
    )

    # labels = []
    # for i in range(len(train_dataset.df)):
    #     method = str(train_dataset.df.iloc[i]["method"]).lower()
    #     labels.append(0 if method == "real" else 1)


    # class_counts = Counter(labels)
    # print("Class counts:", class_counts)
    # print("Total samples:", len(train_dataset))
    # class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    # sample_weights = [class_weights[label] for label in labels]
    # sampler = WeightedRandomSampler(
    # weights=sample_weights,
    # num_samples=len(sample_weights),  # can increase to oversample further
    # replacement=True
    # )






    batch_size = 8
    sampler = sampler.OversamplingBalancedBatchSampler(train_dataset, batch_size=batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        # num_workers=4,
        pin_memory=True
    )

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)
    # # train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    for batch in train_loader:
        print(batch["target"])  # should always have equal 0s and 1s
        # print(batch['video'].shape)
        # print(batch['audio'].shape)
        break

