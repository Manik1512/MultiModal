import torch
from torch.utils.data import Sampler
import random

class BalancedBatchSampler(Sampler):
    """
    Ensures each batch has 50% real and 50% fake samples.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        assert batch_size % 2 == 0, "Batch size must be even for 50-50 sampling"

        # Split indices by class
        self.real_indices = [i for i, row in dataset.df.iterrows() if str(row["method"]).lower() == "real"]
        self.fake_indices = [i for i, row in dataset.df.iterrows() if str(row["method"]).lower() != "real"]

        self.num_batches = min(len(self.real_indices), len(self.fake_indices)) // (batch_size // 2)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        real_pool = self.real_indices.copy()
        fake_pool = self.fake_indices.copy()

        random.shuffle(real_pool)
        random.shuffle(fake_pool)

        for i in range(self.num_batches):
            real_batch = real_pool[i*(self.batch_size//2):(i+1)*(self.batch_size//2)]
            fake_batch = fake_pool[i*(self.batch_size//2):(i+1)*(self.batch_size//2)]
            batch = real_batch + fake_batch
            random.shuffle(batch)
            yield batch



# if __name__ == "__main__":
        
    # from av_dataset import CELEB_AV
    # from transforms import AudioTransform, VideoTransform

    # dataset = CELEB_AV(
    #     unprocessed_dir="/home/manik/Downloads/FakeAVCeleb_v1.2",
    #     csv_file="/home/manik/Downloads/FakeAVCeleb_v1.2/meta_data.csv",
    #     audio_transform=AudioTransform("train"),
    #     video_transform=VideoTransform("train"),
    # )
    # batch_size = 32
    # sampler = BalancedBatchSampler(dataset, batch_size=batch_size)

    # loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    # for batch in loader:
    #     print(batch["target"])  # should always have equal 0s and 1s
    #     break




