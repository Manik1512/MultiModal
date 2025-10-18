import torch
from torch.utils.data import Sampler
import random

class BalancedBatchSampler(Sampler):
    """
    Ensures each batch has 50% real and 50% fake samples.
    It undersamples the majority class to achieve this balance.
    The number of batches is determined by the smaller class size.
    Note: Batch size must be even.
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



import numpy as np

class OversamplingBalancedBatchSampler(Sampler):
    """
    Ensures each batch has a 50-50 class split, with the total number of batches
    determined by `len(dataset) // batch_size`.

    This sampler uses a hybrid strategy:
    - It samples the **majority** class **without replacement**.
    - It samples the **minority** class **with replacement**.

    This is useful when you want to see every majority class sample at most
    once per epoch, while still forcing a balanced batch composition over a
    longer epoch defined by the total dataset size.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        assert batch_size % 2 == 0, "Batch size must be even for 50-50 sampling"

        # Split indices by class
        real_indices = [i for i, row in dataset.df.iterrows() if str(row["method"]).lower() == "real"]
        fake_indices = [i for i, row in dataset.df.iterrows() if str(row["method"]).lower() != "real"]

        # Identify majority and minority classes
        if len(real_indices) > len(fake_indices):
            self.majority_indices = real_indices
            self.minority_indices = fake_indices
        else:
            self.majority_indices = fake_indices
            self.minority_indices = real_indices
        
        if not self.minority_indices:
            raise ValueError("Minority class has no samples.")

        # Calculate the number of batches based on total dataset size
        self.num_batches = len(self.dataset) // self.batch_size
        self.half_batch = self.batch_size // 2

        # Check if we have enough majority samples for the entire epoch without replacement
        required_majority_samples = self.num_batches * self.half_batch
        if required_majority_samples > len(self.majority_indices):
            raise ValueError(
                f"Cannot create {self.num_batches} batches of size {self.batch_size} "
                f"without replacing majority class samples. Required {required_majority_samples} "
                f"majority samples, but only {len(self.majority_indices)} are available. "
                "Consider reducing the batch size or using a different sampling strategy."
            )

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Shuffle the majority indices once for the epoch
        majority_pool = self.majority_indices.copy()
        random.shuffle(majority_pool)

        for i in range(self.num_batches):
            # Get the next chunk of majority samples without replacement
            start_idx = i * self.half_batch
            end_idx = (i + 1) * self.half_batch
            majority_batch = majority_pool[start_idx:end_idx]

            # Sample the minority class with replacement to fill the other half
            minority_batch = random.choices(self.minority_indices, k=self.half_batch)

            # Combine and shuffle for the final batch
            batch = majority_batch + minority_batch
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




