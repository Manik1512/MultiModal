import pytorch_lightning as pl 
pl.seed_everything(42, workers=True)
from datamodule import transforms ,sampler
from datamodule import av_dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torch 
from datetime import datetime
from models.lip_sync_stream import lip_sync_stream
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from hydra import  compose,initialize
initialize(config_path="configs", version_base="1.3")
import pytorch_lightning as pl
import os

cfg = compose(
        config_name="config",
    )


train_dataset = av_dataset.CELEB_AV(
    unprocessed_dir=None,
    csv_file=cfg.trainer.csv_file,
    subset="train",
    # modality_drop_rate=cfg.trainer.modality_drop_rate,
    modality_drop_rate=0,

    preprocessed_dir=cfg.trainer.preprocessed_dir,
    num_frames=cfg.trainer.num_frames,
    debug=False
)

val_dataset = av_dataset.CELEB_AV(
    unprocessed_dir=None,
    csv_file=cfg.trainer.csv_file,
    subset="test",
    modality_drop_rate=1,
    preprocessed_dir=cfg.trainer.preprocessed_dir,
    num_frames=cfg.trainer.num_frames,
    debug=False
)


SAVE_DIR = cfg.trainer.logg_dir
version="OversamplingBalancedBatchSampler"
EXPERIMENT_NAME = "lip_stream/fine_tune_AVSREAM/Start_4_unfreezed_gradientAcc"
logger = TensorBoardLogger(
        save_dir=SAVE_DIR,
        name=EXPERIMENT_NAME,
        version=version
    )
ckpt_saved_path=os.path.join(SAVE_DIR,EXPERIMENT_NAME,version)

# from torch.utils.data import  WeightedRandomSampler
# from collections import Counter



# sampler = sampler.BalancedBatchSampler(train_dataset, batch_size=cfg.trainer.batch_size)

sampler = sampler.OversamplingBalancedBatchSampler(train_dataset, batch_size=cfg.trainer.batch_size)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler,num_workers=cfg.trainer.num_workers,pin_memory=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.trainer.batch_size,num_workers=cfg.trainer.num_workers,pin_memory=True,shuffle=False)
 

test_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=64,
    num_workers=8   ,          # <— use 0 or 1
    pin_memory=False,       # <— turn off for validation
    shuffle=False,
    persistent_workers=False
)
print(f"Size of test dataset: {len(test_loader.dataset)}")
print(f"Number of batches in test_loader: {len(test_loader)}")

# If the number of batches is 0, the test loop will be skipped.
if len(test_loader) == 0:
    print("❌ Error: The test_loader is empty. The test loop will be skipped.")
else:
    print("✅ test_loader is not empty. Starting testing...")

steps_per_epoch = len(train_loader)
# print(f"Steps per epoch: {steps_per_epoch}")


# checkpoint_path = "/home/manik/Documents/experiments/av_stream/lip_stream/fine_tune_AVSREAM/Start_4_unfreezed_BATCH32/OversamplingBalancedBatchSampler/checkpoints/best_model-epoch=18-val_loss=0.03.ckpt"

checkpoint_path="/home/manik/Documents/experiments/av_stream/lip_stream/fine_tune_AVSREAM/Start_2_unfreezed_split_by_source/OversamplingBalancedBatchSampler_accumulate_grad_batches_correctROI_128/checkpoints/best_E20.ckpt"




model = lip_sync_stream(debug=False,cfg=cfg)
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
model.load_state_dict(checkpoint['state_dict'], strict=False)
# av_model=lip_sync_stream.lip_sync_stream(cfg,debug=False,steps_per_epoch=steps_per_epoch,unfreezed_conformers=4)
# model = lip_sync_stream.lip_sync_stream.load_from_checkpoint("/home/manik/Documents/experiments/av_stream/lip_stream/fine_tune_AVSREAM/Start_4_unfreezed_BATCH32/OversamplingBalancedBatchSampler/checkpoints/best_model-epoch=18-val_loss=0.03.ckpt",debug=False)
trainer = pl.Trainer(accelerator='gpu', devices=1)


# trainer.fit(av_model,train_dataloaders=train_loader,val_dataloaders=val_loader,ckpt_path="/home/manik/Documents/experiments/av_stream/lip_stream/fine_tune_AVSREAM/Start_4_unfreezed_BATCH32/OversamplingBalancedBatchSampler_reducedLR/checkpoints/last.ckpt")
trainer.test(model=model, dataloaders=test_loader)