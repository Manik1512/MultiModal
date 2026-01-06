version='4Conf_audioVideoDropout'
EXPERIMENT_NAME = "lip_stream/only_FeatureAddition_GatedFusion_True"
EXPERIMENT_NAME = "lip_stream/only_feature_add_True"



import pytorch_lightning as pl 
pl.seed_everything(42, workers=True)
from datamodule import transforms ,sampler
from datamodule import av_dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torch 
from datetime import datetime
from models import lip_sync_stream
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
    modality_drop_rate=cfg.trainer.modality_drop_rate,
    # modality_drop_rate=0.5,

    preprocessed_dir=cfg.trainer.preprocessed_dir,
    num_frames=cfg.trainer.num_frames,
    debug=False,
    dataset_type="FakeAvCeleb"
)

val_dataset = av_dataset.CELEB_AV(
    unprocessed_dir=None,
    csv_file=cfg.trainer.csv_file,
    subset="val",
    modality_drop_rate=0,
    preprocessed_dir=cfg.trainer.preprocessed_dir,
    num_frames=cfg.trainer.num_frames,
    debug=False,
    dataset_type="FakeAvCeleb"
)


SAVE_DIR = cfg.trainer.logg_dir

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
 

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=cfg.trainer.batch_size,
    num_workers=4   ,          # <— use 0 or 1
    pin_memory=False,       # <— turn off for validation
    shuffle=False,
    persistent_workers=False
)


steps_per_epoch = len(train_loader)
print(f"Steps per epoch: {steps_per_epoch}")

av_model=lip_sync_stream.lip_sync_stream(cfg,
                                         debug=False,
                                         steps_per_epoch=steps_per_epoch,
                                         unfreezed_conformers=cfg.trainer.unfreezed_conformers,
                                         gated_fusion=cfg.trainer.gated_fusion,
                                         feature_add=cfg.trainer.feature_add,
                                         attention_pooling=cfg.trainer.attention_pooling
                                         )
from torchinfo import summary
video= torch.randn((2,25, 1, 56, 56))
audio=torch.randn((2,16000,1))
summary(av_model, input_size=[video.shape,audio.shape])



best_model_callback = ModelCheckpoint(
    dirpath=os.path.join(ckpt_saved_path,'checkpoints/'),
    filename='best_model-{epoch:02d}-{val_loss:.2f}',
    monitor='val_loss',
    mode='min',
    save_top_k=1, # This is the key argument! It saves the single best model.
)
latest_model_callback = ModelCheckpoint(
    dirpath=os.path.join(ckpt_saved_path,'checkpoints/'),
    filename='latest_model-{epoch:02d}',
    save_top_k=0, # Does not save based on a metric
    save_last=True # This is the key argument to save the final model!
)
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=4,
    mode='min'
)

trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto", # Automatically uses GPU if available
        logger=logger,
        precision="16-mixed",
        callbacks=[best_model_callback,latest_model_callback],
        accumulate_grad_batches=6
    )

# ckpt_path="/home/manik/Documents/experiments/av_stream/lip_stream/all_false/version1/checkpoints/last.ckpt"
# trainer.fit(av_model,train_dataloaders=train_loader,val_dataloaders=val_loader,ckpt_path=ckpt_path)
trainer.fit(av_model,train_dataloaders=train_loader,val_dataloaders=val_loader) 