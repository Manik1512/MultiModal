
from datamodule import transforms ,sampler
from datamodule import av_dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torch 
from datetime import datetime
from models import lip_sync_stream
from pytorch_lightning.callbacks import EarlyStopping
from hydra import  compose,initialize
initialize(config_path="configs", version_base="1.3")
import pytorch_lightning as pl

cfg = compose(
        config_name="config",
    )


train_dataset = av_dataset.CELEB_AV(
    unprocessed_dir=None,
    csv_file=cfg.trainer.csv_file,
    subset="train",
    modality_drop_rate=cfg.trainer.modality_drop_rate,
    preprocessed_dir=cfg.trainer.preprocessed_dir,
    num_frames=cfg.trainer.num_frames,
    debug=False
)

val_dataset = av_dataset.CELEB_AV(
    unprocessed_dir=None,
    csv_file=cfg.trainer.csv_file,
    subset="val",
    modality_drop_rate=0,
    preprocessed_dir=cfg.trainer.preprocessed_dir,
    num_frames=cfg.trainer.num_frames,
    debug=False
)


SAVE_DIR = cfg.trainer.logg_dir
EXPERIMENT_NAME = "feature_extractor_freezed"
RUN_NAME = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
# log_path = os.path.join(SAVE_DIR, EXPERIMENT_NAME, RUN_NAME)
logger = TensorBoardLogger(
        save_dir=SAVE_DIR,
        name=EXPERIMENT_NAME,
        version=RUN_NAME
    )

sampler = sampler.BalancedBatchSampler(train_dataset, batch_size=2)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.trainer.batch_size,shuffle=True,num_workers=cfg.trainer.num_workers,pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.trainer.batch_size,num_workers=cfg.trainer.num_workers,pin_memory=True,shuffle=False)


av_model=lip_sync_stream.lip_sync_stream(cfg,debug=False)


trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto", # Automatically uses GPU if available
        logger=logger,
        precision="16-mixed",
        # callbacks=EarlyStopping(monitor='val/loss', patience=3, mode='min')
    )

trainer.fit(av_model,train_dataloaders=train_loader,val_dataloaders=val_loader)