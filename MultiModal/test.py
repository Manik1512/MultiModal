import pytorch_lightning as pl 
pl.seed_everything(42, workers=True)
from datamodule import av_dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torch 
from models.lip_sync_stream import lip_sync_stream
from hydra import  compose,initialize
initialize(config_path="configs", version_base="1.3")
import pytorch_lightning as pl
import os
import argparse
parser = argparse.ArgumentParser(description="Example script with arguments")

cfg = compose(
        config_name="config",
    )

checkpoint_path="/home/manik/Documents/experiments/av_stream/lip_stream/fine_tune_AVSREAM/Start_2_unfreezed_split_by_source/correctROI_128_ModalDrop_Features_add_GatedFuss_CorrectDropout/checkpoints/best_model-epoch=14-val_loss=0.03.ckpt"
# checkpoint_path="/home/manik/Documents/experiments/av_stream/lip_stream/fine_tune_AVSREAM/Start_2_unfreezed_split_by_source/correctROI_128_ModalDrop_Features_add_CorrectDropout/checkpoints/best.ckpt"
# checkpoint_path="/home/manik/Documents/experiments/av_stream/lip_stream/fine_tune_AVSREAM/Start_6V2A_unfreezed_split_by_source/correctROI_128_ModalDrop_Features_add_GatedFuss_CorrectDropout/checkpoints/best.ckpt"

checkpoint_path="/home/manik/Documents/experiments/av_stream/lip_stream/fine_tune_AVSREAM/Start_4V2A_unfreezed_split_by_source/correctROI_128_ModalDrop_Features_add_GatedFuss_CorrectDropout/checkpoints/best.ckpt"


model = lip_sync_stream(debug=False,cfg=cfg,feature_add=True,gated_fusion=False)
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
model.load_state_dict(checkpoint['state_dict'], strict=False)
trainer = pl.Trainer(accelerator='gpu', devices=1)


if __name__ == "__main__":
    parser.add_argument("--dataset", type=str, default="FakeAvCeleb", help="dataset u r testing on")
    parser.add_argument("--dropout", type=int, default=0, help="dataset u r testing on")

    args = parser.parse_args()
    deepfake_types = ["RealVideo-RealAudio","RealVideo-FakeAudio","FakeVideo-FakeAudio","FakeVideo-RealAudio"]
    test_datasets=[]
    if args.dataset=="FakeAvCeleb":

        
        for deepfake in deepfake_types:
            val_dataset = av_dataset.CELEB_AV(
            unprocessed_dir=None,
            csv_file=cfg.trainer.csv_file,
            subset="test",
            modality_drop_rate=args.dropout,
            preprocessed_dir=cfg.trainer.preprocessed_dir,
            num_frames=cfg.trainer.num_frames,
            debug=False,
            deepfake_type=deepfake,
            dataset_type=args.dataset,
            )
            test_datasets.append(val_dataset)
        
        val_dataset = av_dataset.CELEB_AV(
            unprocessed_dir=None,
            csv_file=cfg.trainer.csv_file,
            subset="test",
            modality_drop_rate=args.dropout,
            preprocessed_dir=cfg.trainer.preprocessed_dir,
            num_frames=cfg.trainer.num_frames,
            debug=False,
            deepfake_type=None,
            dataset_type=args.dataset,
            )
        test_datasets.append(val_dataset)
        deepfake_types.append("FULL_dataset")
    

    elif args.dataset=="other":
        val_dataset = av_dataset.CELEB_AV(
        unprocessed_dir=None,
        csv_file="/home/manik/Downloads/DeepfakeTIMIT/meta_data.csv",
        subset="test",
        modality_drop_rate=args.dropout,
        preprocessed_dir=cfg.trainer.preprocessed_dir,
        num_frames=cfg.trainer.num_frames,
        debug=False,
        deepfake_type=None,
        dataset_type="directory_based",
        )
        test_datasets.append(val_dataset)

    data_loaders=[]
    for test_dataset in test_datasets:
        test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        num_workers=8   ,          # <— use 0 or 1
        pin_memory=False,       # <— turn off for validation
        shuffle=False,
        persistent_workers=False
        )
        data_loaders.append(test_loader)

    
    if args.dataset=="FakeAvCeleb":
        for idx in range(0,len(deepfake_types)):
            print("Predicting for===>",deepfake_types[idx])
            trainer.test(model=model, dataloaders=data_loaders[idx])

    elif args.dataset=="other":
        print("Predicting for===>DeepfakeTIMIT",)
        trainer.test(model=model, dataloaders=data_loaders[0])
