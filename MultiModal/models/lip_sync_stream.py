import sys
sys.path.insert(0, "../")
from datamodule.transforms import TextTransform
from pytorch_lightning import LightningModule
from espnet_av.nets.pytorch_backend.e2e_asr_conformer_av import E2E
import torch
import torchmetrics
from hydra import initialize, compose
from torch import nn
import torch.functional as F
initialize(config_path="../configs", version_base="1.3")

class Feature_extraction_av(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.backbone_args = self.cfg.model.audiovisual_backbone
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)

    def forward(self, video, audio):
        video_feat, _ = self.model.encoder(video.unsqueeze(0).to(self.device), None)

        print("video shape after encoder",video_feat.shape)

        audio_feat, _ = self.model.aux_encoder(audio.unsqueeze(0).to(self.device), None)

        print("audio shape after encoder",audio_feat.shape)
        audiovisual_feat = self.model.fusion(torch.cat((video_feat, audio_feat), dim=-1))
        print("audiovisual shape after encoder",audiovisual_feat.shape)

        return audiovisual_feat
    
    def load_weights(self, ckpt_path):
        """
        Load model weights from a checkpoint file.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=False)
        print("Model weights loaded successfully")



class lip_sync_stream(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.feature_dim=self.cfg.model.lip_sync_model.feature_dim
        self.feature_extractor = Feature_extraction_av(cfg)
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.feature_extractor.load_weights(self.cfg.model.lip_sync_model.avsr_path)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(self.cfg.model.lip_sync_model.dropout_rate),
            nn.Linear(128, 1),
        )

        self.criterion = nn.BCEWithLogitsLoss()

        metrics = {
            "acc": torchmetrics.Accuracy(task="binary"),
            "prec": torchmetrics.Precision(task="binary"),
            "rec": torchmetrics.Recall(task="binary"),
            "f1": torchmetrics.F1Score(task="binary"),
        }
        self.train_metrics = nn.ModuleDict({k: m.clone() for k, m in metrics.items()})


        """Here you just reuse the original metric objects for validation.
            Since you cloned them for training already, no problem of overlap.
            train and val should have different metrics objects  """
        self.val_metrics = nn.ModuleDict(metrics)

    def forward(self, video, audio):
        features =self.feature_extractor(video, audio)
        features = torch.mean(features, dim=1)
        features = self.feature_norm(features)
        logits=self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, step_type="train")
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, step_type="val")
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def _step(self, batch, step_type):
        video, audio, y = batch
        logits = self(video, audio).squeeze(1)
        loss = self.criterion(logits, y.float())
        preds = (torch.sigmoid(logits) > 0.5).int()

        if step_type == "train":
            # update train metrics
            for m in self.train_metrics.values():
                m.update(preds, y.int())
        else:
            # update val metrics
            for m in self.val_metrics.values():
                m.update(preds, y.int())

        return loss
    
    def on_train_epoch_end(self):
        """Log metrics only every 2 epochs."""
        if (self.current_epoch + 1) % 2 == 0:
            for name, metric in self.train_metrics.items():
                val = metric.compute()
                self.log(f"train/{name}", val, prog_bar=True)
                metric.reset()

    def on_validation_epoch_end(self):
        """Log metrics every validation epoch."""
        for name, metric in self.val_metrics.items():
            val = metric.compute()
            self.log(f"val/{name}", val, prog_bar=True)
            metric.reset()
    



if __name__ == "__main__":
    
    cfg = compose(
        config_name="config",
    )
    video= torch.randn((50, 1, 56, 56))
    audio=torch.randn((32000,1))

    model = lip_sync_stream(cfg)
    features=model.forward(video, audio)
    print("ouput shape", features.shape)
