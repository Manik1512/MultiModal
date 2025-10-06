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
from einops import rearrange
import torchmetrics as tm
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score


class Feature_extraction_av(LightningModule):
    def __init__(self, cfg,debug):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.backbone_args = self.cfg.model.audiovisual_backbone
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)
        self.debug=debug

    def forward(self, video, audio):
        # print("in forward of feature extraction , video=>",video)
        video_feat, _ = self.model.encoder(video.unsqueeze(0).to(self.device), None)

        # print("video shape after encoder",video_feat.shape)

        audio_feat, _ = self.model.aux_encoder(audio.unsqueeze(0).to(self.device), None)

        
        audiovisual_feat = self.model.fusion(torch.cat((video_feat, audio_feat), dim=-1))

        if self.debug:
            print("audio shape after encoder",audio_feat.shape)
            print("audiovisual shape after encoder",audiovisual_feat.shape)

        return audiovisual_feat
    
    def load_weights(self, ckpt_path):
        """
        Load model weights from a checkpoint file.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=False)
        print("Model weights loaded successfully")



# class lip_sync_stream(LightningModule):
#     def __init__(self, cfg,debug):
#         super().__init__()
#         self.save_hyperparameters(cfg)
#         self.cfg = cfg
#         self.feature_dim=self.cfg.model.lip_sync_model.feature_dim
#         self.feature_extractor = Feature_extraction_av(cfg,debug=debug)
#         self.feature_norm = nn.LayerNorm(self.feature_dim)
#         self.feature_extractor.load_weights(self.cfg.model.lip_sync_model.avsr_path)
#         self.classifier = nn.Sequential(
#             nn.Linear(self.feature_dim, 512),
#             nn.LayerNorm(512),
#             nn.GELU(),  
#             nn.Dropout(self.cfg.model.lip_sync_model.dropout_rate),

#             nn.Linear(512, 128),
#             nn.LayerNorm(128),
#             nn.GELU(),  
#             nn.Dropout(self.cfg.model.lip_sync_model.dropout_rate),

#             nn.Linear(128, 1),
#         )

#         self.criterion = nn.BCEWithLogitsLoss()

#         metrics = {
#             "acc": torchmetrics.Accuracy(task="binary"),
#             "prec": torchmetrics.Precision(task="binary"),
#             "rec": torchmetrics.Recall(task="binary"),
#             "f1": torchmetrics.F1Score(task="binary"),
#         }
#         self.train_metrics = nn.ModuleDict({k: m.clone() for k, m in metrics.items()})
#         self.debug=debug

#         """Here you just reuse the original metric objects for validation.
#             Since you cloned them for training already, no problem of overlap.
#             train and val should have different metrics objects  """
#         self.val_metrics = nn.ModuleDict(metrics)

#         self.feature_extractor.model.decoder=torch.nn.Identity()
#         self.feature_extractor.model.fusion=torch.nn.Identity()
#         self.feature_extractor.model.ctc=torch.nn.Identity()
#         self.feature_extractor.model.criterion=torch.nn.Identity()

#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False


#         self.validation_outputs = []

#     def forward(self, video, audio):
#         video = rearrange(video, 'b t c h w -> b c t h w')
#         features =self.feature_extractor(video, audio)
#         features = torch.mean(features, dim=1)
#         if self.debug:
#             print("shape after mean ",features.shape)
#             print("input video shape",video.shape)
#         features = self.feature_norm(features)
#         logits=self.classifier(features)
#         return logits

#     def training_step(self, batch, batch_idx):
#         loss, preds, y  = self._step(batch, step_type="train")
#         preds = preds.detach().cpu()
#         y = y.detach().cpu()

#         for m in self.train_metrics.values():
#             m.update(preds, y.int())

#         # log batch loss
#         self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
#         return loss


#     @torch.no_grad()
#     def validation_step(self, batch, batch_idx):
#         loss, preds, y = self._step(batch, step_type="val")

#         # move preds & labels to CPU before metric updates
#         preds = preds.detach().cpu()
#         y = y.detach().cpu()

#         for m in self.val_metrics.values():
#             m.update(preds, y.int())

#         # log batch loss (optional, but use on_epoch instead of on_step)
#         self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)

#         return loss
    
#     def _step(self, batch, step_type):
#         video = batch["video"]
#         audio = batch["audio"]
#         y = batch["target"]
#         logits = self(video, audio).squeeze(1)
#         loss = self.criterion(logits, y.float())
#         preds = (torch.sigmoid(logits) > 0.5).int()

#         return loss, preds, y
    
#     def on_train_epoch_end(self):
#         # compute once per epoch
#         for name, metric in self.train_metrics.items():
#             val = metric.compute()
#             self.log(f"train/{name}", val, prog_bar=True)
#             metric.reset()

#     def on_validation_epoch_end(self):
#         for name, metric in self.val_metrics.items():
#             val = metric.compute()
#             self.log(f"val/{name}", val, prog_bar=True)
#             metric.reset()
            
#     def configure_optimizers(self):
#         optimizer=torch.optim.AdamW(self.parameters(), lr=self.cfg.trainer.learning_rate,weight_decay=self.cfg.trainer.weight_decay)
#         scheduler = {
#         'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode='min',
#             factor=0.5,
#             patience=3,
#             verbose=True,
#             min_lr=1e-6
#         ),
#         'monitor': 'val_loss',  # must match the metric you log in validation_step
#         'interval': 'epoch',    # check metric each epoch
#         'frequency': 1
#     }
#         return {"optimizer": optimizer, "lr_scheduler": scheduler}


class lip_sync_stream(LightningModule):
    def __init__(self, cfg,debug):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.feature_dim=self.cfg.model.lip_sync_model.feature_dim
        self.feature_extractor = Feature_extraction_av(cfg,debug=debug)
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.feature_extractor.load_weights(self.cfg.model.lip_sync_model.avsr_path)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),  
            nn.Dropout(self.cfg.model.lip_sync_model.dropout_rate),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),  
            nn.Dropout(self.cfg.model.lip_sync_model.dropout_rate),

            nn.Linear(128, 1),
        )

        self.criterion = nn.BCEWithLogitsLoss()

        self.accuracy = tm.Accuracy(task="binary", threshold=0.5)
        self.f1_score = tm.F1Score(task="binary", threshold=0.5)
        self.precision = tm.Precision(task="binary", threshold=0.5)
        self.recall = tm.Recall(task="binary", threshold=0.5)
        # self.train_metrics = nn.ModuleDict({k: m.clone() for k, m in metrics.items()})
        self.debug=debug

        """Here you just reuse the original metric objects for validation.
            Since you cloned them for training already, no problem of overlap.
            train and val should have different metrics objects  """
        # self.val_metrics = nn.ModuleDict(metrics)

        self.feature_extractor.model.decoder=torch.nn.Identity()
        self.feature_extractor.model.fusion=torch.nn.Identity()
        self.feature_extractor.model.ctc=torch.nn.Identity()
        self.feature_extractor.model.criterion=torch.nn.Identity()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False


        self.validation_outputs = []

    def forward(self, video, audio):
        video = rearrange(video, 'b t c h w -> b c t h w')
        features =self.feature_extractor(video, audio)
        features = torch.mean(features, dim=1)
        if self.debug:
            print("shape after mean ",features.shape)
            print("input video shape",video.shape)
        features = self.feature_norm(features)
        logits=self.classifier(features)
        return logits


    def training_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch, step_type="train")
        probs = torch.sigmoid(logits).detach().cpu()
        y = y.detach().cpu()
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch, step_type="val")

        # keep everything on GPU for loss, move to CPU only for metrics if needed
        probs = torch.sigmoid(logits)

        # update metrics directly (Lightning/TorchMetrics can handle GPU tensors)
        self.accuracy.update(probs, y.int())
        self.f1_score.update(probs, y.int())
        self.precision.update(probs, y.int())
        self.recall.update(probs, y.int())

        # log the loss as scalar
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def _step(self, batch, step_type):
        video = batch["video"]
        audio = batch["audio"]
        y = batch["target"]
        logits = self(video, audio).squeeze(1)
        loss = self.criterion(logits, y.float())
        # preds = (torch.sigmoid(logits) > 0.5).int()

        return loss, logits, y

    def on_validation_epoch_end(self):
        self.log_dict({
            "val_accuracy": self.accuracy.compute(),
            "val_f1": self.f1_score.compute(),
            "val_precision": self.precision.compute(),
            "val_recall": self.recall.compute(),
        }, prog_bar=True)

        # Reset metrics
        self.accuracy.reset()
        self.f1_score.reset()
        self.precision.reset()
        self.recall.reset()
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.trainer.learning_rate,
            weight_decay=self.cfg.trainer.weight_decay
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True,
                min_lr=1e-6
            ),
            "monitor": "val_loss",   # must match exact key from self.log
            "interval": "epoch",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



if __name__ == "__main__":
    initialize(config_path="../configs", version_base="1.3")

    from torchinfo import summary
    from collections import Counter
    cfg = compose(
        config_name="config",
    )
    video= torch.randn((2,25, 1, 56, 56))
    audio=torch.randn((2,16000,1))

    model = lip_sync_stream(cfg,debug=True)
    features=model.forward(video, audio)
    print("output shape", features.shape)


    # summary(model, input_size=[video.shape,audio.shape])

    # print(model.parameters)