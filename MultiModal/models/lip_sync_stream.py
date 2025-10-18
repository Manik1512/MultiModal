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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchmetrics# from torchmetrics.classification import (
#     BinaryPrecision, BinaryRecall, BinaryF1Score,
#     BinaryAUROC, BinaryAveragePrecision
# )
import seaborn as sns
import matplotlib.pyplot as plt
import gc

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
    def __init__(self, cfg,debug,steps_per_epoch=None,unfreezed_conformers=0):
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
            nn.ReLU(),  
            nn.Dropout(self.cfg.model.lip_sync_model.dropout_rate),
            nn.Linear(512, 1),
        )

        self.train_auc = torchmetrics.AUROC(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.criterion = nn.BCEWithLogitsLoss()

        # self.accuracy = tm.Accuracy(task="binary", threshold=0.5, average=None)
        # self.f1_score = tm.F1Score(task="binary", threshold=0.5 , average=None)
        # self.precision = tm.Precision(task="binary", threshold=0.5 , average=None)
        # self.recall = tm.Recall(task="binary", threshold=0.5 , average=None)



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


        count_till=12-unfreezed_conformers
        count=0
        # for layer in self.feature_extractor.model.aux_encoder.encoders:             # yeh last ke unfreeze kerta hai 

        #     if count >=count_till:
        #         for param in layer.parameters():
        #             param.requires_grad = True
        #     count=count+1
        # count=0

        # for layer in self.feature_extractor.model.encoder.encoders:

        #     if count >=count_till:
        #         for param in layer.parameters():
        #             param.requires_grad = True
        #     count=count+1


        for layer in self.feature_extractor.model.aux_encoder.encoders:             

            if count <unfreezed_conformers:
                for param in layer.parameters():
                  param.requires_grad = True

            else:
                layer=nn.Identity()
            count=count+1
        count=0

        for layer in self.feature_extractor.model.encoder.encoders:

            if count <unfreezed_conformers:
                for param in layer.parameters():
                   param.requires_grad = True

            else:
                layer=nn.Identity()
            
            count=count+1



        # self.validation_outputs = []
        self.steps_per_epoch=steps_per_epoch


        self.val_tp = 0
        self.val_fp = 0
        self.val_fn = 0
        self.val_tn = 0

        self.train_tp = 0
        self.train_fp = 0
        self.train_fn = 0
        self.train_tn = 0
            
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
        # probs = torch.sigmoid(logits).detach().cpu()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        preds = preds.detach()
        y = y.detach()

        tp = ((preds == 1) & (y == 1)).sum().item()
        fp = ((preds == 1) & (y == 0)).sum().item()
        fn = ((preds == 0) & (y == 1)).sum().item()
        tn = ((preds == 0) & (y == 0)).sum().item()

        self.train_auc.update(probs, y)


        self.train_tp += tp
        self.train_fp += fp
        self.train_fn += fn
        self.train_tn += tn

        
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch, step_type="val")
        logits=logits.detach()
        loss= loss.detach()
        probs = torch.sigmoid(logits)
        probs=probs.detach()
        preds = (probs > 0.5).int()

        preds = preds.detach()
        y = y.detach()

        tp = ((preds == 1) & (y == 1)).sum().item()
        fp = ((preds == 1) & (y == 0)).sum().item()
        fn = ((preds == 0) & (y == 1)).sum().item()
        tn = ((preds == 0) & (y == 0)).sum().item()

        self.val_auc.update(probs, y)



        self.val_tp += tp
        self.val_fp += fp
        self.val_fn += fn
        self.val_tn += tn

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return None

      
    def on_train_epoch_end(self):
        precision = self.train_tp / (self.train_tp + self.train_fp + 1e-8)
        recall = self.train_tp / (self.train_tp + self.train_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (self.train_tp + self.train_tn) / (
            self.train_tp + self.train_tn + self.train_fp + self.train_fn + 1e-8
        )

        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({
            "train_accuracy": accuracy,
            "train_f1": f1,
            "train_precision": precision,
            "train_recall": recall,
            # "train_auc": auc
        }, prog_bar=True)

        # Reset for next epoch
        self.train_tp = self.train_fp = self.train_fn = self.train_tn = 0

        gc.collect()
        torch.cuda.empty_cache()


    def on_validation_epoch_end(self):
        precision = self.val_tp / (self.val_tp + self.val_fp + 1e-8)
        recall = self.val_tp / (self.val_tp + self.val_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (self.val_tp + self.val_tn) / (
            self.val_tp + self.val_tn + self.val_fp + self.val_fn + 1e-8
        )

        fig = self.plot_confusion_matrix_from_values(self.val_tp, self.val_fp, self.val_tn, self.val_fn)
        self.logger.experiment.add_figure("Confusion Matrix", fig, self.current_epoch)
        plt.close(fig)


        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)

        self.log_dict({
            "val_accuracy": accuracy,
            "val_f1": f1,
            "val_precision": precision,
            "val_recall": recall,
            # "val_auc": auc
        }, prog_bar=True,
        on_epoch=True
        )

        # Reset for next epoch
        self.val_tp = self.val_fp = self.val_fn = self.val_tn = 0
        gc.collect()
        torch.cuda.empty_cache()



    def _step(self, batch, step_type):
        video = batch["video"]
        audio = batch["audio"]
        y = batch["target"]
        logits = self(video, audio).squeeze(1)
        loss = self.criterion(logits, y.float())
        # preds = (torch.sigmoid(logits) > 0.5).int()

        return loss, logits, y



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.trainer.learning_rate,
            weight_decay=self.cfg.trainer.weight_decay
        )

        # steps_per_epoch = self.steps_per_epoch
        # t_0 = 3 * steps_per_epoch  # one cyclee is 5 epochs

        # scheduler = CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=t_0,      # Number of steps in the first restart cycle.
        #     T_mult=1,   # Factor by which to increase T_i after a restart. T_mult=1 means the cycle length is constant.
        #     eta_min=1e-6, # Minimum learning rate.
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)


        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}
        }
    def plot_confusion_matrix_from_values(self,tp, fp, tn, fn, class_names=["Negative", "Positive"]):
        cm = [[tn, fp],
            [fn, tp]]
        
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        return fig


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

    print(model.feature_extractor.model.aux_encoder.encoders[0])
    len=0
    for encoder_layer in model.feature_extractor.model.aux_encoder.encoders:
        len=len+1
    print("len of audio encoder layers",len)    