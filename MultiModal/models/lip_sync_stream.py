import sys
sys.path.insert(0, "../")
from datamodule.transforms import TextTransform
from pytorch_lightning import LightningModule
from espnet_av.nets.pytorch_backend.e2e_asr_conformer_av import E2E
import torch
import torchmetrics
from hydra import initialize, compose
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchmetrics# from torchmetrics.classification import (
import seaborn as sns
import matplotlib.pyplot as plt
import gc


# class CrossAttention(nn.Module):
#     """ Cross-Attention module allowing a primary stream to attend to a secondary stream.
#         If no secondary stream is provided, it defaults to self-attention.
#         if attention_pooling is True, it uses a learnable query vector for attention pooling.

#         For consistency take , dim_heads=dim//heads, in that case => inner_dim=dim
#     """
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.0,attention_pooling=False):
#         super().__init__()

#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5  # Manual scaling is not needed for F.scaled_dot_product_attention but good practice to have
#         self.dim_head = dim_head
        
#         self.dropout_rate = dropout
#         # Keys and Values are projected from the 'context' stream in one go
#         self.attention_pooling=attention_pooling
#         if self.attention_pooling:
#             self.q=nn.Parameter(torch.randn(1, dim_head*heads)) 
#         else :
#             self.to_q = nn.Linear(dim, inner_dim, bias=False)

#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, context=None, mask=None):
#         """
#         Args:
#             x (torch.Tensor): The primary stream(query stream).
#                               Shape: (batch, seq_len_q, dim)

#             context :        This is the second stream (key/value stream).
#                                             If None, performs self-attention on 'x'.
#                                             Shape: (batch, seq_len_kv, dim)
#             mask (torch.Tensor, optional): Boolean attention mask. Not typically used
#                                            in cross-attention but included for completeness.
#                                            Shape: (batch, seq_len_q, seq_len_kv)

#         Returns:
#             torch.Tensor: The output tensor after attention.
#                           Shape: (batch, seq_len_q, dim)
#         """
#         # If no context is provided, default to self-attention
#         # This makes the module more versatile
#         context = context if context is not None else x

#         if self.attention_pooling:
#             q = self.q.unsqueeze(0).repeat(x.shape[0], 1, 1)  # (b, 1, inner_dim)  ,inner_dim=heads*dim_head
            

#         else :
#             # (b, n, dim) -> (b, n, h*d)
#             q = self.to_q(x)
#             # q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        

#         # (b, m, dim) -> (b, m, h*d*2)
#         k, v = self.to_kv(context).chunk(2, dim=-1)

#         # (b, m, h*d) -> (b, h, m, d)
#         q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads) # (b, n, h*d) -> (b, h, n, d)
#         k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
#         v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

#         out = F.scaled_dot_product_attention(
#             q, k, v, attn_mask=mask, dropout_p=self.dropout_rate if self.training else 0.0
#         )

        
#         if self.attention_pooling:
#             out = out.squeeze(1)  # (b, dim)
#         # (b, h, n, d) -> (b, n, h*d)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

class AttentionPooling(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        
        self.dropout_rate = dropout
        
        self.q=nn.Parameter(torch.randn(1, dim_head*heads)) 
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self,x,mask=None):
            # x : (b, n, dim)
            q = self.q.unsqueeze(0).repeat(x.shape[0], 1, 1)  # (b, 1, inner_dim)  ,inner_dim=heads*dim_head
            k, v = self.to_kv(x).chunk(2, dim=-1)

            # (b, m, h*d) -> (b, h, m, d)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads) # (b, 1, h*d) → (b, h, 1, d)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout_rate if self.training else 0.0
            )
            out = out.squeeze(2)  # (b, dim)
            out = rearrange(out, 'b h d -> b (h d)')
            out = self.to_out(out)
            return out


class GatedFusion(nn.Module):
    """Weighted Addition of two features
       where weights are learnable parameters
      and the function of input"""
    def __init__(self,
                 dim
                 ):
        super().__init__()
        self.gate_layer=nn.Linear(dim*2,dim,bias=True)
        self.sigmoid=nn.Sigmoid()
        self.norm = nn.LayerNorm(dim)

    def forward(self,x1,x2):
        "Input=Output->(B,T,D)"
        alpha=self.sigmoid(self.gate_layer(torch.cat((x1, x2), dim=-1)))
        # print("sgidfhgkshfjsh")
        fused_output = (alpha * x2) + ((1 - alpha) * x1)
        return self.norm(fused_output)
        




class Feature_extraction_av(LightningModule):
    def __init__(self, cfg,debug,feature_add,gated_fusion):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.backbone_args = self.cfg.model.audiovisual_backbone
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)
        self.debug=debug
        self.gated_fusion=gated_fusion
        self.feature_add=feature_add
        if self.feature_add:
            self.feature_dim=self.cfg.model.lip_sync_model.feature_dim//2
            self.audio_norm=nn.LayerNorm(self.feature_dim)
            self.video_norm=nn.LayerNorm(self.feature_dim)
            if self.gated_fusion:
                self.fusion_layer=GatedFusion(self.feature_dim)
        else:
            self.feature_dim=self.cfg.model.lip_sync_model.feature_dim

        

        
    def forward(self, video, audio):
        # print("in forward of feature extraction , video=>",video)
        video_feat, _ = self.model.encoder(video.unsqueeze(0).to(self.device), None)
        audio_feat, _ = self.model.aux_encoder(audio.unsqueeze(0).to(self.device), None)

        
        if self.feature_add:
            audio_feat=self.audio_norm(audio_feat)
            video_feat=self.video_norm(video_feat)
            if self.gated_fusion:
                fused_feat=self.fusion_layer(audio_feat,video_feat)
            else:
                fused_feat=audio_feat+video_feat
        else:
            fused_feat=torch.concat((video_feat,audio_feat),dim=-1)
        audiovisual_feat = self.model.fusion(fused_feat)

        if self.debug:
            print("video shape before fusion",video_feat.shape)
            print("audio shape before fusion",audio_feat.shape)
            print("audiovisual shape after encoder",audiovisual_feat.shape)

        return audiovisual_feat
    
    def load_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=False)
        print("Model weights loaded successfully")



class lip_sync_stream(LightningModule):
    def __init__(self, cfg,
                 debug,
                 steps_per_epoch=None,
                 unfreezed_conformers=4,
                 feature_add=False,
                 gated_fusion=False,
                 attention_pooling=False

                 ):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.feature_add=feature_add
        if feature_add:
            self.feature_dim=self.cfg.model.lip_sync_model.feature_dim//2
        else: 
            self.feature_dim=self.cfg.model.lip_sync_model.feature_dim
        self.feature_extractor = Feature_extraction_av(cfg,debug=debug,feature_add=feature_add,gated_fusion=gated_fusion)
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.feature_extractor.load_weights(self.cfg.model.lip_sync_model.avsr_path)

        print(f"shape of extracted features from FeatExtr=>{self.feature_dim}" if debug else "")

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),  
            nn.Dropout(self.cfg.model.lip_sync_model.dropout_rate),
            nn.Linear(128, 1),
        )

        self.train_auc = torchmetrics.AUROC(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.test_auc = torchmetrics.AUROC(task="binary")   
        # self.val_auc = torchmetrics.AUROC(task="binary")
        self.criterion = nn.BCEWithLogitsLoss()
        self.debug=debug

        self.attention_pooling=None
        if attention_pooling:
            self.attention_pooling=AttentionPooling(heads=self.cfg.model.lip_sync_model.pooling_heads,
                                                    dim_head=self.cfg.model.lip_sync_model.pooling_dk,
                                                    dim=self.feature_dim)
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


        count=0

        for layer in self.feature_extractor.model.aux_encoder.encoders:             

            if count <unfreezed_conformers:
                for param in layer.parameters():
                  param.requires_grad = True

            else:
                layer=nn.Identity()
            count=count+1
        count=0
        unfreezed_conformers=6
        for layer in self.feature_extractor.model.encoder.encoders:

            if count <unfreezed_conformers:
                for param in layer.parameters():
                   param.requires_grad = True

            else:
                layer=nn.Identity()
            count=count+1

        self.steps_per_epoch=steps_per_epoch

        self.val_tp = 0
        self.val_fp = 0
        self.val_fn = 0
        self.val_tn = 0

        self.train_tp = 0
        self.train_fp = 0
        self.train_fn = 0
        self.train_tn = 0

        self.test_tp = 0
        self.test_fp = 0
        self.test_fn = 0
        self.test_tn = 0
            
    def forward(self, video, audio):
        video = rearrange(video, 'b t c h w -> b c t h w')
        features =self.feature_extractor(video, audio)

        if self.attention_pooling is not None:
            features = self.attention_pooling(features)
        else:
            features = torch.mean(features, dim=1)   #MEAN over time dimesnion (B, T, D) -> (B, D)
        if self.debug:
            print("shape after mean ",features.shape)
        if not self.feature_add:
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
    
    def test_step(self, batch, batch_idx):
        # The '_step' call might be for 'test' or 'val', depending on your implementation.
        # We'll assume it's generic enough to be used for both.
        loss, logits, y = self._step(batch, step_type="test")

        # Detach all tensors to prevent memory leaks during inference
        loss = loss.detach()
        logits = logits.detach()
        y = y.detach()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        # Manually calculate confusion matrix components
        tp = ((preds == 1) & (y == 1)).sum().item()
        fp = ((preds == 1) & (y == 0)).sum().item()
        fn = ((preds == 0) & (y == 1)).sum().item()
        tn = ((preds == 0) & (y == 0)).sum().item()

        # Update metrics
        self.test_auc.update(probs, y)
        

        self.test_tp += tp
        self.test_fp += fp
        self.test_fn += fn
        self.test_tn += tn

        return None

    
    def on_test_epoch_end(self):
        # Calculate final AUC
        final_auc = self.test_auc.compute()

        # Avoid division by zero
        epsilon = 1e-6

        # Calculate metrics from the accumulated confusion matrix components
        accuracy = (self.test_tp + self.test_tn) / (self.test_tp + self.test_tn + self.test_fp + self.test_fn + epsilon)
        precision = self.test_tp / (self.test_tp + self.test_fp + epsilon)
        recall = self.test_tp / (self.test_tp + self.test_fn + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
      
        print("\n" + "="*50)
        print("Test Metrics Summary")
        print("="*50)
        print(f"Total True Positives (TP):  {self.test_tp}")
        print(f"Total False Positives (FP): {self.test_fp}")
        print(f"Total True Negatives (TN):  {self.test_tn}")
        print(f"Total False Negatives (FN): {self.test_fn}")
        print("-" * 50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1_score:.4f}")
        print(f"AUC:       {final_auc:.4f}")
        print("="*50 + "\n")

        # It's good practice to reset the metric after computing
        self.test_tp = self.test_fp = self.test_fn = self.test_tn = 0
        self.test_auc.reset()




if __name__ == "__main__":
    initialize(config_path="../configs", version_base="1.3")

    from torchinfo import summary
    from collections import Counter
    cfg = compose(
        config_name="config",
    )
    video= torch.randn((2,25, 1, 56, 56))
    audio=torch.randn((2,16000,1))

    model = lip_sync_stream(cfg,
                                         debug=True,
                                         unfreezed_conformers=2,
                                         gated_fusion=False,
                                         feature_add=True,
                                         attention_pooling=False
                                         )
    features=model.forward(video, audio)
    print("output shape", features.shape)


    summary(model, input_size=[video.shape,audio.shape])

    # print(model.feature_extractor.model.aux_encoder.encoders[0])

    print("fusion ",model.feature_extractor.model.fusion)

    # len=0
    # for encoder_layer in model.feature_extractor.model.aux_encoder.encoders:
    #     len=len+1
    # print("len of audio encoder layers",len)    
