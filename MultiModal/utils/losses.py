import torch.nn as nn
import torch.nn.functional as F
import torch

class ComputeLoss(nn.Module):
    def __init__(self,pretrain=True,center_loss=False):
        super().__init__()
        self.pretrain = pretrain
        if self.pretrain:
            self.loss=nn.MSELoss()
        else:
            self.loss=nn.BCEWithLogitsLoss()
            # self.loss2=nn.centre_loss.CenterLoss(num_classes=4, feat_dim=1280, use_gpu=True)

    def forward(self, spatial_features=None, temporal_features=None, teacher_output=None,
                logits=None, labels=None):
        
        """args:
        spatial_features:  (batch_size*T, num_spatial_features, feature_dim)
        temporal_features: (batch_size*T,1, feature_dim)
        teacher_output:(batch_size*T, num_features+1, feature_dim)"""
        if self.pretrain:
            assert teacher_output is not None, "teacher_output must be provided in pretrain mode"
            assert spatial_features is not None and temporal_features is not None

            loss = self.loss(spatial_features, teacher_output[:, 1:, :]) + \
                   self.loss(temporal_features, teacher_output[:, 0:1, :])
        else:
            assert logits is not None and labels is not None
            loss = self.loss(logits, labels)

        return loss
    

if __name__=="__main__":
    loss_fn = ComputeLoss(pretrain=True)
    # Example usage
    spatial_features = torch.randn(10,500, 1280)
    temporal_features = torch.randn(10, 1, 1280)
    teacher_output = torch.randn(10, 501, 1280)

    loss = loss_fn(spatial_features=spatial_features, temporal_features=temporal_features, teacher_output=teacher_output)
    print("Loss:", loss.item())