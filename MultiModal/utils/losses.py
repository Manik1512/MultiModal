import torch.nn as nn
import torch.nn.functional as F
import torch


class Pretraining_loss(nn.Module):
    def __init__(self,spatial=False):
        super().__init__()
        self.spatial=spatial
        self.loss=nn.MSELoss()
      

    def forward(self, spatial_features=None, temporal_features=None, teacher_cls_token=None,teacher_unmasked_patches=None):
        
        """args:
        spatial_features:  (batch_size*T, unmasked_patches, feature_dim)
        teacher_cls_token: (batch_size*T,1, feature_dim)
        teacher_unmasked_patches:(batch_size*T, unmasked_patches, feature_dim)"""
        if self.spatial:
            loss = self.loss(spatial_features, teacher_cls_token) + \
                self.loss(temporal_features, teacher_unmasked_patches)
            
        else:
            loss=self.loss(temporal_features, teacher_unmasked_patches)
        return loss
    

if __name__=="__main__":
    loss_fn = Pretraining_loss(pretrain=True)
    # Example usage
    spatial_features = torch.randn(10,500, 1280)
    temporal_features = torch.randn(10, 1, 1280)
    teacher_output = torch.randn(10, 501, 1280)

    loss = loss_fn(spatial_features=spatial_features, temporal_features=temporal_features, teacher_output=teacher_output)
    print("Loss:", loss.item())