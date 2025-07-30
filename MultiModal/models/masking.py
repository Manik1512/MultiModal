import torch.nn as nn
import torch
import torchvision.transforms as T
import timm
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import random
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
class Masking:
    def __init__(self,model_name,device,masking_ratio):
        self.model = timm.create_model(model_name, pretrained=True).eval().to(device)
        self.transform=T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
        self.transform_no_norm=T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
        self.device = device
        self.qk_store = {}
        self.token_store = {}
        self.attn = self.model.blocks[-1].attn
        self.num_heads = self.attn.num_heads
        self.head_dim = self.attn.head_dim

        self.register_hooks(self.model)
        self.masking_ratio = masking_ratio

    def hook_qkv(self,module, input, output):
    
        B, N, _ = output.shape
        qkv = output.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        self.qk_store['q'] = qkv[0]  # (B, heads, N, head_dim)
        self.qk_store['k'] = qkv[1]  # (B, heads, N, head_dim)


    def hook_tokens(self,module, input, output):
        self.token_store['tokens'] = input[0]  # shape: (B, N, D)


    def register_hooks(self,model):
        self.attn.qkv.register_forward_hook(self.hook_qkv)
        last_block = model.blocks[-1]
        last_block.register_forward_hook(self.hook_tokens)


    def load_image(self, df, path_col="img_path",norm=True):
        image_tensors = []
        for idx in range(0, df.shape[0]):
            path = df[path_col][idx]
            image = Image.open(path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0) if norm else self.transform_no_norm(image).unsqueeze(0)
            image_tensors.append(image_tensor)
        
        batch_tensor = torch.cat(image_tensors, dim=0)  # shape: [B, C, H, W]
        return batch_tensor.to(self.device)

        
    def predict(self,df):
        image_tensor = self.load_image(df)
        with torch.no_grad():
            output = self.model(image_tensor)

        tokens = self.token_store['tokens']
        q = self.qk_store['q']  # (B, heads, N, head_dim)
        k = self.qk_store['k']  # (B, heads, N, head_dim)

        return {""
        "cls_token":output,
        "ouput_patches":self.token_store['tokens'],
        'q':q,
        'k':k}


    def returnn_attention_probs(self,df):
        out=self.predict(df)
        for  key in out:
            print(f"{key}: {out[key].shape if isinstance(out[key], torch.Tensor) else out[key]}")
 
        k= out["k"][:,:,1:,:]  # (heads, N, head_dim)
        q_cls = out["q"][:, :, 0:1, :]  # (B, heads, 1, head_dim)
        attn_scores = torch.matmul(q_cls, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1).squeeze(2)  # (B, heads, N)
        attn_probs = attn_probs.mean(1)# (N,)
        print(attn_probs.shape)
        return attn_probs
    

    def mask_batch(self,patches):
        """
        Args:
            df: DataFrame with column 'img_path'
        
        Returns:
            masked_images: Tensor of shape [B, 3, 224, 224] with low-attention patches masked out
        """
        # Step 1: Get attention probabilities for the batch
        attn_probs = self.returnn_attention_probs(df)  # shape: [B, N]
        B, N = attn_probs.shape
        grid_size = int(N ** 0.5)  # its H/patch size
        num_keep = int(N * (1 - self.masking_ratio))

        topk_indices = torch.topk(attn_probs, num_keep, dim=1).indices  # [B, num_keep]

        # Step 3: Create binary mask: True = keep, False = mask
        mask = torch.zeros((B, N), dtype=torch.bool, device=attn_probs.device)
        batch_indices = torch.arange(B, device=attn_probs.device).unsqueeze(1)
        mask[batch_indices, topk_indices] = True
        mask = mask.view(B, grid_size, grid_size)  # [B, G, G]

        mask=mask.unsqueeze(1)  # [B, 1, G, G]
        print("mask shape:",mask.shape)
        print("patches shape:",patches.shape)
        mask_images= patches * mask.float()  # [B, 96, G, G]
        print("mask_images shape:",mask_images.shape)
       
        return mask_images

    def visualize_attention_mask(self,images,mask_images,channels=16):
        """randomly select image from a batch and visualize the attention mask"""
        
        idx=random.randint(0, df["img_path"].__len__()-1)
        # idx=1
        plt.figure(figsize=(20, 20))
        plt.subplot(1, 3, 1)
        plt.imshow(images[idx].cpu().permute(1, 2, 0).numpy())
        plt.title("Original")

        one_channel=random.randint(0,mask_images.shape[1]-1)
        plt.subplot(1, 3, 2)
        mask = mask_images.detach()[idx]
        plt.imshow(mask[one_channel].cpu().numpy(), cmap="magma")  # visualize 1st channel
        plt.title(f"Mask Channel {one_channel} ({(1 - self.masking_ratio)*100:.0f}% kept)")

        grid = make_grid(mask[0:channels,:,:].unsqueeze(1), nrow=int(channels**0.5), normalize=True, pad_value=1)  # (3, H, W)
        plt.subplot(1, 3, 3)
        plt.imshow(TF.to_pil_image(grid), cmap="magma")
        plt.title(f"first {channels} channels of Masked Patches")
        plt.show()


if __name__ == "__main__":

    mask=Masking("vit_base_patch8_224.dino",device="cuda",masking_ratio=0.65)
    img_path1= "/home/manik/Downloads/Screenshot from 2025-07-27 15-46-20.png"
    img_path2= "/home/manik/Downloads/Screenshot from 2025-07-28 21-36-51.png"
    img_path3= "/home/manik/Downloads/Screenshot from 2025-07-24 13-48-51.png"
    img_path4= "/home/manik/Downloads/Screenshot from 2025-07-21 14-20-33.png"
    img_path5= "/home/manik/Downloads/Screenshot from 2025-07-30 15-39-40.png"

    df=pd.DataFrame(
        {
            "img_path":[img_path1,img_path2,img_path3,img_path4,img_path5],
        }
    )


    image_tensor = mask.load_image(df, norm=True)  # shape: [B, 3, 224, 224]
    image_tensor_vis = mask.load_image(df, norm=False)  # shape: [B, 3, 224, 224]

    torch.manual_seed(42)  # for reproducibility
    conv2d=nn.Conv2d(3,96,stride=8,padding=0,kernel_size=8).to(mask.device)
    patches=conv2d(image_tensor)  # shape: [B, 96, G, G]

    mask_images=mask.mask_batch(patches)  # shape: [B, 96, G, G]
    mask.visualize_attention_mask(image_tensor_vis, mask_images, channels=16)



    