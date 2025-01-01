from transparent_background import Remover
import torchvision.transforms.v2 as T
from PIL import Image
import torch
import numpy as np

class TransparentBackgroundRembg:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "model_path": ("STRING", {"default": "/kaggle/working/latest.pth", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("Foreground", "Mask", "Foreground_Transparent")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image: torch.Tensor, model_path: str):
        remover = Remover(mode='base-nightly', ckpt=model_path, device='cuda:0')
        bgrgba = None
        
        image = image.permute([0, 3, 1, 2])
        output = []
        for img in image:
            img = T.ToPILImage()(img)
            img = remover.process(img, type='rgba')
            output.append(T.ToTensor()(img))
            
        output = torch.stack(output, dim=0)
        output = output.permute([0, 2, 3, 1])
#         mask = output[:, :, :, 3] if output.shape[3] == 4 else torch.ones_like(output[:, :, :, 0])

#         if output.shape[3] == 4:
#             print("Has 4 channels")
#             # Extract the RGB channels and the alpha channel separately
#             output_rgb = output[:, :, :, :3]
#             mask = output[:, :, :, 3]
#         else:
#             print("Not having 4 channels")
#             print("Has: {}".format(output.shape[3]))
#             # If there's no alpha channel, just use the output as-is
#             output_rgb = output
#             mask = torch.ones_like(output[:, :, :, 0])  # Create a mask of ones
#         return(output_rgb, mask,)
        

        if output.shape[3] == 4:
            print("Has 4 channels")
            alpha_channel = output[:, :, :, 3]
            mask = alpha_channel
            output_rgb = output[:, :, :, :3]
#             output_rgb[~mask] = 0  # Set the background to black where alpha was 0
        else:
            print("Not having 4 channels")
            print("Has: {}".format(output.shape[3]))
            output_rgb = output
            mask = torch.ones_like(output[:, :, :, 0])  # Create a mask of ones

        return output_rgb, mask, output

NODE_CLASS_MAPPINGS = {
    "Transparentbackground RemBg": TransparentBackgroundRembg
}
