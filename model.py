import torch
import torch.nn as nn

import copy

# Image patching + linear proj
# 256 x 256 x 3 image
# num_patches = (256 / 16) ** 2 = 256
# 256 x 256 x 3 -> 256 x 256 x 768

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2

        self.flatten_patch_size = patch_size * patch_size * in_channels
        self.proj = nn.Linear(self.flatten_patch_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))


    def forward(self, x):
        # x: [batch_size, in_channels, img_size, img_size]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.n_patches, self.flatten_patch_size)
        x = self.proj(x) + self.positional_encoding
        return x
    
#Transformer Block
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, src):
        outputs = []
        output = src
        for layer in self.layers:
            output = layer(output)
            outputs.append(output)
        return outputs

# UNETREncoder
class UNETREncoder(nn.Module):
    def __init__(self, 
                 image_size=256, 
                 patch_size=16, 
                 in_channels=3, 
                 embed_dim=768, 
                 num_heads=12, 
                 num_layers=12, 
                 dropout_rate=0.1,
                 extract_layers=[3, 6, 9, 12]):
        super().__init__()
        self.extract_layers = extract_layers
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers, dropout_rate)

    def forward(self, x):
        x = self.patch_embedding(x)
        layer_outputs = self.transformer(x)
        return [layer_outputs[i - 1] for i in self.extract_layers]

# Decoder 

class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.up(x)
        return x  
    
class YellowBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())     

    def forward(self, x):
        x = self.block(x)
        return x
    
class BlueBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.green = GreenBlock(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.yellow = YellowBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.green(x)
        x = self.yellow(x)
        return x
    
class GreyBlock(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size=1, stride=1):
        super().__init__()
        self.block = nn.Conv2d(in_channels, num_classes, kernel_size, stride)
        
    def forward(self, x):
        x = self.block(x)
        return x
    

class UNETR(nn.Module):
    def __init__(self, 
                 image_size=256, 
                 patch_size=16, 
                 in_channels=3, 
                 embed_dim=768, 
                 num_heads=12, 
                 num_layers=12, 
                 dropout_rate=0.1,
                 extract_layers=[3, 6, 9, 12],
                 num_classes=2):
        super().__init__()
        self.encoder = UNETREncoder(image_size, patch_size, in_channels, embed_dim, num_heads, num_layers, dropout_rate, extract_layers)
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        #top yellow block 
        self.z0 = nn.Sequential(
            YellowBlock(in_channels, 64),
            YellowBlock(64, 64)
        )
        
        #blue blocks
        self.z3 = nn.Sequential(
            BlueBlock(768, 512),
            BlueBlock(512, 256),
            BlueBlock(256, 128),
        )

        self.z6 = nn.Sequential(
            BlueBlock(768, 512),
            BlueBlock(512, 256),
        )

        self.z9 = nn.Sequential(
            BlueBlock(768, 512),
        )

        self.z12 = GreenBlock(768, 512)

        #bottom part of yellow + green upsampling with concats
        self.c1 = nn.Sequential(
            YellowBlock(512 + 512, 512),
            YellowBlock(512, 512),
            GreenBlock(512, 256)
        )

        self.c2 = nn.Sequential(
            YellowBlock(256 + 256, 256),
            YellowBlock(256, 256),
            GreenBlock(256, 128)
        )

        self.c3 = nn.Sequential(
            YellowBlock(128 + 128, 128),
            YellowBlock(128, 128),
            GreenBlock(128, 64)
        )

        self.c4 = nn.Sequential(
            YellowBlock(64 + 64, 64),
            YellowBlock(64, 64),
            GreyBlock(64, num_classes))
        

    def forward(self, x):
        encoder_out= self.encoder(x)
        # reshape embeded outputs of (batch_size, n_patches, embed_dim) to (batch_size, num_channels, H, W)
        z3, z6, z9, z12 = \
            [z.reshape(-1, 768, self.image_size // self.patch_size, self.image_size // self.patch_size) for z in encoder_out]
        
        print("Encoder output z3", z3.shape)
        print("Encoder output z6", z6.shape)
        print("Encoder output z9", z9.shape)
        print("Encoder output z12", z12.shape)

        # 0 layer encoder out
        out0 = self.z0(x) # out 64c x H x W
        print("0 layer blue conv output", out0.shape)

        # 3 layer encoder out
        out3 = self.z3(z3) # out 128c x H x W
        print("3 layer blue conv output", out3.shape)

        # 6 layer encoder out
        out6 = self.z6(z6) # out 256c x H x W
        print("6 layer blue conv output", out6.shape)

        # 9 layer encoder out
        out9 = self.z9(z9) # out 512c x H x W
        print("9 layer blue conv output", out9.shape)

        # 12 layer encoder out
        out12 = self.z12(z12) # out 512c x H x W
        print("12 layer green conv output", out12.shape)

        #upsampling decoder + concats aka skip connections

        # 12 layer encoder out + 9 layer encoder out
        print("concat1(9 + 12)", torch.cat([out12, out9], dim=1).shape)
        c1 = self.c1(torch.cat([out12, out9], dim=1))

        # concat1(9 + 12) + 6 layer encoder out
        print("concat2(9 + 12) + 6 layer encoder output", torch.cat([c1, out6], dim=1).shape)
        c2 = self.c2(torch.cat([c1, out6], dim=1))

        # concat2(6 + 9 + 12) + 3 layer encoder out
        print("concat3(6 + 9 + 12) + 3 layer encoder output", torch.cat([c2, out3], dim=1).shape)
        c3 = self.c3(torch.cat([c2, out3], dim=1))

        # concat3(3 + 6 + 9 + 12) + 0 layer encoder out
        print("concat4(3 + 6 + 9 + 12) + 0 layer encoder output", torch.cat([c3, out0], dim=1).shape)
        c4 = self.c4(torch.cat([c3, out0], dim=1))

        print("Final output", c4.shape)

        return c4

