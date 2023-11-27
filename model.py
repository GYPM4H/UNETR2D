import torch
import torch.nn as nn

import logging

class LoggerControl:
    def __init__(self, name=__name__):
        self.logger = logging.getLogger(name)

    def enable_logging(self, level=logging.INFO):
        logging.basicConfig(level=level)
        self.logger.setLevel(level)

    def disable_logging(self):
        logging.disable(logging.CRITICAL)
      
    def info(self, message):
        self.logger.info(message)


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

# Decoder blocks
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

class UNETR2D(nn.Module):
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
        
        logger.info(f"z0 shape: {x.shape}")
        logger.info(f"z3 shape: {z3.shape}")
        logger.info(f"z6 shape: {z6.shape}")
        logger.info(f"z9 shape: {z9.shape}")
        logger.info(f"z12 shape: {z12.shape}")

        # 0 layer encoder out
        out0 = self.z0(x) # out 64c x H x W
        logger.info(f"out0 shape: {out0.shape}")

        # 3 layer encoder out
        out3 = self.z3(z3) # out 128c x H x W
        logger.info(f"out3 shape: {out3.shape}")

        # 6 layer encoder out
        out6 = self.z6(z6) # out 256c x H x W
        logger.info(f"out6 shape: {out6.shape}")

        # 9 layer encoder out
        out9 = self.z9(z9) # out 512c x H x W
        logger.info(f"out9 shape: {out9.shape}")

        # 12 layer encoder out
        out12 = self.z12(z12) # out 512c x H x W
        logger.info(f"out12 shape: {out12.shape}")

        #upsampling decoder + concats aka skip connections

        # 12 layer encoder out + 9 layer encoder out
        conc1 = torch.cat([out12, out9], dim=1)
        logger.info(f"concat1(9 + 12) + 6 layer encoder output: {conc1.shape}")
        c1 = self.c1(conc1)
        logger.info(f"after c1 + upsample blocks shape: {c1.shape}")

        # concat1(9 + 12) + 6 layer encoder out
        conc2 = torch.cat([c1, out6], dim=1)
        logger.info(f"concat2(6 + 9 + 12) + 3 layer encoder output: {conc2.shape}")
        c2 = self.c2(conc2)
        logger.info(f"after c2 + upsample blocks shape: {c2.shape}")

        # concat2(6 + 9 + 12) + 3 layer encoder out
        conc3 = torch.cat([c2, out3], dim=1)
        logger.info(f"concat3(3 + 6 + 9 + 12) + 0 layer encoder output: {conc3.shape}")
        c3 = self.c3(conc3)
        logger.info(f"after c3 + upsample blocks shape: {c3.shape}")

        # concat3(3 + 6 + 9 + 12) + 0 layer encoder out
        conc4 = torch.cat([c3, out0], dim=1)
        logger.info(f"concat4(0 + 3 + 6 + 9 + 12) + 0 layer encoder output: {conc4.shape}")
        c4 = self.c4(conc4)
        logger.info(f"after c4 + conv blocks shape: {c4.shape}")

        logger.info(f"final output shape: {c4.shape}")

        return c4

if __name__ == "__main__":
    logger = LoggerControl()
    logger.enable_logging()
    
    x = torch.randn(1, 3, 512, 512)
    unetr = UNETR2D(image_size=512, patch_size=16, num_classes=2)
    y = unetr(x)
