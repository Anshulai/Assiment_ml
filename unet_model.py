import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block used in UNet"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ColorEmbedding(nn.Module):
    """Embedding layer for color names"""
    def __init__(self, num_colors, embed_dim):
        super(ColorEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_colors, embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, color_indices):
        # color_indices: (batch_size,)
        embeddings = self.embedding(color_indices)  # (batch_size, embed_dim)
        return embeddings


class ConditionalUNet(nn.Module):
    """UNet model conditioned on color input"""
    def __init__(self, n_channels=3, n_classes=3, num_colors=8, bilinear=True):
        super(ConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Color embedding
        self.color_embed_dim = 64
        self.color_embedding = ColorEmbedding(num_colors, self.color_embed_dim)
        
        # UNet encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Color conditioning - inject color information at bottleneck
        self.color_projection = nn.Linear(self.color_embed_dim, 1024 // factor)
        
        # UNet decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x, color_indices):
        # x: (batch_size, channels, height, width)
        # color_indices: (batch_size,)
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Color conditioning at bottleneck
        color_embed = self.color_embedding(color_indices)  # (batch_size, embed_dim)
        color_features = self.color_projection(color_embed)  # (batch_size, 1024//factor)
        
        # Reshape color features to match spatial dimensions
        batch_size, channels = color_features.shape
        h, w = x5.shape[2], x5.shape[3]
        color_features = color_features.view(batch_size, channels, 1, 1)
        color_features = color_features.expand(batch_size, channels, h, w)
        
        # Add color conditioning to bottleneck features
        x5 = x5 + color_features
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return torch.sigmoid(logits)  # Output in [0, 1] range


def create_color_mapping():
    """Create mapping from color names to indices"""
    colors = ['blue', 'cyan', 'green', 'magenta', 'orange', 'purple', 'red', 'yellow']
    color_to_idx = {color: idx for idx, color in enumerate(colors)}
    idx_to_color = {idx: color for idx, color in enumerate(colors)}
    return color_to_idx, idx_to_color


if __name__ == "__main__":
    # Test the model
    model = ConditionalUNet(n_channels=3, n_classes=3, num_colors=8)
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    color_indices = torch.tensor([0, 1])  # blue, cyan
    
    output = model(x, color_indices)
    print(f"Input shape: {x.shape}")
    print(f"Color indices: {color_indices}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
