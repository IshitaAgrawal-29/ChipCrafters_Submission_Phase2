import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. STANDALONE ARCHITECTURE (NAFNet-SR)
# ==============================================================================
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight * ((x - u) / torch.sqrt(s + self.eps)) + self.bias

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.conv2 = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.conv3 = nn.Conv2d(c, c, 1)
        self.conv4 = nn.Conv2d(c, c * 2, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(c, c, 1)
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.beta, self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1))), nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):
        x = self.conv3(self.sg1(self.conv2(self.conv1(self.norm1(inp)))) * self.sca(self.sg1(self.conv2(self.conv1(self.norm1(inp))))))
        y = inp + x * self.beta
        return y + self.conv5(self.sg2(self.conv4(self.norm2(y)))) * self.gamma

class NAFNetSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, width=48):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        curr_width = width

        for num_b in [2, 2, 4]:
            self.encoders.append(nn.Sequential(*[NAFBlock(curr_width) for _ in range(num_b)]))
            self.downs.append(nn.Conv2d(curr_width, curr_width * 2, 2, stride=2))
            curr_width *= 2

        self.middle = nn.Sequential(*[NAFBlock(curr_width) for _ in range(2)])

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for num_b in [2, 2, 2]:
            self.ups.append(nn.Sequential(
                nn.Conv2d(curr_width, curr_width * 2, 1),
                nn.PixelShuffle(2)
            ))
            curr_width = curr_width // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(curr_width) for _ in range(num_b)]))

        self.up_head = nn.Sequential(
            nn.Conv2d(width, width * 4, 3, padding=1),
            nn.PixelShuffle(2)
        )
        self.outro = nn.Conv2d(width, out_channels, 3, padding=1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        feat = self.intro(x)
        enc_feats = []

        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            enc_feats.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(enc_feats)):
            feat = dec(up(feat) + skip)

        residual = self.outro(self.up_head(feat))
        return torch.clamp(base + residual, 0.0, 1.0)

# ==============================================================================
# 2. INFERENCE PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Semiconductor Image Restoration Inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing noisy .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save restored .npy files")
    parser.add_argument("--model_path", type=str, default="best_chipcrafter_model.pth", help="Path to saved model weights")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # Load Model
    print("[*] Initializing NAFNet-SR Architecture...")
    model = NAFNetSR(width=48).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("[*] Weights loaded successfully.")

    # Process files
    files = [f for f in os.listdir(args.input_dir) if f.endswith('.npy')]
    print(f"[*] Found {len(files)} files to restore. Starting inference...")

    with torch.no_grad():
        for filename in files:
            filepath = os.path.join(args.input_dir, filename)
            
            # Load Noisy LR image
            lr_img = np.load(filepath).astype(np.float32)
            lr_tensor = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0).to(device)
            
            # Forward pass
            pred_tensor = model(lr_tensor)
            
            # Save Restored HR image
            pred_img = pred_tensor.squeeze().cpu().numpy()
            out_filepath = os.path.join(args.output_dir, filename)
            np.save(out_filepath, pred_img)
            
    print(f"[*] Inference complete! Restored files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
