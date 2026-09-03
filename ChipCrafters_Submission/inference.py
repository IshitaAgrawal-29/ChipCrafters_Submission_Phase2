import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# [Insert NAFNet Architecture Classes Here - LayerNorm2d, SimpleGate, NAFBlock, NAFNetSR]
# (Note: For the actual submission, ensure the model classes are pasted into this file)

def restore_images(input_dir, output_dir, model_path="best_chipcrafter_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model (Assuming NAFNetSR is defined in this file)
    model = NAFNetSR(width=48).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            lr_img = np.load(os.path.join(input_dir, filename)).astype(np.float32)
            lr_t = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(lr_t)
                
            pred_img = pred.squeeze().cpu().numpy()
            np.save(os.path.join(output_dir, filename), pred_img)
            print(f"Restored: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing noisy .npy files")
    parser.add_argument("--output_dir", required=True, help="Directory to save restored .npy files")
    args = parser.parse_args()
    
    restore_images(args.input_dir, args.output_dir)
