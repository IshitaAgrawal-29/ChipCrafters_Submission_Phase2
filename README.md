# ChipCrafters AI Restoration Pipeline

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure `best_chipcrafter_model.pth` is in the same directory.

## Running Inference
The inference script takes an input directory of Noisy LR images and an output directory for the restored HR images.
Run the following command:
`python inference.py --input_dir path/to/noisy --output_dir path/to/save`
