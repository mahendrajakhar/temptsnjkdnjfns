import os
import torch
import imageio
import gradio as gr
from huggingface_hub import hf_hub_download
import numpy as np

# -------------------------------
# Step 1: Download Required Model Files
# -------------------------------
REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",  # Prefer using safetensors for security.
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
]

MODEL_DIR = "./flan_t5"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model-00001-of-00002.safetensors")

def download_model_files():
    """Download required model files from the Hugging Face Hub if not present."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    print("Checking for required model files...")
    for filename in REQUIRED_FILES:
        file_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(file_path):
            print(f"{filename} not found locally. Downloading...")
            hf_hub_download(repo_id="google/flan-t5-xl", filename=filename, local_dir=MODEL_DIR)
            print(f"Downloaded {filename}.")
        else:
            print(f"{filename} is already present.")
    print("All required files are available.")

# -------------------------------
# Step 2: Import and Load the Goku Model
# -------------------------------
# IMPORTANT: This assumes that the repository defines a GokuModel class that handles
# realistic video generation. The model is expected to have a method (here called
# generate_frames) that takes a text prompt and returns a list of video frames (as numpy arrays).
try:
    from goku.model import GokuModel
except ImportError:
    raise ImportError("GokuModel not found in goku/model.py. Please ensure the model is properly implemented.")

def load_goku_model():
    """Initialize the Goku model and load pre-trained weights."""
    print("Initializing Goku model...")
    model = GokuModel()  # Instantiate the model (should include architecture definition)
    # Load weights from the downloaded file(s).
    model.load_weights(MODEL_DIR)
    model.eval()  # Set to evaluation mode
    return model

# -------------------------------
# Step 3: Generate Video Using the Goku Model
# -------------------------------
def generate_video_from_text(prompt: str, num_frames: int = 16, fps: int = 4) -> str:
    """
    Generate a realistic video from a text prompt using the Goku AI model.
    
    Args:
        prompt (str): The text prompt describing the video content.
        num_frames (int): Number of frames to generate.
        fps (int): Frames per second for the output video.
    
    Returns:
        str: Path to the generated video file.
    """
    # Ensure that model files are present
    download_model_files()
    
    # Load the model
    model = load_goku_model()

    print(f"Generating {num_frames} frames for prompt: '{prompt}'")
    # The generate_frames method is assumed to implement the diffusion process or
    # autoregressive video generation pipeline that produces a list of frames.
    with torch.no_grad():
        # This call should return a list of numpy arrays (H x W x C)
        frames = model.generate_frames(prompt, num_frames=num_frames)
    
    # Verify that frames are generated
    if not frames or not isinstance(frames, list):
        raise ValueError("No frames were generated. Check the model implementation.")

    # Compile frames into a video file (MP4)
    video_path = "generated_video.mp4"
    print("Compiling frames into video...")
    # Specify codec for realistic video encoding (libx264 is common)
    imageio.mimwrite(video_path, frames, fps=fps, codec="libx264")
    print(f"Video generated and saved to {video_path}")
    return video_path

# -------------------------------
# Step 4: Gradio UI
# -------------------------------
def generate_video(prompt: str):
    """
    Gradio-friendly function to generate a video from a text prompt.
    
    Returns:
        str: Path to the generated video file.
    """
    return generate_video_from_text(prompt)

iface = gr.Interface(
    fn=generate_video,
    inputs=gr.Textbox(lines=4, placeholder="Enter your detailed video prompt here..."),
    outputs=gr.Video(),
    title="Goku AI Video Generator",
    description="Generate realistic video content using the Goku AI model. Provide a detailed text prompt and let the model create a video."
)

if __name__ == "__main__":
    iface.launch(share=True)
