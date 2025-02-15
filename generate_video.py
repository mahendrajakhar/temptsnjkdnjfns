import os
import time
import random
import numpy as np
from huggingface_hub import hf_hub_download
import gradio as gr
import imageio
from PIL import Image, ImageDraw, ImageFont

# -------------------------------
# Step 1: Download Model Files
# -------------------------------
REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",  # Recommend using *.safetensors for safety.
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
]

MODEL_DIR = "./flan_t5"

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
# Step 2: Import (or simulate) Goku Model Components
# -------------------------------
try:
    from goku.model import Block, AdaLayerNorm, FeedForward, RMSNorm, Attention, apply_rotary_emb
except ImportError:
    print("Goku model components not found – proceeding with dummy implementations.")
    # For this demo, these components aren't used directly.

# -------------------------------
# Step 3: Define the Goku Video Generator
# -------------------------------
class GokuVideoGenerator:
    def __init__(self):
        # In a real implementation, initialize the full network architecture here.
        print("Initialized GokuVideoGenerator (dummy demo version).")
        self.num_frames = 10           # Number of frames in the video
        self.frame_size = (256, 256)   # Frame dimensions (width, height)
    
    def load_weights(self, model_dir):
        """Simulate loading model weights from the given directory."""
        weights_path = os.path.join(model_dir, "model-00001-of-00002.safetensors")
        if os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}...")
            # In practice, load the weights (e.g., using torch.load or safetensors).
            print("Weights loaded successfully.")
        else:
            print("Weights file not found in", model_dir)
    
    def generate_video_from_text(self, text_prompt):
        """
        Generate a dummy video from a text prompt.
        In a real system, you would use a text encoder and a diffusion
        or transformer-based network to generate video frames.
        Here we simulate this by generating images with the prompt drawn on them.
        """
        print(f"Encoding text prompt: '{text_prompt}'")
        seed = abs(hash(text_prompt)) % (10**8)
        random.seed(seed)
        np.random.seed(seed)
        
        print("Generating video frames...")
        frames = []
        for i in range(self.num_frames):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = Image.new("RGB", self.frame_size, color)
            draw = ImageDraw.Draw(img)
            text = f"Frame {i+1}\n{text_prompt}"
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                font = ImageFont.load_default()
            draw.text((10, 10), text, fill=(255, 255, 255), font=font)
            frames.append(np.array(img))
            time.sleep(0.1)  # Simulate processing delay
        
        video_path = "generated_video.mp4"
        print("Compiling frames into video...")
        # Removed 'quality' argument to avoid the error.
        imageio.mimwrite(video_path, frames, fps=2)
        print(f"Video generated and saved to {video_path}")
        return video_path

# -------------------------------
# Step 4: Video Generation Function
# -------------------------------
def generate_video(text_prompt: str):
    """
    Full pipeline: checks model files, loads the Goku video generator,
    and generates a video from the provided text prompt.
    """
    download_model_files()
    generator = GokuVideoGenerator()
    generator.load_weights(MODEL_DIR)
    video_path = generator.generate_video_from_text(text_prompt)
    return video_path

# -------------------------------
# Step 5: Create Gradio UI and Sample Input
# -------------------------------
iface = gr.Interface(
    fn=generate_video,
    inputs=gr.Textbox(lines=2, placeholder="Enter your video prompt here..."),
    outputs=gr.Video(),
    title="Goku AI Video Generator",
    description="Enter a text prompt to generate a video using the Goku open source model (demo version)."
)

def sample_run():
    sample_prompt = "A futuristic city skyline at dusk with vibrant neon lights."
    print(f"Running sample input: {sample_prompt}")
    generated_video = generate_video(sample_prompt)
    print(f"Sample video generated at: {generated_video}")

if __name__ == "__main__":
    # Uncomment the following line to run a sample input from the terminal.
    # sample_run()
    
    # Launch the Gradio UI with sharing enabled.
    iface.launch(share=True)
