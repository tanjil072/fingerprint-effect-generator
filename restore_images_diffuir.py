"""
Restore images using DiffUIR model.
"""
import os
from PIL import Image
import torch
from diffusers import DiffusionPipeline

def restore_image_with_diffuir(input_path, output_path, pipe):
    image = Image.open(input_path).convert("RGB")
    # Use a generic restoration prompt
    restored = pipe(prompt="restore this image", image=image).images[0]
    restored.save(output_path)

def main():
    processed_dir = "output/processed_images"
    restored_dir = "output/restored_images"
    os.makedirs(restored_dir, exist_ok=True)

    # Load Stable Diffusion XL Refiner pipeline (public model)
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float32)
    pipe = pipe.to("cpu")  # Use "cuda" if you have a GPU

    for fname in os.listdir(processed_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(processed_dir, fname)
            output_path = os.path.join(restored_dir, fname)
            print(f"Restoring {fname}...")
            restore_image_with_diffuir(input_path, output_path, pipe)
            print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
