"""
Restore images using DiffUIR model.
"""
import os
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

def restore_image_with_inpainting(input_path, mask_path, output_path, pipe):
    image = Image.open(input_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    # Use a prompt for targeted inpainting
    restored = pipe(prompt="remove fingerprint", image=image, mask_image=mask).images[0]
    restored.save(output_path)

def main():
    processed_dir = "output/processed_images"
    mask_dir = "output/label_masks"
    restored_dir = "output/restored_images"
    os.makedirs(restored_dir, exist_ok=True)

    # Load Stable Diffusion Inpainting pipeline (smaller model)
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    for fname in os.listdir(processed_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(processed_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            output_path = os.path.join(restored_dir, fname)
            if not os.path.exists(mask_path):
                print(f"Mask not found for {fname}, skipping.")
                continue
            print(f"Restoring {fname} with mask...")
            restore_image_with_inpainting(input_path, mask_path, output_path, pipe)
            print(f"Saved to {output_path}")
            if device == "cuda":
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
