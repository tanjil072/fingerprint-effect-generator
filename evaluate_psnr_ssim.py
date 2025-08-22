import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
import numpy as np

input_dir = 'input_images'
output_dir = 'output/restored_images'

results = []

for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    if os.path.exists(output_path):
        img1 = imread(input_path)
        img2 = imread(output_path)
        # Resize restored image to match input image shape
        if img2.shape != img1.shape:
            from skimage.transform import resize
            img2 = resize(img2, img1.shape, preserve_range=True, anti_aliasing=True).astype(img1.dtype)
        # Convert to grayscale if needed
        if img1.ndim == 3:
            img1 = np.mean(img1, axis=2)
        if img2.ndim == 3:
            img2 = np.mean(img2, axis=2)
        # Determine data_range for PSNR/SSIM
        if img1.dtype == np.uint8:
            data_range = 255
        else:
            data_range = img1.max() - img1.min()
        psnr = peak_signal_noise_ratio(img1, img2, data_range=data_range)
        ssim = structural_similarity(img1, img2, data_range=data_range)
        results.append((filename, psnr, ssim))


# Save results to a text file

# Compute averages
if results:
    avg_psnr = sum([psnr for _, psnr, _ in results]) / len(results)
    avg_ssim = sum([ssim for _, _, ssim in results]) / len(results)
else:
    avg_psnr = 0
    avg_ssim = 0

with open('psnr_ssim_results.txt', 'w') as f:
    for filename, psnr, ssim in results:
        line = f"{filename}: PSNR={psnr:.2f}, SSIM={ssim:.4f}\n"
        f.write(line)
        print(line.strip())
    avg_line = f"Average PSNR for dataset: {avg_psnr:.2f}\nAverage SSIM for dataset: {avg_ssim:.4f}\n"
    f.write(avg_line)
    print(avg_line.strip())
