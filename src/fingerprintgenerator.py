import random
import cv2
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from fingerprint import Fingerprint

"""
This script generates fingerprint effects on images
A completely new implementation for fingerprint simulation
Author: AI Assistant
"""

def generate_fingerprints(image_path, cfg):
    """
    This function generates fingerprint effects with random positions and patterns
    """
    
    max_prints = cfg["maxPrints"]
    min_prints = cfg["minPrints"] 
    print_num = random.randint(min_prints, max_prints)
    
    max_size = cfg["maxSize"]
    min_size = cfg["minSize"]
    max_intensity = cfg["maxIntensity"]
    min_intensity = cfg["minIntensity"]
    blur_radius = cfg["blur_radius"]
    return_label = cfg["return_label"]
    noise_level = cfg["noise_level"]
    brightness_factor = cfg.get("brightness_factor", 0.4)
    contrast_boost = cfg.get("contrast_boost", 1.3)
    
    # Load background image
    pil_bg_img = Image.open(image_path)
    bg_img = np.asarray(pil_bg_img)
    img_h, img_w = bg_img.shape[:2]
    
    # Generate random positions for fingerprints
    fingerprint_positions = []
    fingerprint_list = []
    
    # Create non-overlapping positions
    for i in range(print_num):
        attempts = 0
        while attempts < 50:  # Max attempts to find non-overlapping position
            size = random.randint(min_size, max_size)
            x = random.randint(size, img_w - size)
            y = random.randint(size, img_h - size)
            
            # Check for overlap with existing fingerprints
            overlap = False
            for existing_pos, existing_size in fingerprint_positions:
                distance = math.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                if distance < (size + existing_size) * 0.7:  # Allow some overlap
                    overlap = True
                    break
            
            if not overlap:
                fingerprint_positions.append(((x, y), size))
                break
            attempts += 1
        
        if attempts < 50:  # Only add if we found a good position
            intensity = random.uniform(min_intensity, max_intensity)
            pattern_type = random.choice(['loop', 'whorl', 'arch'])
            fingerprint = Fingerprint(i + 1, (x, y), size, pattern_type, intensity)
            fingerprint_list.append(fingerprint)
    
    # Create combined alpha map for all fingerprints
    combined_alpha = np.zeros((img_h, img_w), dtype=np.float64)
    
    # Process each fingerprint
    for fingerprint in fingerprint_list:
        center_x, center_y = fingerprint.get_center()
        size = fingerprint.get_size()
        
        # Calculate region of interest
        roi_size = size * 2
        roi_x1 = max(0, center_x - size)
        roi_y1 = max(0, center_y - size)
        roi_x2 = min(img_w, center_x + size)
        roi_y2 = min(img_h, center_y + size)
        
        # Extract background region
        bg_region = bg_img[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if bg_region.size > 0:
            # Update fingerprint texture with background
            fingerprint.update_texture(bg_region)
            
            # Get fingerprint alpha map
            fp_alpha = fingerprint.get_alpha_map()
            
            # Calculate actual region to apply (considering image boundaries)
            fp_h, fp_w = fp_alpha.shape
            
            # Calculate offsets
            bg_h, bg_w = bg_region.shape[:2]
            
            # Resize fingerprint alpha to match background region if needed
            if fp_h != bg_h or fp_w != bg_w:
                fp_alpha_resized = cv2.resize(fp_alpha, (bg_w, bg_h))
            else:
                fp_alpha_resized = fp_alpha
            
            # Add to combined alpha map
            combined_alpha[roi_y1:roi_y2, roi_x1:roi_x2] += fp_alpha_resized
    
    # Normalize combined alpha map
    if np.max(combined_alpha) > 0:
        combined_alpha = np.clip(combined_alpha / np.max(combined_alpha) * 255, 0, 255)
    
    # Apply fingerprint effects to the image
    result_img = pil_bg_img.copy()
    
    for fingerprint in fingerprint_list:
        center_x, center_y = fingerprint.get_center()
        size = fingerprint.get_size()
        
        # Calculate region of interest
        roi_x1 = max(0, center_x - size)
        roi_y1 = max(0, center_y - size)
        roi_x2 = min(img_w, center_x + size)
        roi_y2 = min(img_h, center_y + size)
        
        # Get fingerprint texture
        fp_texture = fingerprint.get_texture()
        
        if fp_texture is not None:
            # Resize texture to match ROI if needed
            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1
            
            if roi_w > 0 and roi_h > 0:
                fp_texture_resized = fp_texture.resize((roi_w, roi_h), Image.LANCZOS)
                
                # Extract alpha channel for proper blending
                if fp_texture_resized.mode == 'RGBA':
                    alpha_channel = fp_texture_resized.split()[-1]
                    alpha_array = np.array(alpha_channel).astype(np.float32) / 255.0
                    
                    # Apply additional feathering to edges for smoother blending
                    alpha_array = gaussian_filter(alpha_array, sigma=0.8)
                    
                    # Create smooth alpha mask
                    alpha_smooth = Image.fromarray((alpha_array * 255).astype(np.uint8), 'L')
                    
                    # Create enhanced fingerprint effect
                    enhanced = ImageEnhance.Contrast(fp_texture_resized)
                    enhanced = enhanced.enhance(contrast_boost)
                    
                    # Apply brightness reduction for smudge effect
                    brightness_enhancer = ImageEnhance.Brightness(enhanced)
                    enhanced = brightness_enhancer.enhance(brightness_factor)
                    
                    # Apply minimal blur to maintain detail
                    if blur_radius > 0:
                        enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=blur_radius * 0.5))
                    
                    # Create subtle dark overlay for realism
                    overlay_strength = int(60 * fingerprint.get_intensity())
                    dark_overlay = Image.new('RGBA', (roi_w, roi_h), 
                                           (40, 40, 40, overlay_strength))
                    
                    # Blend overlay with fingerprint
                    enhanced = Image.alpha_composite(enhanced.convert('RGBA'), dark_overlay)
                    
                    # Use the smooth alpha for natural blending
                    enhanced.putalpha(alpha_smooth)
                    
                    # Paste with smooth blending
                    try:
                        result_img.paste(enhanced, (roi_x1, roi_y1), enhanced)
                    except Exception as e:
                        print(f"Warning: Could not paste fingerprint {fingerprint.get_key()}: {e}")
                else:
                    # Fallback for non-RGBA textures
                    try:
                        result_img.paste(fp_texture_resized, (roi_x1, roi_y1))
                    except Exception as e:
                        print(f"Warning: Could not paste fingerprint {fingerprint.get_key()}: {e}")
    
    # Add subtle noise for realism
    if noise_level > 0:
        result_array = np.asarray(result_img).astype(np.float32)
        noise = np.random.normal(0, noise_level * 10, result_array.shape)
        result_array = np.clip(result_array + noise, 0, 255)
        result_img = Image.fromarray(result_array.astype(np.uint8))
    
    # Return results
    if return_label:
        # Create binary label map
        label_map = np.zeros((img_h, img_w), dtype=np.uint8)
        label_map[combined_alpha > 10] = 1  # Threshold for binary mask
        label_img = Image.fromarray(label_map * 255)  # Scale to 0-255 for visibility
        
        return result_img, label_img
    
    return result_img
