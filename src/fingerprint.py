import cv2
import math
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import random
from scipy import ndimage

class Fingerprint:
    def __init__(self, key, center_xy=None, size=None, pattern_type=None, intensity=None):
        self.key = key
        self.center = center_xy
        self.size = size
        self.pattern_type = pattern_type or random.choice(['loop', 'whorl', 'arch'])
        self.intensity = intensity or random.uniform(0.2, 0.6)
        self.rotation = random.uniform(0, 360)  # Random rotation
        
        # Create the fingerprint pattern
        self.alpha_map = np.zeros((self.size * 2, self.size * 2))
        self.pattern_map = np.zeros((self.size * 2, self.size * 2))
        self.texture = None
        
        self._create_fingerprint_pattern()
    
    def _create_fingerprint_pattern(self):
        """Create the fingerprint ridge pattern"""
        h, w = self.alpha_map.shape
        center_x, center_y = w // 2, h // 2
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        
        # Distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Angle from center
        angle = np.arctan2(y - center_y, x - center_x)
        
        if self.pattern_type == 'loop':
            self._create_loop_pattern(distance, angle, center_x, center_y)
        elif self.pattern_type == 'whorl':
            self._create_whorl_pattern(distance, angle, center_x, center_y)
        elif self.pattern_type == 'arch':
            self._create_arch_pattern(distance, angle, center_x, center_y)
        
        # Apply rotation
        if self.rotation != 0:
            self.alpha_map = ndimage.rotate(self.alpha_map, self.rotation, reshape=False)
            self.pattern_map = ndimage.rotate(self.pattern_map, self.rotation, reshape=False)
        
        # Create natural oval-shaped fingerprint boundary instead of rectangular
        h, w = self.alpha_map.shape
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        
        # Create elliptical mask for natural fingerprint shape
        # Make it slightly oval like real fingerprints
        semi_major = self.size * 0.9  # Length
        semi_minor = self.size * 0.7  # Width
        
        # Elliptical distance calculation
        ellipse_distance = ((x - center_x)**2 / semi_major**2 + 
                           (y - center_y)**2 / semi_minor**2)
        
        # Create smooth falloff mask
        oval_mask = np.where(ellipse_distance <= 1.0, 1.0, 0.0)
        
        # Add smooth edge falloff for natural blending
        edge_falloff = np.where(ellipse_distance <= 1.2, 
                               np.exp(-(ellipse_distance - 1.0) * 8), 0.0)
        oval_mask = np.maximum(oval_mask, edge_falloff)
        
        # Apply natural noise to edge for organic appearance
        edge_noise = np.random.normal(0, 0.05, oval_mask.shape)
        oval_mask = np.clip(oval_mask + edge_noise, 0, 1)
        
        # Apply blur for realistic effect
        self.alpha_map = ndimage.gaussian_filter(self.alpha_map, sigma=1.5)
        
        # Apply the oval mask to create natural fingerprint shape
        self.alpha_map = self.alpha_map * oval_mask
        
        # Create a more visible fingerprint with stronger intensity
        if np.max(self.alpha_map) > 0:
            # Normalize first
            self.alpha_map = self.alpha_map / np.max(self.alpha_map)
            # Apply stronger intensity scaling
            self.alpha_map = self.alpha_map * self.intensity * 255
    
    def _create_loop_pattern(self, distance, angle, center_x, center_y):
        """Create a loop fingerprint pattern"""
        h, w = self.alpha_map.shape
        
        # Create more natural loop-like ridges that fade towards edges
        for i in range(3, min(self.size//2, 40), 4):
            # Create elliptical loops
            a = i * 1.3  # Semi-major axis  
            b = i * 0.8  # Semi-minor axis
            
            # Ellipse equation with slight distortion for naturalness
            ellipse_condition = ((distance * np.cos(angle - 0.2))**2 / a**2 + 
                               (distance * np.sin(angle - 0.2))**2 / b**2)
            
            # Create natural ridge pattern with varying frequency
            ridge_freq = 0.4 + np.random.uniform(-0.1, 0.1)
            ridge_pattern = np.sin(distance * ridge_freq + i * 0.3) * 0.7 + 0.3
            ridge_pattern = np.clip(ridge_pattern, 0, 1)
            
            # Natural ridge strength that decreases toward edge
            ridge_strength = 1.0 - (i / min(self.size//2, 40))
            ridge_strength = max(ridge_strength, 0.2)
            
            # Apply natural distance falloff
            natural_falloff = np.exp(-distance / (self.size * 0.6))
            
            # Apply to elliptical region with smooth boundaries
            mask = (ellipse_condition <= 1.0) & (distance < self.size * 0.8)
            self.alpha_map[mask] += (ridge_pattern[mask] * ridge_strength * 
                                   natural_falloff[mask] * 150)
    
    def _create_whorl_pattern(self, distance, angle, center_x, center_y):
        """Create a whorl fingerprint pattern"""
        # Create more natural spiral whorl pattern
        for i in range(5, min(self.size//2, 35), 3):
            # Enhanced spiral equation with natural variation
            spiral_distance = i * 0.7 + 2.5 * angle + np.random.uniform(-0.2, 0.2)
            
            # Create natural whorl ridges with varying intensity
            ridge_freq = 0.6 + np.random.uniform(-0.1, 0.1)
            ridge_pattern = (np.sin(spiral_distance * ridge_freq) * 
                           np.sin(distance * 0.4) * 0.8 + 0.2)
            ridge_pattern = np.clip(ridge_pattern, 0, 1)
            
            # Natural distance falloff
            falloff = np.exp(-distance / (self.size * 0.6))
            ridge_strength = 1.0 - (i / min(self.size//2, 35))
            ridge_strength = max(ridge_strength, 0.3)
            
            # Apply with smooth boundaries
            mask = distance < self.size * 0.8
            self.alpha_map[mask] += (ridge_pattern[mask] * falloff[mask] * 
                                   ridge_strength * 160)
    
    def _create_arch_pattern(self, distance, angle, center_x, center_y):
        """Create an arch fingerprint pattern"""
        h, w = self.alpha_map.shape
        y, x = np.ogrid[:h, :w]
        
        # Create more natural arch-like pattern
        for i in range(4, min(self.size//2, 35), 3):
            # Enhanced arch equation with natural curvature
            arch_y = (y - center_y) + 0.003 * (x - center_x)**2
            
            # Create natural horizontal ridges that curve smoothly
            ridge_freq = 0.25 + np.random.uniform(-0.05, 0.05)
            ridge_pattern = np.sin(arch_y * ridge_freq + i * 0.12) * 0.7 + 0.3
            ridge_pattern = np.clip(ridge_pattern, 0, 1)
            
            # Apply natural distance-based masking
            falloff = np.exp(-distance / (self.size * 0.6))
            ridge_strength = 1.0 - (i / min(self.size//2, 35))
            ridge_strength = max(ridge_strength, 0.25)
            
            # Natural arch boundary (more oval, less rectangular)
            mask = ((distance < self.size * 0.8) & 
                   (np.abs(y - center_y) < self.size * 0.6))
            
            self.alpha_map[mask] += (ridge_pattern[mask] * falloff[mask] * 
                                   ridge_strength * 140)
    
    def update_texture(self, background):
        """Apply fingerprint effect to background image"""
        bg_pil = Image.fromarray(np.uint8(background))
        
        # Apply slight blur and distortion to simulate smudging
        distorted_bg = bg_pil.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        # Convert back to numpy for processing
        distorted_bg = np.asarray(distorted_bg)
        
        # Apply slight geometric distortion
        h, w = self.alpha_map.shape
        if background.shape[0] >= h and background.shape[1] >= w:
            # Create subtle lens-like distortion
            y, x = np.ogrid[:h, :w]
            center_x, center_y = w // 2, h // 2
            
            # Barrel distortion
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            distortion_factor = 1 + 0.0001 * r**2
            
            # Apply distortion (simplified)
            distorted_bg = ndimage.zoom(distorted_bg, 
                                      [1.02, 1.02, 1] if len(distorted_bg.shape) == 3 else [1.02, 1.02])
            
            # Crop back to original size if needed
            if distorted_bg.shape[0] > h:
                start_y = (distorted_bg.shape[0] - h) // 2
                distorted_bg = distorted_bg[start_y:start_y + h]
            if distorted_bg.shape[1] > w:
                start_x = (distorted_bg.shape[1] - w) // 2
                distorted_bg = distorted_bg[:, start_x:start_x + w]
        
        # Create RGBA texture
        if len(distorted_bg.shape) == 3:
            alpha_channel = self.alpha_map.astype(np.uint8)
            texture_rgba = np.dstack([distorted_bg, alpha_channel])
        else:
            alpha_channel = self.alpha_map.astype(np.uint8)
            texture_rgba = np.dstack([
                np.stack([distorted_bg] * 3, axis=-1), 
                alpha_channel
            ])
        
        self.texture = Image.fromarray(texture_rgba.astype('uint8'), 'RGBA')
    
    def get_center(self):
        return self.center
    
    def get_size(self):
        return self.size
    
    def get_alpha_map(self):
        return self.alpha_map
    
    def get_texture(self):
        return self.texture
    
    def get_key(self):
        return self.key
    
    def get_intensity(self):
        return self.intensity
