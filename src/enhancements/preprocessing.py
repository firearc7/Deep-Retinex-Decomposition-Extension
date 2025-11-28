"""
Advanced Preprocessing Module for Low-Light Images
Prepares images before Deep Retinex processing for better quality
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from scipy.ndimage import gaussian_filter


class ImagePreprocessor:
    """
    Preprocessing pipeline for low-light images
    Applied BEFORE feeding images to RetinexNet
    """
    
    def __init__(self):
        self.preprocessing_stats = {}
    
    def analyze_image_characteristics(self, image: np.ndarray) -> Dict:
        """
        Analyze image to determine preprocessing strategy
        
        Returns statistics about:
        - Brightness level
        - Noise level
        - Contrast
        - Color cast
        """
        stats = {}
        
        # Ensure float [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Brightness
        stats['mean_brightness'] = np.mean(image)
        stats['is_very_dark'] = stats['mean_brightness'] < 0.15
        stats['is_dark'] = stats['mean_brightness'] < 0.3
        
        # Contrast (standard deviation)
        stats['contrast'] = np.std(image)
        stats['is_low_contrast'] = stats['contrast'] < 0.15
        
        # Noise estimation (using Laplacian variance)
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        stats['noise_level'] = laplacian_var
        stats['is_noisy'] = laplacian_var > 500
        
        # Color cast detection
        if len(image.shape) == 3:
            channel_means = image.mean(axis=(0, 1))
            max_mean = channel_means.max()
            min_mean = channel_means.min()
            stats['color_cast'] = (max_mean - min_mean) / (max_mean + 1e-6)
            stats['has_color_cast'] = stats['color_cast'] > 0.15
        else:
            stats['color_cast'] = 0
            stats['has_color_cast'] = False
        
        # Dynamic range
        stats['dynamic_range'] = image.max() - image.min()
        stats['is_low_dynamic_range'] = stats['dynamic_range'] < 0.5
        
        return stats
    
    def denoise_adaptive(self, image: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Apply adaptive denoising based on noise level
        
        Args:
            noise_level: Estimated noise level from analysis
        """
        if noise_level < 200:
            # Low noise - mild denoising
            return cv2.fastNlMeansDenoisingColored(
                (image * 255).astype(np.uint8),
                None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
            ).astype(np.float32) / 255.0
        elif noise_level < 500:
            # Moderate noise
            return cv2.fastNlMeansDenoisingColored(
                (image * 255).astype(np.uint8),
                None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21
            ).astype(np.float32) / 255.0
        else:
            # High noise - aggressive denoising
            return cv2.fastNlMeansDenoisingColored(
                (image * 255).astype(np.uint8),
                None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
            ).astype(np.float32) / 255.0
    
    def correct_color_cast(self, image: np.ndarray, method: str = 'gray_world') -> np.ndarray:
        """
        Correct color cast using various methods
        
        Args:
            method: 'gray_world', 'white_patch', or 'auto'
        """
        if len(image.shape) != 3:
            return image
        
        if image.max() > 1.0:
            image = image / 255.0
        
        if method == 'gray_world':
            # Gray World assumption: average color should be gray
            avg = image.mean(axis=(0, 1))
            gray_avg = avg.mean()
            scaling = gray_avg / (avg + 1e-6)
            corrected = image * scaling[None, None, :]
            
        elif method == 'white_patch':
            # White Patch assumption: brightest point should be white
            max_vals = image.max(axis=(0, 1))
            scaling = 1.0 / (max_vals + 1e-6)
            corrected = image * scaling[None, None, :]
            
        else:  # auto
            # Combination of both methods
            gw_result = self.correct_color_cast(image, 'gray_world')
            wp_result = self.correct_color_cast(image, 'white_patch')
            corrected = 0.6 * gw_result + 0.4 * wp_result
        
        return np.clip(corrected, 0, 1)
    
    def enhance_dark_regions(self, image: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        """
        Selectively enhance very dark regions
        Uses spatially-varying gamma correction
        
        Args:
            threshold: Brightness threshold for dark regions
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        # Create brightness mask
        if len(image.shape) == 3:
            brightness = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            brightness = image
        
        # Dark region mask (smooth transition)
        dark_mask = np.clip((threshold - brightness) / threshold, 0, 1)
        
        # Apply stronger gamma to dark regions
        gamma = 0.5  # Brightening gamma
        enhanced = np.power(image, gamma)
        
        # Blend based on mask
        if len(image.shape) == 3:
            dark_mask = dark_mask[:, :, None]
        
        result = dark_mask * enhanced + (1 - dark_mask) * image
        
        return np.clip(result, 0, 1)
    
    def normalize_illumination(self, image: np.ndarray, method: str = 'retinex') -> np.ndarray:
        """
        Normalize illumination variations using single-scale retinex
        
        Args:
            method: 'retinex' or 'homomorphic'
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        if method == 'retinex':
            # Single-scale retinex
            sigma = 80
            image_log = np.log(image + 1e-6)
            blurred_log = gaussian_filter(image_log, sigma=sigma)
            normalized = image_log - blurred_log
            
            # Convert back and normalize
            result = np.exp(normalized)
            result = (result - result.min()) / (result.max() - result.min() + 1e-6)
            
        else:  # homomorphic
            # Homomorphic filtering in frequency domain
            result = np.zeros_like(image)
            
            for c in range(image.shape[2] if len(image.shape) == 3 else 1):
                if len(image.shape) == 3:
                    channel = image[:, :, c]
                else:
                    channel = image
                
                # FFT
                f = np.fft.fft2(np.log(channel + 1e-6))
                fshift = np.fft.fftshift(f)
                
                # High-pass filter
                rows, cols = channel.shape
                crow, ccol = rows // 2, cols // 2
                r = 30
                mask = np.ones((rows, cols))
                y, x = np.ogrid[:rows, :cols]
                mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= r * r
                mask[mask_area] = 0.3
                
                fshift = fshift * mask
                
                # Inverse FFT
                f_ishift = np.fft.ifftshift(fshift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                
                if len(image.shape) == 3:
                    result[:, :, c] = img_back
                else:
                    result = img_back
            
            # Normalize
            result = (result - result.min()) / (result.max() - result.min() + 1e-6)
        
        return result
    
    def preprocess_pipeline(self, image: np.ndarray, 
                          denoise: bool = True,
                          color_correction: bool = True,
                          enhance_dark: bool = True,
                          adaptive: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Full preprocessing pipeline with adaptive processing
        
        Args:
            denoise: Apply denoising if needed
            color_correction: Correct color cast
            enhance_dark: Enhance very dark regions
            adaptive: Automatically determine what to apply based on analysis
        
        Returns:
            Preprocessed image and processing stats
        """
        # Ensure [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0
        
        # Analyze image
        stats = self.analyze_image_characteristics(image)
        self.preprocessing_stats = stats
        
        processed = image.copy()
        applied_steps = []
        
        # Step 1: Denoising (if needed)
        if denoise and (not adaptive or stats['is_noisy']):
            processed = self.denoise_adaptive(processed, stats['noise_level'])
            applied_steps.append(f"denoising (noise_level={stats['noise_level']:.1f})")
        
        # Step 2: Color cast correction (if needed)
        if color_correction and (not adaptive or stats['has_color_cast']):
            processed = self.correct_color_cast(processed, method='auto')
            applied_steps.append(f"color_correction (cast={stats['color_cast']:.3f})")
        
        # Step 3: Enhance very dark regions (if needed)
        if enhance_dark and (not adaptive or stats['is_very_dark']):
            processed = self.enhance_dark_regions(processed, threshold=0.25)
            applied_steps.append(f"dark_enhancement (brightness={stats['mean_brightness']:.3f})")
        
        # Prepare stats for output
        output_stats = {
            'input_stats': stats,
            'applied_steps': applied_steps,
            'improvement': {
                'brightness': processed.mean() - image.mean(),
                'contrast': processed.std() - image.std()
            }
        }
        
        return processed, output_stats
    
    def preprocess_for_retinex(self, image: np.ndarray, 
                              preset: str = 'auto') -> Tuple[np.ndarray, Dict]:
        """
        Convenient wrapper for preprocessing before RetinexNet
        
        Args:
            preset: 'auto' (adaptive), 'minimal', 'standard', 'aggressive', 'none'
        """
        if preset == 'none':
            return image, {'applied_steps': []}
        
        elif preset == 'minimal':
            # Only denoise if very noisy
            return self.preprocess_pipeline(image, 
                                          denoise=True, 
                                          color_correction=False,
                                          enhance_dark=False, 
                                          adaptive=True)
        
        elif preset == 'standard':
            # Standard preprocessing
            return self.preprocess_pipeline(image,
                                          denoise=True,
                                          color_correction=True,
                                          enhance_dark=False,
                                          adaptive=False)
        
        elif preset == 'aggressive':
            # All preprocessing steps
            return self.preprocess_pipeline(image,
                                          denoise=True,
                                          color_correction=True,
                                          enhance_dark=True,
                                          adaptive=False)
        
        else:  # auto
            # Fully adaptive
            return self.preprocess_pipeline(image,
                                          denoise=True,
                                          color_correction=True,
                                          enhance_dark=True,
                                          adaptive=True)
