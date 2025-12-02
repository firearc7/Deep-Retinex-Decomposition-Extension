# Traditional DIP Enhancement Techniques for Low-Light Image Enhancement

import cv2
import numpy as np
from typing import Tuple, Optional


class TraditionalEnhancements:
    # Collection of traditional DIP methods for image enhancement
    
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        # Apply histogram equalization to improve contrast
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(image.astype(np.uint8))
    
    @staticmethod
    def clahe(image: np.ndarray, clip_limit: float = 2.0, 
              tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        # Contrast Limited Adaptive Histogram Equalization
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        is_normalized = image.max() <= 1.0
        
        if len(image.shape) == 3:
            if is_normalized:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
            
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return result.astype(np.float32) / 255.0 if is_normalized else result.astype(np.float32)
        else:
            img_uint8 = (image * 255).astype(np.uint8) if is_normalized else image.astype(np.uint8)
            result = clahe_obj.apply(img_uint8)
            return result.astype(np.float32) / 255.0 if is_normalized else result.astype(np.float32)
    
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        # Apply gamma correction to adjust brightness
        if image.max() > 1.0:
            image = image / 255.0
        corrected = np.power(image, 1.0 / gamma)
        return corrected
    
    @staticmethod
    def adaptive_gamma_correction(image: np.ndarray) -> np.ndarray:
        # Automatically determine optimal gamma based on image statistics
        if image.max() > 1.0:
            image = image / 255.0
        
        mean_luminance = np.mean(image)
        gamma = -0.3 / (np.log10(mean_luminance + 1e-6))
        gamma = np.clip(gamma, 0.5, 3.0)
        
        return np.power(image, 1.0 / gamma)
    
    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9, 
                        sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        # Bilateral filtering for edge-preserving smoothing
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            return result.astype(np.float32) / 255.0
        else:
            return cv2.bilateralFilter(image.astype(np.uint8), d, sigma_color, sigma_space).astype(np.float32)
    
    @staticmethod
    def guided_filter(image: np.ndarray, guide: Optional[np.ndarray] = None, 
                     radius: int = 8, eps: float = 0.01) -> np.ndarray:
        # Guided filter for edge-preserving smoothing
        if guide is None:
            guide = image
            
        image = image.astype(np.float32)
        guide = guide.astype(np.float32)
        
        mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
        mean_image = cv2.boxFilter(image, -1, (radius, radius))
        mean_guide_image = cv2.boxFilter(guide * image, -1, (radius, radius))
        cov_guide_image = mean_guide_image - mean_guide * mean_image
        mean_guide_guide = cv2.boxFilter(guide * guide, -1, (radius, radius))
        var_guide = mean_guide_guide - mean_guide * mean_guide
        
        a = cov_guide_image / (var_guide + eps)
        b = mean_image - a * mean_guide
        
        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))
        
        return mean_a * guide + mean_b
    
    @staticmethod
    def unsharp_masking(image: np.ndarray, kernel_size: int = 5, 
                       sigma: float = 1.0, amount: float = 1.5, 
                       threshold: float = 0) -> np.ndarray:
        # Unsharp masking for sharpening
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sharpened = image + amount * (image - blurred)
        
        if threshold > 0:
            low_contrast_mask = np.abs(image - blurred) < threshold
            sharpened = np.where(low_contrast_mask, image, sharpened)
        
        return np.clip(sharpened, 0, 1 if image.max() <= 1.0 else 255)
    
    @staticmethod
    def multi_scale_retinex(image: np.ndarray, scales: list = [15, 80, 250], 
                           weights: Optional[list] = None) -> np.ndarray:
        # Multi-Scale Retinex for illumination normalization
        if weights is None:
            weights = [1.0 / len(scales)] * len(scales)
        
        if image.max() > 1.0:
            image = image / 255.0
        
        image = np.maximum(image, 1e-6)
        retinex = np.zeros_like(image, dtype=np.float32)
        
        for scale, weight in zip(scales, weights):
            blurred = cv2.GaussianBlur(image, (0, 0), scale)
            blurred = np.maximum(blurred, 1e-6)
            retinex += weight * (np.log(image + 1e-6) - np.log(blurred + 1e-6))
        
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6)
        return retinex
    
    @staticmethod
    def color_balance(image: np.ndarray, percent: float = 1) -> np.ndarray:
        # Automatic color balance using percentage stretching
        out_channels = []
        for channel in cv2.split(image):
            low_val = np.percentile(channel, percent)
            high_val = np.percentile(channel, 100 - percent)
            channel = np.clip(channel, low_val, high_val)
            channel = ((channel - low_val) / (high_val - low_val + 1e-6))
            out_channels.append(channel)
        
        return cv2.merge(out_channels)
    
    @staticmethod
    def local_contrast_enhancement(image: np.ndarray, grid_size: int = 8) -> np.ndarray:
        # Enhance local contrast by grid-based processing
        h, w = image.shape[:2]
        enhanced = np.zeros_like(image, dtype=np.float32)
        
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                i_end = min(i + grid_size, h)
                j_end = min(j + grid_size, w)
                
                patch = image[i:i_end, j:j_end]
                patch_min = patch.min()
                patch_max = patch.max()
                
                if patch_max > patch_min:
                    enhanced[i:i_end, j:j_end] = (patch - patch_min) / (patch_max - patch_min)
                else:
                    enhanced[i:i_end, j:j_end] = patch
        
        return enhanced
    
    @staticmethod
    def tone_mapping(image: np.ndarray, alpha: float = 0.5, beta: float = 0.5) -> np.ndarray:
        # Simple tone mapping for HDR-like effect
        if image.max() > 1.0:
            image = image / 255.0
        
        global_comp = np.mean(image)
        local_comp = cv2.GaussianBlur(image, (0, 0), 30)
        mapped = (image / (alpha * global_comp + beta * local_comp + 1e-6))
        
        return np.clip(mapped, 0, 1)
    
    @staticmethod
    def multi_scale_detail_enhancement(image: np.ndarray, num_scales: int = 3, 
                                      detail_strength: float = 1.5) -> np.ndarray:
        # Multi-scale detail enhancement using Laplacian pyramid
        if image.max() > 1.0:
            image = image / 255.0
        
        # Build Gaussian pyramid
        gaussian_pyramid = [image]
        for i in range(num_scales):
            gaussian_pyramid.append(cv2.pyrDown(gaussian_pyramid[-1]))
        
        # Build Laplacian pyramid
        laplacian_pyramid = []
        for i in range(num_scales):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian = gaussian_pyramid[i] - upsampled
            laplacian_pyramid.append(laplacian)
        
        # Enhance details
        enhanced_laplacian = [detail * detail_strength for detail in laplacian_pyramid]
        
        # Reconstruct
        reconstructed = gaussian_pyramid[-1]
        for i in range(num_scales - 1, -1, -1):
            size = (enhanced_laplacian[i].shape[1], enhanced_laplacian[i].shape[0])
            reconstructed = cv2.pyrUp(reconstructed, dstsize=size)
            reconstructed = reconstructed + enhanced_laplacian[i]
        
        return np.clip(reconstructed, 0, 1)
    
    @staticmethod
    def anisotropic_diffusion(image: np.ndarray, iterations: int = 10, 
                            kappa: float = 50, gamma: float = 0.1) -> np.ndarray:
        # Perona-Malik anisotropic diffusion for edge-preserving smoothing
        if image.max() > 1.0:
            image = image / 255.0
        
        img = image.copy().astype(np.float32)
        
        for _ in range(iterations):
            grad_n = np.roll(img, 1, axis=0) - img
            grad_s = np.roll(img, -1, axis=0) - img
            grad_e = np.roll(img, -1, axis=1) - img
            grad_w = np.roll(img, 1, axis=1) - img
            
            c_n = np.exp(-(grad_n / kappa) ** 2)
            c_s = np.exp(-(grad_s / kappa) ** 2)
            c_e = np.exp(-(grad_e / kappa) ** 2)
            c_w = np.exp(-(grad_w / kappa) ** 2)
            
            img += gamma * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
        
        return np.clip(img, 0, 1)
    
    @staticmethod
    def detail_preserving_smoothing(image: np.ndarray, sigma_s: float = 60, 
                                   sigma_r: float = 0.4) -> np.ndarray:
        # Domain transform for edge-preserving smoothing
        if image.max() > 1.0:
            image = image / 255.0
        
        img_uint8 = (image * 255).astype(np.uint8)
        result = cv2.edgePreservingFilter(img_uint8, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
        
        return result.astype(np.float32) / 255.0
    
    @staticmethod
    def contrast_stretching(image: np.ndarray, lower_percentile: float = 2, 
                          upper_percentile: float = 98) -> np.ndarray:
        # Percentile-based contrast stretching
        if image.max() > 1.0:
            image = image / 255.0
        
        if len(image.shape) == 3:
            stretched = np.zeros_like(image)
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                p_low = np.percentile(channel, lower_percentile)
                p_high = np.percentile(channel, upper_percentile)
                stretched[:, :, c] = np.clip((channel - p_low) / (p_high - p_low + 1e-6), 0, 1)
        else:
            p_low = np.percentile(image, lower_percentile)
            p_high = np.percentile(image, upper_percentile)
            stretched = np.clip((image - p_low) / (p_high - p_low + 1e-6), 0, 1)
        
        return stretched
    
    @staticmethod
    def shadow_enhancement(image: np.ndarray, shadow_threshold: float = 0.3,
                         enhancement_factor: float = 1.5) -> np.ndarray:
        # Selectively enhance shadow regions
        if image.max() > 1.0:
            image = image / 255.0
        
        if len(image.shape) == 3:
            luminance = cv2.cvtColor((image * 255).astype(np.uint8), 
                                    cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            luminance = image
        
        shadow_mask = np.clip((shadow_threshold - luminance) / shadow_threshold, 0, 1)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 5)
        
        if len(image.shape) == 3:
            shadow_mask = shadow_mask[:, :, None]
        
        enhanced = image * (1 + shadow_mask * (enhancement_factor - 1))
        return np.clip(enhanced, 0, 1)
