"""
Traditional Digital Image Processing Enhancement Techniques
This module provides various DIP methods to enhance the output of Deep Retinex decomposition
"""
import cv2
import numpy as np
from typing import Tuple, Optional


class TraditionalEnhancements:
    """
    Collection of traditional DIP techniques for image enhancement
    These can be applied to:
    1. Enhanced illumination map (I_delta) - BEST PLACE
    2. Reflectance map (R)
    3. Final output image (S)
    """
    
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to improve contrast
        Best applied to: Illumination map or final grayscale image
        """
        if len(image.shape) == 3:
            # Convert to YCrCb and equalize Y channel
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(image.astype(np.uint8))
    
    @staticmethod
    def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization
        Best applied to: Illumination map (prevents over-amplification of noise)
        
        Args:
            clip_limit: Threshold for contrast limiting
            tile_size: Size of grid for histogram equalization
        """
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        
        # Ensure image is in proper format
        is_normalized = image.max() <= 1.0
        
        if len(image.shape) == 3:
            # Convert to uint8 first
            if is_normalized:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
            
            # Apply to each channel or convert to LAB
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Convert back to normalized if input was normalized
            return result.astype(np.float32) / 255.0 if is_normalized else result.astype(np.float32)
        else:
            img_uint8 = (image * 255).astype(np.uint8) if is_normalized else image.astype(np.uint8)
            result = clahe_obj.apply(img_uint8)
            return result.astype(np.float32) / 255.0 if is_normalized else result.astype(np.float32)
    
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """
        Apply gamma correction to adjust brightness
        Best applied to: Illumination map or final output
        
        Args:
            gamma: Gamma value (>1 brightens, <1 darkens)
        """
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
            
        corrected = np.power(image, 1.0 / gamma)
        return corrected
    
    @staticmethod
    def adaptive_gamma_correction(image: np.ndarray) -> np.ndarray:
        """
        Automatically determine optimal gamma based on image statistics
        Best applied to: Illumination map
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        # Calculate mean luminance
        mean_luminance = np.mean(image)
        
        # Adaptive gamma calculation
        gamma = -0.3 / (np.log10(mean_luminance + 1e-6))
        gamma = np.clip(gamma, 0.5, 3.0)
        
        return np.power(image, 1.0 / gamma)
    
    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, 
                        sigma_space: float = 75) -> np.ndarray:
        """
        Bilateral filtering for edge-preserving smoothing
        Best applied to: Illumination map (reduces noise while preserving edges)
        
        Args:
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
        """
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            return result.astype(np.float32) / 255.0
        else:
            return cv2.bilateralFilter(image.astype(np.uint8), d, sigma_color, sigma_space).astype(np.float32)
    
    @staticmethod
    def guided_filter(image: np.ndarray, guide: Optional[np.ndarray] = None, 
                     radius: int = 8, eps: float = 0.01) -> np.ndarray:
        """
        Guided filter for edge-preserving smoothing
        Best applied to: Illumination map with reflectance as guide
        
        Args:
            guide: Guide image (if None, uses input image)
            radius: Radius of local window
            eps: Regularization parameter
        """
        if guide is None:
            guide = image
            
        # Ensure float type
        image = image.astype(np.float32)
        guide = guide.astype(np.float32)
        
        # Mean of guide and input
        mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
        mean_image = cv2.boxFilter(image, -1, (radius, radius))
        
        # Correlation
        mean_guide_image = cv2.boxFilter(guide * image, -1, (radius, radius))
        
        # Covariance
        cov_guide_image = mean_guide_image - mean_guide * mean_image
        
        # Variance
        mean_guide_guide = cv2.boxFilter(guide * guide, -1, (radius, radius))
        var_guide = mean_guide_guide - mean_guide * mean_guide
        
        # Linear coefficients
        a = cov_guide_image / (var_guide + eps)
        b = mean_image - a * mean_guide
        
        # Mean of coefficients
        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))
        
        # Output
        return mean_a * guide + mean_b
    
    @staticmethod
    def unsharp_masking(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, 
                       amount: float = 1.5, threshold: float = 0) -> np.ndarray:
        """
        Unsharp masking for sharpening
        Best applied to: Final output or reflectance map
        
        Args:
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation for Gaussian
            amount: Strength of sharpening
            threshold: Minimum difference required to sharpen
        """
        # Blur the image
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        # Calculate the sharpened image
        sharpened = image + amount * (image - blurred)
        
        # Apply threshold
        if threshold > 0:
            low_contrast_mask = np.abs(image - blurred) < threshold
            sharpened = np.where(low_contrast_mask, image, sharpened)
        
        return np.clip(sharpened, 0, 1 if image.max() <= 1.0 else 255)
    
    @staticmethod
    def multi_scale_retinex(image: np.ndarray, scales: list = [15, 80, 250], 
                           weights: Optional[list] = None) -> np.ndarray:
        """
        Multi-Scale Retinex for illumination normalization
        Best applied to: Can enhance the illumination adjustment
        
        Args:
            scales: List of Gaussian kernel sizes
            weights: Weights for each scale (default: equal weights)
        """
        if weights is None:
            weights = [1.0 / len(scales)] * len(scales)
        
        if image.max() > 1.0:
            image = image / 255.0
        
        # Ensure non-zero values
        image = np.maximum(image, 1e-6)
        
        retinex = np.zeros_like(image, dtype=np.float32)
        
        for scale, weight in zip(scales, weights):
            # Gaussian blur
            blurred = cv2.GaussianBlur(image, (0, 0), scale)
            blurred = np.maximum(blurred, 1e-6)
            
            # Log domain subtraction
            retinex += weight * (np.log(image + 1e-6) - np.log(blurred + 1e-6))
        
        # Normalize to [0, 1]
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6)
        
        return retinex
    
    @staticmethod
    def color_balance(image: np.ndarray, percent: float = 1) -> np.ndarray:
        """
        Automatic color balance using percentage stretching
        Best applied to: Final output for color correction
        
        Args:
            percent: Percentage of pixels to clip from each end
        """
        out_channels = []
        for channel in cv2.split(image):
            # Calculate percentiles
            low_val = np.percentile(channel, percent)
            high_val = np.percentile(channel, 100 - percent)
            
            # Stretch
            channel = np.clip(channel, low_val, high_val)
            channel = ((channel - low_val) / (high_val - low_val + 1e-6))
            
            out_channels.append(channel)
        
        return cv2.merge(out_channels)
    
    @staticmethod
    def local_contrast_enhancement(image: np.ndarray, grid_size: int = 8) -> np.ndarray:
        """
        Enhance local contrast by grid-based processing
        Best applied to: Final output
        
        Args:
            grid_size: Size of local grid
        """
        h, w = image.shape[:2]
        enhanced = np.zeros_like(image, dtype=np.float32)
        
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                i_end = min(i + grid_size, h)
                j_end = min(j + grid_size, w)
                
                patch = image[i:i_end, j:j_end]
                
                # Normalize patch
                patch_min = patch.min()
                patch_max = patch.max()
                
                if patch_max > patch_min:
                    enhanced[i:i_end, j:j_end] = (patch - patch_min) / (patch_max - patch_min)
                else:
                    enhanced[i:i_end, j:j_end] = patch
        
        return enhanced
    
    @staticmethod
    def tone_mapping(image: np.ndarray, alpha: float = 0.5, beta: float = 0.5) -> np.ndarray:
        """
        Simple tone mapping for HDR-like effect
        Best applied to: Final output
        
        Args:
            alpha: Global adaptation factor
            beta: Local adaptation factor
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        # Global component
        global_comp = np.mean(image)
        
        # Local component (using bilateral filter approximation)
        local_comp = cv2.GaussianBlur(image, (0, 0), 30)
        
        # Tone mapping
        mapped = (image / (alpha * global_comp + beta * local_comp + 1e-6))
        
        return np.clip(mapped, 0, 1)
