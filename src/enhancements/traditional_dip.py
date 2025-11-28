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
    
    @staticmethod
    def multi_scale_detail_enhancement(image: np.ndarray, num_scales: int = 3, 
                                      detail_strength: float = 1.5) -> np.ndarray:
        """
        Multi-scale detail enhancement using Laplacian pyramid
        Best applied to: Reflectance or final output
        
        Args:
            num_scales: Number of pyramid levels
            detail_strength: Amplification factor for details
        """
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
        
        # Enhance details (amplify Laplacian levels)
        enhanced_laplacian = [detail * detail_strength for detail in laplacian_pyramid]
        
        # Reconstruct
        reconstructed = gaussian_pyramid[-1]
        for i in range(num_scales - 1, -1, -1):
            size = (enhanced_laplacian[i].shape[1], enhanced_laplacian[i].shape[0])
            reconstructed = cv2.pyrUp(reconstructed, dstsize=size)
            reconstructed = reconstructed + enhanced_laplacian[i]
        
        return np.clip(reconstructed, 0, 1)
    
    @staticmethod
    def adaptive_bilateral_filter(image: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Adaptive bilateral filter with edge-aware parameters
        Best applied to: Illumination map
        
        Args:
            window_size: Size of local window for adaptation
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        # Calculate local variance for each pixel
        mean = cv2.boxFilter(image, -1, (window_size, window_size))
        mean_sq = cv2.boxFilter(image**2, -1, (window_size, window_size))
        variance = mean_sq - mean**2
        
        # Normalize variance to [0, 1]
        variance = (variance - variance.min()) / (variance.max() - variance.min() + 1e-6)
        
        # Apply bilateral filter with varying parameters
        # High variance (edges) -> less filtering
        # Low variance (smooth) -> more filtering
        result = np.zeros_like(image)
        
        # Convert to uint8 for bilateral filter
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply with adaptive sigma
        base_sigma_color = 50
        base_sigma_space = 50
        
        for i in range(0, image.shape[0], window_size):
            for j in range(0, image.shape[1], window_size):
                i_end = min(i + window_size, image.shape[0])
                j_end = min(j + window_size, image.shape[1])
                
                patch = img_uint8[i:i_end, j:j_end]
                var_patch = variance[i:i_end, j:j_end].mean()
                
                # Adapt parameters based on local variance
                sigma_color = int(base_sigma_color * (1 - var_patch * 0.5))
                sigma_space = int(base_sigma_space * (1 - var_patch * 0.5))
                
                filtered = cv2.bilateralFilter(patch, 9, sigma_color, sigma_space)
                result[i:i_end, j:j_end] = filtered.astype(np.float32) / 255.0
        
        return result
    
    @staticmethod
    def anisotropic_diffusion(image: np.ndarray, iterations: int = 10, 
                            kappa: float = 50, gamma: float = 0.1) -> np.ndarray:
        """
        Perona-Malik anisotropic diffusion for edge-preserving smoothing
        Best applied to: Illumination map (better than bilateral for gradual transitions)
        
        Args:
            iterations: Number of diffusion iterations
            kappa: Conduction coefficient (controls edge sensitivity)
            gamma: Rate of diffusion (0 < gamma <= 0.25 for stability)
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        img = image.copy().astype(np.float32)
        
        for _ in range(iterations):
            # Calculate gradients
            grad_n = np.roll(img, 1, axis=0) - img  # North
            grad_s = np.roll(img, -1, axis=0) - img  # South
            grad_e = np.roll(img, -1, axis=1) - img  # East
            grad_w = np.roll(img, 1, axis=1) - img  # West
            
            # Conduction coefficient (edge-stopping function)
            c_n = np.exp(-(grad_n / kappa) ** 2)
            c_s = np.exp(-(grad_s / kappa) ** 2)
            c_e = np.exp(-(grad_e / kappa) ** 2)
            c_w = np.exp(-(grad_w / kappa) ** 2)
            
            # Update image
            img += gamma * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
        
        return np.clip(img, 0, 1)
    
    @staticmethod
    def detail_preserving_smoothing(image: np.ndarray, sigma_s: float = 60, 
                                   sigma_r: float = 0.4) -> np.ndarray:
        """
        Domain transform for edge-preserving smoothing (fast alternative to bilateral)
        Best applied to: Any stage where smoothing is needed
        
        Args:
            sigma_s: Spatial sigma
            sigma_r: Range sigma
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        # Use OpenCV's edge-preserving filter
        img_uint8 = (image * 255).astype(np.uint8)
        result = cv2.edgePreservingFilter(img_uint8, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
        
        return result.astype(np.float32) / 255.0
    
    @staticmethod
    def contrast_stretching(image: np.ndarray, lower_percentile: float = 2, 
                          upper_percentile: float = 98) -> np.ndarray:
        """
        Percentile-based contrast stretching
        Best applied to: Final output or illumination
        
        Args:
            lower_percentile: Lower clip percentile
            upper_percentile: Upper clip percentile
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        # Calculate percentiles for each channel
        if len(image.shape) == 3:
            stretched = np.zeros_like(image)
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                p_low = np.percentile(channel, lower_percentile)
                p_high = np.percentile(channel, upper_percentile)
                
                stretched[:, :, c] = np.clip(
                    (channel - p_low) / (p_high - p_low + 1e-6), 0, 1
                )
        else:
            p_low = np.percentile(image, lower_percentile)
            p_high = np.percentile(image, upper_percentile)
            stretched = np.clip((image - p_low) / (p_high - p_low + 1e-6), 0, 1)
        
        return stretched
    
    @staticmethod
    def shadow_enhancement(image: np.ndarray, shadow_threshold: float = 0.3,
                         enhancement_factor: float = 1.5) -> np.ndarray:
        """
        Selectively enhance shadow regions
        Best applied to: Final output for low-light images
        
        Args:
            shadow_threshold: Brightness threshold for shadows
            enhancement_factor: How much to brighten shadows
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        # Calculate luminance
        if len(image.shape) == 3:
            luminance = cv2.cvtColor((image * 255).astype(np.uint8), 
                                    cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            luminance = image
        
        # Create shadow mask with smooth transition
        shadow_mask = np.clip((shadow_threshold - luminance) / shadow_threshold, 0, 1)
        
        # Smooth the mask to avoid artifacts
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 5)
        
        # Apply enhancement
        if len(image.shape) == 3:
            shadow_mask = shadow_mask[:, :, None]
        
        enhanced = image * (1 + shadow_mask * (enhancement_factor - 1))
        
        return np.clip(enhanced, 0, 1)
    
    @staticmethod
    def haze_removal(image: np.ndarray, omega: float = 0.95, t0: float = 0.1,
                    window_size: int = 15) -> np.ndarray:
        """
        Dark channel prior for haze/fog removal (adapted for low-light)
        Can help with atmospheric effects in low-light outdoor scenes
        Best applied to: Reflectance map
        
        Args:
            omega: Haze retention factor (0-1)
            t0: Minimum transmission
            window_size: Window for dark channel calculation
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        # Dark channel
        min_channel = np.min(image, axis=2) if len(image.shape) == 3 else image
        dark_channel = cv2.erode(min_channel, np.ones((window_size, window_size)))
        
        # Atmospheric light (brightest region in dark channel)
        flat_dark = dark_channel.flatten()
        num_pixels = len(flat_dark)
        num_brightest = int(num_pixels * 0.001)
        indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
        
        if len(image.shape) == 3:
            brightest_pixels = image.reshape(-1, 3)[indices]
            atmospheric_light = brightest_pixels.max(axis=0)
        else:
            atmospheric_light = flat_dark[indices].max()
        
        # Transmission map
        transmission = 1 - omega * (dark_channel / (atmospheric_light.max() + 1e-6))
        transmission = np.maximum(transmission, t0)
        
        # Recover scene radiance
        if len(image.shape) == 3:
            transmission = transmission[:, :, None]
            atmospheric_light = atmospheric_light[None, None, :]
        
        recovered = (image - atmospheric_light) / (transmission + 1e-6) + atmospheric_light
        
        return np.clip(recovered, 0, 1)
    
    @staticmethod
    def ssr_with_color_restoration(image: np.ndarray, sigma: float = 80,
                                   gain: float = 128, offset: float = 128) -> np.ndarray:
        """
        Single-Scale Retinex with color restoration
        Best applied to: Can be alternative to model-based illumination adjustment
        
        Args:
            sigma: Gaussian kernel sigma
            gain: Amplification gain
            offset: Output offset
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        # Apply SSR
        image_log = np.log(image + 1e-6)
        
        if len(image.shape) == 3:
            retinex = np.zeros_like(image)
            for c in range(3):
                blurred = cv2.GaussianBlur(image[:, :, c], (0, 0), sigma)
                retinex[:, :, c] = image_log[:, :, c] - np.log(blurred + 1e-6)
        else:
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            retinex = image_log - np.log(blurred + 1e-6)
        
        # Normalize
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6)
        
        # Color restoration
        if len(image.shape) == 3:
            intensity = image.sum(axis=2) / 3.0
            for c in range(3):
                alpha = np.log(gain * image[:, :, c] / (intensity + 1e-6) + 1e-6)
                retinex[:, :, c] = retinex[:, :, c] * alpha
        
        # Gain and offset
        retinex = gain * retinex + offset
        retinex = retinex / 255.0
        
        return np.clip(retinex, 0, 1)
