"""
Enhancement Pipeline combining Deep Retinex with Traditional DIP
"""
import numpy as np
from typing import Dict, Optional, Tuple
from .traditional_dip import TraditionalEnhancements


class EnhancementPipeline:
    """
    Applies traditional DIP techniques at optimal stages of Retinex decomposition
    
    Enhancement Points:
    1. Post-Decomposition: Enhance R (reflectance) and I (illumination) separately
    2. Post-Relighting: Enhance I_delta (adjusted illumination)
    3. Post-Reconstruction: Enhance final output S
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.enhancements = TraditionalEnhancements()
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration for enhancements"""
        return {
            'apply_to_illumination': True,
            'apply_to_reflectance': False,
            'apply_to_output': True,
            
            # Illumination enhancements
            'illumination_methods': ['clahe', 'bilateral_filter', 'adaptive_gamma'],
            'clahe_params': {'clip_limit': 2.0, 'tile_size': (8, 8)},
            'bilateral_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
            'gamma': 2.2,
            
            # Reflectance enhancements
            'reflectance_methods': [],  # Usually don't enhance reflectance
            
            # Output enhancements
            'output_methods': ['unsharp_mask', 'color_balance'],
            'unsharp_params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.5},
            'color_balance_percent': 1,
        }
    
    def enhance_illumination(self, illumination: np.ndarray) -> np.ndarray:
        """
        Enhance illumination map using traditional DIP techniques
        This is the MOST IMPORTANT enhancement point
        
        Args:
            illumination: Illumination map from DecomNet or RelightNet (H x W x 1) or (H x W)
        
        Returns:
            Enhanced illumination map
        """
        if not self.config['apply_to_illumination']:
            return illumination
        
        enhanced = illumination.copy()
        
        # Ensure 2D for processing
        if len(enhanced.shape) == 3 and enhanced.shape[2] == 1:
            enhanced = enhanced.squeeze()
        
        for method in self.config['illumination_methods']:
            if method == 'clahe':
                enhanced = self.enhancements.clahe(
                    enhanced, 
                    **self.config['clahe_params']
                )
            elif method == 'bilateral_filter':
                enhanced = self.enhancements.bilateral_filter(
                    enhanced,
                    **self.config['bilateral_params']
                )
            elif method == 'adaptive_gamma':
                enhanced = self.enhancements.adaptive_gamma_correction(enhanced)
            elif method == 'gamma':
                enhanced = self.enhancements.gamma_correction(
                    enhanced,
                    gamma=self.config['gamma']
                )
            elif method == 'guided_filter':
                enhanced = self.enhancements.guided_filter(enhanced)
        
        # Restore original shape
        if len(illumination.shape) == 3 and illumination.shape[2] == 1:
            enhanced = np.expand_dims(enhanced, axis=-1)
        
        return enhanced
    
    def enhance_reflectance(self, reflectance: np.ndarray) -> np.ndarray:
        """
        Enhance reflectance map (usually minimal processing needed)
        
        Args:
            reflectance: Reflectance map from DecomNet (H x W x 3)
        
        Returns:
            Enhanced reflectance map
        """
        if not self.config['apply_to_reflectance']:
            return reflectance
        
        enhanced = reflectance.copy()
        
        for method in self.config['reflectance_methods']:
            if method == 'bilateral_filter':
                enhanced = self.enhancements.bilateral_filter(enhanced)
            elif method == 'color_balance':
                enhanced = self.enhancements.color_balance(enhanced)
        
        return enhanced
    
    def enhance_output(self, output: np.ndarray) -> np.ndarray:
        """
        Enhance final output image
        
        Args:
            output: Final reconstructed image (H x W x 3)
        
        Returns:
            Enhanced output image
        """
        if not self.config['apply_to_output']:
            return output
        
        enhanced = output.copy()
        
        for method in self.config['output_methods']:
            if method == 'unsharp_mask':
                enhanced = self.enhancements.unsharp_masking(
                    enhanced,
                    **self.config['unsharp_params']
                )
            elif method == 'color_balance':
                enhanced = self.enhancements.color_balance(
                    enhanced,
                    percent=self.config['color_balance_percent']
                )
            elif method == 'tone_mapping':
                enhanced = self.enhancements.tone_mapping(enhanced)
            elif method == 'local_contrast':
                enhanced = self.enhancements.local_contrast_enhancement(enhanced)
        
        return enhanced
    
    def process_full_pipeline(self, R: np.ndarray, I: np.ndarray, 
                             I_delta: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Process full enhancement pipeline
        
        Args:
            R: Reflectance map (H x W x 3)
            I: Original illumination map (H x W x 1)
            I_delta: Enhanced illumination from RelightNet (H x W x 1), optional
        
        Returns:
            Enhanced output image and intermediate results
        """
        results = {}
        
        # Enhance reflectance (if configured)
        R_enhanced = self.enhance_reflectance(R)
        results['reflectance_enhanced'] = R_enhanced
        
        # Enhance illumination
        if I_delta is not None:
            I_enhanced = self.enhance_illumination(I_delta)
            results['illumination_type'] = 'I_delta_enhanced'
        else:
            I_enhanced = self.enhance_illumination(I)
            results['illumination_type'] = 'I_enhanced'
        
        results['illumination_enhanced'] = I_enhanced
        
        # Reconstruct image: S = R * I
        # Ensure illumination has 3 channels
        if len(I_enhanced.shape) == 2:
            I_enhanced_3 = np.stack([I_enhanced] * 3, axis=-1)
        elif I_enhanced.shape[2] == 1:
            I_enhanced_3 = np.concatenate([I_enhanced] * 3, axis=-1)
        else:
            I_enhanced_3 = I_enhanced
        
        # Reconstruct
        reconstructed = R_enhanced * I_enhanced_3
        results['reconstructed'] = reconstructed
        
        # Enhance final output
        final_output = self.enhance_output(reconstructed)
        results['final_output'] = final_output
        
        return final_output, results


class EnhancementFactory:
    """Factory class to create different enhancement configurations"""
    
    @staticmethod
    def create_config(preset: str) -> Dict:
        """
        Create enhancement configuration from preset
        
        Presets:
        - 'none': No enhancements
        - 'minimal': Only essential enhancements
        - 'balanced': Balanced enhancement (recommended)
        - 'aggressive': Strong enhancements
        - 'illumination_only': Only enhance illumination
        - 'output_only': Only enhance final output
        """
        presets = {
            'none': {
                'apply_to_illumination': False,
                'apply_to_reflectance': False,
                'apply_to_output': False,
                'illumination_methods': [],
                'reflectance_methods': [],
                'output_methods': [],
            },
            
            'minimal': {
                'apply_to_illumination': True,
                'apply_to_reflectance': False,
                'apply_to_output': False,
                'illumination_methods': ['adaptive_gamma'],
                'reflectance_methods': [],
                'output_methods': [],
            },
            
            'balanced': {
                'apply_to_illumination': True,
                'apply_to_reflectance': False,
                'apply_to_output': True,
                'illumination_methods': ['clahe', 'bilateral_filter'],
                'clahe_params': {'clip_limit': 2.0, 'tile_size': (8, 8)},
                'bilateral_params': {'d': 7, 'sigma_color': 50, 'sigma_space': 50},
                'reflectance_methods': [],
                'output_methods': ['unsharp_mask', 'color_balance'],
                'unsharp_params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.2},
                'color_balance_percent': 1,
            },
            
            'aggressive': {
                'apply_to_illumination': True,
                'apply_to_reflectance': False,
                'apply_to_output': True,
                'illumination_methods': ['clahe', 'bilateral_filter', 'adaptive_gamma'],
                'clahe_params': {'clip_limit': 3.0, 'tile_size': (8, 8)},
                'bilateral_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
                'reflectance_methods': [],
                'output_methods': ['unsharp_mask', 'color_balance', 'tone_mapping'],
                'unsharp_params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 2.0},
                'color_balance_percent': 2,
            },
            
            'illumination_only': {
                'apply_to_illumination': True,
                'apply_to_reflectance': False,
                'apply_to_output': False,
                'illumination_methods': ['clahe', 'bilateral_filter', 'adaptive_gamma'],
                'clahe_params': {'clip_limit': 2.0, 'tile_size': (8, 8)},
                'bilateral_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
                'reflectance_methods': [],
                'output_methods': [],
            },
            
            'output_only': {
                'apply_to_illumination': False,
                'apply_to_reflectance': False,
                'apply_to_output': True,
                'illumination_methods': [],
                'reflectance_methods': [],
                'output_methods': ['unsharp_mask', 'color_balance'],
                'unsharp_params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.5},
                'color_balance_percent': 1,
            },
        }
        
        return presets.get(preset, presets['balanced'])
