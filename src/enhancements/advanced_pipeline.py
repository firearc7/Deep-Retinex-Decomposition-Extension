"""
Enhanced Pipeline with Preprocessing and Advanced DIP Techniques
Combines preprocessing, Deep Retinex, and multi-stage enhancement
"""
import numpy as np
from typing import Dict, Optional, Tuple, List
from .preprocessing import ImagePreprocessor
from .traditional_dip import TraditionalEnhancements
from .pipeline import EnhancementPipeline, EnhancementFactory


class AdvancedEnhancementPipeline:
    """
    Multi-stage enhancement pipeline with preprocessing and advanced DIP
    
    Pipeline Stages:
    1. Preprocessing (before model)
    2. Deep Retinex decomposition + relighting (model)
    3. Multi-stage DIP enhancement (after model)
    4. Post-processing refinement
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.preprocessor = ImagePreprocessor()
        self.enhancements = TraditionalEnhancements()
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Default configuration for advanced pipeline"""
        return {
            # Preprocessing stage
            'preprocessing': {
                'enabled': True,
                'preset': 'auto',  # auto, minimal, standard, aggressive, none
                'custom': {
                    'denoise': True,
                    'color_correction': True,
                    'enhance_dark': True,
                    'adaptive': True
                }
            },
            
            # Illumination enhancement (applied to I_delta)
            'illumination_enhancement': {
                'enabled': True,
                'methods': [
                    {'name': 'anisotropic_diffusion', 'params': {'iterations': 8, 'kappa': 30}},
                    {'name': 'clahe', 'params': {'clip_limit': 2.0, 'tile_size': (8, 8)}},
                    {'name': 'adaptive_gamma_correction', 'params': {}}
                ]
            },
            
            # Reflectance enhancement (applied to R)
            'reflectance_enhancement': {
                'enabled': False,
                'methods': [
                    {'name': 'haze_removal', 'params': {'omega': 0.85}},
                    {'name': 'contrast_stretching', 'params': {}}
                ]
            },
            
            # Output enhancement (applied to reconstructed S)
            'output_enhancement': {
                'enabled': True,
                'methods': [
                    {'name': 'multi_scale_detail_enhancement', 'params': {'num_scales': 3, 'detail_strength': 1.3}},
                    {'name': 'shadow_enhancement', 'params': {'shadow_threshold': 0.35, 'enhancement_factor': 1.4}},
                    {'name': 'unsharp_masking', 'params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.2}},
                    {'name': 'color_balance', 'params': {'percent': 1}}
                ]
            },
            
            # Post-processing refinement
            'postprocessing': {
                'enabled': True,
                'methods': [
                    {'name': 'contrast_stretching', 'params': {'lower_percentile': 1, 'upper_percentile': 99}},
                    {'name': 'detail_preserving_smoothing', 'params': {'sigma_s': 40, 'sigma_r': 0.3}}
                ]
            }
        }
    
    def _apply_enhancement_methods(self, image: np.ndarray, methods: List[Dict]) -> np.ndarray:
        """
        Apply a list of enhancement methods sequentially
        
        Args:
            image: Input image
            methods: List of {'name': method_name, 'params': {param_dict}}
        """
        enhanced = image.copy()
        
        for method_config in methods:
            method_name = method_config['name']
            params = method_config.get('params', {})
            
            if hasattr(self.enhancements, method_name):
                method = getattr(self.enhancements, method_name)
                try:
                    enhanced = method(enhanced, **params)
                except Exception as e:
                    print(f"  ⚠ Warning: {method_name} failed: {e}")
                    # Continue with previous result
        
        return enhanced
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Preprocessing stage before model inference
        
        Returns:
            Preprocessed image and processing statistics
        """
        if not self.config['preprocessing']['enabled']:
            return image, {'applied_steps': []}
        
        preset = self.config['preprocessing']['preset']
        
        if preset == 'custom':
            custom_params = self.config['preprocessing']['custom']
            preprocessed, stats = self.preprocessor.preprocess_pipeline(
                image, **custom_params
            )
        else:
            preprocessed, stats = self.preprocessor.preprocess_for_retinex(
                image, preset=preset
            )
        
        return preprocessed, stats
    
    def enhance_illumination_map(self, I: np.ndarray) -> np.ndarray:
        """
        Enhance illumination map with advanced techniques
        
        Args:
            I: Illumination map (H x W x 1) or (H x W)
        """
        if not self.config['illumination_enhancement']['enabled']:
            return I
        
        # Ensure 2D for processing
        original_shape = I.shape
        if len(I.shape) == 3 and I.shape[2] == 1:
            I = I.squeeze()
        
        # Apply enhancement methods
        enhanced = self._apply_enhancement_methods(
            I, 
            self.config['illumination_enhancement']['methods']
        )
        
        # Restore shape
        if len(original_shape) == 3:
            enhanced = np.expand_dims(enhanced, axis=-1)
        
        return enhanced
    
    def enhance_reflectance_map(self, R: np.ndarray) -> np.ndarray:
        """
        Enhance reflectance map (usually minimal)
        
        Args:
            R: Reflectance map (H x W x 3)
        """
        if not self.config['reflectance_enhancement']['enabled']:
            return R
        
        enhanced = self._apply_enhancement_methods(
            R,
            self.config['reflectance_enhancement']['methods']
        )
        
        return enhanced
    
    def enhance_output_image(self, S: np.ndarray) -> np.ndarray:
        """
        Enhance final reconstructed image
        
        Args:
            S: Reconstructed image (H x W x 3)
        """
        if not self.config['output_enhancement']['enabled']:
            return S
        
        enhanced = self._apply_enhancement_methods(
            S,
            self.config['output_enhancement']['methods']
        )
        
        return enhanced
    
    def postprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Final post-processing refinement
        
        Args:
            image: Enhanced image
        """
        if not self.config['postprocessing']['enabled']:
            return image
        
        refined = self._apply_enhancement_methods(
            image,
            self.config['postprocessing']['methods']
        )
        
        return refined
    
    def process_full_pipeline(self, R: np.ndarray, I: np.ndarray,
                            I_delta: Optional[np.ndarray] = None,
                            original_image: Optional[np.ndarray] = None,
                            return_intermediates: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Process complete advanced enhancement pipeline
        
        Args:
            R: Reflectance map from DecomNet
            I: Original illumination map from DecomNet
            I_delta: Enhanced illumination from RelightNet (optional)
            original_image: Original input image (for preprocessing analysis)
            return_intermediates: Return all intermediate results
        
        Returns:
            Final enhanced image and intermediate results (if requested)
        """
        results = {}
        
        # Stage 1: Enhance reflectance (if configured)
        R_enhanced = self.enhance_reflectance_map(R)
        if return_intermediates:
            results['reflectance_enhanced'] = R_enhanced
        
        # Stage 2: Enhance illumination
        if I_delta is not None:
            I_enhanced = self.enhance_illumination_map(I_delta)
            if return_intermediates:
                results['illumination_type'] = 'I_delta_enhanced'
        else:
            I_enhanced = self.enhance_illumination_map(I)
            if return_intermediates:
                results['illumination_type'] = 'I_enhanced'
        
        if return_intermediates:
            results['illumination_enhanced'] = I_enhanced
        
        # Stage 3: Reconstruct image (S = R * I)
        # Ensure illumination has 3 channels
        if len(I_enhanced.shape) == 2:
            # Shape: (H, W) -> (H, W, 3)
            I_enhanced_3 = np.stack([I_enhanced] * 3, axis=-1)
        elif len(I_enhanced.shape) == 3 and I_enhanced.shape[-1] == 1:
            # Shape: (H, W, 1) -> (H, W, 3)
            I_enhanced_2d = I_enhanced[:, :, 0]  # Extract 2D
            I_enhanced_3 = np.stack([I_enhanced_2d] * 3, axis=-1)
        elif len(I_enhanced.shape) == 4:
            # Somehow got 4D, squeeze extra dimensions
            I_enhanced = np.squeeze(I_enhanced)
            if len(I_enhanced.shape) == 2:
                I_enhanced_3 = np.stack([I_enhanced] * 3, axis=-1)
            else:
                I_enhanced_3 = I_enhanced
        else:
            I_enhanced_3 = I_enhanced
        
        reconstructed = R_enhanced * I_enhanced_3
        if return_intermediates:
            results['reconstructed'] = reconstructed
        
        # Stage 4: Enhance output
        output_enhanced = self.enhance_output_image(reconstructed)
        if return_intermediates:
            results['output_enhanced'] = output_enhanced
        
        # Stage 5: Post-processing refinement
        final_output = self.postprocess_image(output_enhanced)
        if return_intermediates:
            results['final_output'] = final_output
        
        return final_output, results


class AdvancedEnhancementFactory:
    """
    Factory to create advanced enhancement presets
    """
    
    @staticmethod
    def create_config(preset: str) -> Dict:
        """
        Create configuration for advanced enhancement pipeline
        
        Presets:
        - 'minimal_plus': Light preprocessing + minimal enhancement
        - 'balanced_plus': Standard preprocessing + balanced enhancement (RECOMMENDED)
        - 'aggressive_plus': Full preprocessing + aggressive enhancement
        - 'quality_focused': Focus on detail and quality
        - 'speed_focused': Faster processing with fewer steps
        - 'outdoor_optimized': Optimized for outdoor low-light scenes
        - 'indoor_optimized': Optimized for indoor low-light scenes
        """
        
        presets = {
            'minimal_plus': {
                'preprocessing': {
                    'enabled': True,
                    'preset': 'minimal'
                },
                'illumination_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'clahe', 'params': {'clip_limit': 2.0, 'tile_size': (8, 8)}},
                        {'name': 'adaptive_gamma_correction', 'params': {}}
                    ]
                },
                'reflectance_enhancement': {'enabled': False, 'methods': []},
                'output_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'unsharp_masking', 'params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.0}},
                        {'name': 'color_balance', 'params': {'percent': 1}}
                    ]
                },
                'postprocessing': {'enabled': False, 'methods': []}
            },
            
            'balanced_plus': {
                'preprocessing': {
                    'enabled': True,
                    'preset': 'auto'
                },
                'illumination_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'anisotropic_diffusion', 'params': {'iterations': 8, 'kappa': 30, 'gamma': 0.1}},
                        {'name': 'clahe', 'params': {'clip_limit': 2.5, 'tile_size': (8, 8)}},
                        {'name': 'adaptive_gamma_correction', 'params': {}}
                    ]
                },
                'reflectance_enhancement': {'enabled': False, 'methods': []},
                'output_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'multi_scale_detail_enhancement', 'params': {'num_scales': 3, 'detail_strength': 1.3}},
                        {'name': 'shadow_enhancement', 'params': {'shadow_threshold': 0.35, 'enhancement_factor': 1.4}},
                        {'name': 'unsharp_masking', 'params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.2}},
                        {'name': 'color_balance', 'params': {'percent': 1}}
                    ]
                },
                'postprocessing': {
                    'enabled': True,
                    'methods': [
                        {'name': 'contrast_stretching', 'params': {'lower_percentile': 1, 'upper_percentile': 99}}
                    ]
                }
            },
            
            'aggressive_plus': {
                'preprocessing': {
                    'enabled': True,
                    'preset': 'aggressive'
                },
                'illumination_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'anisotropic_diffusion', 'params': {'iterations': 12, 'kappa': 25, 'gamma': 0.15}},
                        {'name': 'clahe', 'params': {'clip_limit': 3.5, 'tile_size': (8, 8)}},
                        {'name': 'adaptive_gamma_correction', 'params': {}}
                    ]
                },
                'reflectance_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'contrast_stretching', 'params': {'lower_percentile': 2, 'upper_percentile': 98}}
                    ]
                },
                'output_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'multi_scale_detail_enhancement', 'params': {'num_scales': 4, 'detail_strength': 1.8}},
                        {'name': 'shadow_enhancement', 'params': {'shadow_threshold': 0.4, 'enhancement_factor': 1.6}},
                        {'name': 'unsharp_masking', 'params': {'kernel_size': 7, 'sigma': 1.5, 'amount': 2.0}},
                        {'name': 'color_balance', 'params': {'percent': 2}},
                        {'name': 'tone_mapping', 'params': {'alpha': 0.4, 'beta': 0.6}}
                    ]
                },
                'postprocessing': {
                    'enabled': True,
                    'methods': [
                        {'name': 'contrast_stretching', 'params': {'lower_percentile': 1, 'upper_percentile': 99}},
                        {'name': 'detail_preserving_smoothing', 'params': {'sigma_s': 40, 'sigma_r': 0.3}}
                    ]
                }
            },
            
            'quality_focused': {
                'preprocessing': {
                    'enabled': True,
                    'preset': 'standard'
                },
                'illumination_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'anisotropic_diffusion', 'params': {'iterations': 15, 'kappa': 35, 'gamma': 0.1}},
                        {'name': 'adaptive_bilateral_filter', 'params': {'window_size': 5}},
                        {'name': 'clahe', 'params': {'clip_limit': 2.0, 'tile_size': (8, 8)}},
                    ]
                },
                'reflectance_enhancement': {'enabled': False, 'methods': []},
                'output_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'multi_scale_detail_enhancement', 'params': {'num_scales': 4, 'detail_strength': 1.5}},
                        {'name': 'shadow_enhancement', 'params': {'shadow_threshold': 0.3, 'enhancement_factor': 1.3}},
                        {'name': 'unsharp_masking', 'params': {'kernel_size': 5, 'sigma': 1.2, 'amount': 1.5}},
                        {'name': 'color_balance', 'params': {'percent': 0.5}}
                    ]
                },
                'postprocessing': {
                    'enabled': True,
                    'methods': [
                        {'name': 'detail_preserving_smoothing', 'params': {'sigma_s': 50, 'sigma_r': 0.25}},
                        {'name': 'contrast_stretching', 'params': {'lower_percentile': 0.5, 'upper_percentile': 99.5}}
                    ]
                }
            },
            
            'speed_focused': {
                'preprocessing': {
                    'enabled': True,
                    'preset': 'minimal'
                },
                'illumination_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'clahe', 'params': {'clip_limit': 2.0, 'tile_size': (8, 8)}},
                    ]
                },
                'reflectance_enhancement': {'enabled': False, 'methods': []},
                'output_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'unsharp_masking', 'params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.2}},
                        {'name': 'color_balance', 'params': {'percent': 1}}
                    ]
                },
                'postprocessing': {'enabled': False, 'methods': []}
            },
            
            'outdoor_optimized': {
                'preprocessing': {
                    'enabled': True,
                    'preset': 'auto'
                },
                'illumination_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'anisotropic_diffusion', 'params': {'iterations': 10, 'kappa': 40, 'gamma': 0.1}},
                        {'name': 'clahe', 'params': {'clip_limit': 2.5, 'tile_size': (12, 12)}},
                        {'name': 'adaptive_gamma_correction', 'params': {}}
                    ]
                },
                'reflectance_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'haze_removal', 'params': {'omega': 0.9, 't0': 0.15}}
                    ]
                },
                'output_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'multi_scale_detail_enhancement', 'params': {'num_scales': 3, 'detail_strength': 1.4}},
                        {'name': 'shadow_enhancement', 'params': {'shadow_threshold': 0.4, 'enhancement_factor': 1.5}},
                        {'name': 'unsharp_masking', 'params': {'kernel_size': 7, 'sigma': 1.5, 'amount': 1.3}},
                        {'name': 'color_balance', 'params': {'percent': 1.5}}
                    ]
                },
                'postprocessing': {
                    'enabled': True,
                    'methods': [
                        {'name': 'contrast_stretching', 'params': {'lower_percentile': 1, 'upper_percentile': 99}}
                    ]
                }
            },
            
            'indoor_optimized': {
                'preprocessing': {
                    'enabled': True,
                    'preset': 'auto'
                },
                'illumination_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'anisotropic_diffusion', 'params': {'iterations': 8, 'kappa': 25, 'gamma': 0.12}},
                        {'name': 'clahe', 'params': {'clip_limit': 3.0, 'tile_size': (8, 8)}},
                        {'name': 'adaptive_gamma_correction', 'params': {}}
                    ]
                },
                'reflectance_enhancement': {'enabled': False, 'methods': []},
                'output_enhancement': {
                    'enabled': True,
                    'methods': [
                        {'name': 'multi_scale_detail_enhancement', 'params': {'num_scales': 3, 'detail_strength': 1.2}},
                        {'name': 'shadow_enhancement', 'params': {'shadow_threshold': 0.3, 'enhancement_factor': 1.3}},
                        {'name': 'unsharp_masking', 'params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.1}},
                        {'name': 'color_balance', 'params': {'percent': 0.8}}
                    ]
                },
                'postprocessing': {
                    'enabled': True,
                    'methods': [
                        {'name': 'detail_preserving_smoothing', 'params': {'sigma_s': 35, 'sigma_r': 0.35}},
                        {'name': 'contrast_stretching', 'params': {'lower_percentile': 1, 'upper_percentile': 99}}
                    ]
                }
            }
        }
        
        return presets.get(preset, presets['balanced_plus'])
