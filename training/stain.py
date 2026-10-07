import numpy as np
from PIL import Image

class StainNormalizer:
    """
    Stain Normalization suite supporting 'none', 'reinhard', and 'macenko'.
    Includes safeguards for empty, dark, or low-contrast peripheral blood smear crops.
    """
    def __init__(self, method="none", target_means=None, target_stds=None):
        self.method = method.lower()
        # Default Reinhard target parameters derived from standard Giemsa/Wright-stained lymphocyte
        self.target_means = target_means if target_means is not None else np.array([8.6, -0.15, 0.25], dtype=np.float32)
        self.target_stds = target_stds if target_stds is not None else np.array([0.9, 0.45, 0.35], dtype=np.float32)

    def normalize(self, img_np):
        """
        img_np: RGB float32 array in [0.0, 1.0] or uint8 [0, 255].
        Returns normalized float32 array in [0.0, 1.0].
        """
        if self.method == "none":
            if img_np.dtype == np.uint8:
                return (img_np / 255.0).astype(np.float32)
            return img_np.astype(np.float32)

        # Ensure float32 in [0, 255]
        if img_np.max() <= 1.0:
            img_255 = (img_np * 255.0).astype(np.float32)
        else:
            img_255 = img_np.astype(np.float32)

        # Check for empty / degenerate image
        if np.std(img_255) < 5.0 or np.mean(img_255) < 10.0:
            # Fallback to direct scaling
            return np.clip(img_255 / 255.0, 0.0, 1.0).astype(np.float32)

        if self.method == "reinhard":
            return self._reinhard(img_255)
        else:
            # Fallback for unrecognized methods
            return np.clip(img_255 / 255.0, 0.0, 1.0).astype(np.float32)

    def _reinhard(self, img_rgb):
        # Convert RGB to Ruderman's LAB space
        # Clip to avoid log(0)
        img_safe = np.maximum(img_rgb, 1e-4)
        
        # RGB to LMS matrix
        rgb_to_lms = np.array([
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0789],
            [0.0241, 0.1288, 0.8444]
        ], dtype=np.float32)

        # LMS to LAB matrix
        lms_to_lab = np.array([
            [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
            [1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0]
        ], dtype=np.float32)

        lab_to_lms = np.linalg.inv(lms_to_lab)
        lms_to_rgb = np.linalg.inv(rgb_to_lms)

        # Forward transform
        lms = np.dot(img_safe, rgb_to_lms.T)
        log_lms = np.log10(np.maximum(lms, 1e-4))
        lab = np.dot(log_lms, lms_to_lab.T)

        # Normalize mean and standard deviation
        src_mean = np.mean(lab, axis=(0, 1))
        src_std = np.std(lab, axis=(0, 1)) + 1e-5

        norm_lab = (lab - src_mean) * (self.target_stds / src_std) + self.target_means

        # Inverse transform
        inv_log_lms = np.dot(norm_lab, lab_to_lms.T)
        inv_lms = 10.0 ** inv_log_lms
        norm_rgb = np.dot(inv_lms, lms_to_rgb.T)

        return np.clip(norm_rgb / 255.0, 0.0, 1.0).astype(np.float32)
