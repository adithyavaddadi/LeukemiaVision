import os
import io
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def aspect_safe_pad_to_square(pil_img, pad_color=(0, 0, 0)):
    """
    Pads a PIL Image to a square without distorting cell aspect ratio.
    """
    w, h = pil_img.size
    if w == h:
        return pil_img
    
    max_dim = max(w, h)
    new_img = Image.new("RGB", (max_dim, max_dim), pad_color)
    paste_x = (max_dim - w) // 2
    paste_y = (max_dim - h) // 2
    new_img.paste(pil_img, (paste_x, paste_y))
    return new_img

def preprocess_image_numpy(image_input, target_size=(224, 224)):
    """
    NumPy / PIL based preprocessing for inference & backend serving.
    image_input: PIL Image, file path, or raw bytes.
    Returns:
        tensor_numpy: shape (1, 3, H, W) normalized float32 array
        original_pil: PIL image in RGB mode
    """
    if isinstance(image_input, (str, os.PathLike)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")
        
    original_pil = pil_img.copy()
    
    # 1. Aspect-safe pad to square
    squared = aspect_safe_pad_to_square(pil_img)
    
    # 2. Bicubic resize
    resized = squared.resize(target_size, Image.Resampling.BICUBIC)
    
    # 3. Convert to float32 [0.0, 1.0]
    img_np = np.array(resized, dtype=np.float32) / 255.0
    
    # 4. Standardize with ImageNet stats
    norm_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    
    # 5. Transpose to CHW (Channel, Height, Width)
    chw_np = np.transpose(norm_np, (2, 0, 1))
    
    # 6. Add batch dimension (1, 3, H, W)
    batch_np = np.expand_dims(chw_np, axis=0).astype(np.float32)
    
    return batch_np, original_pil

def check_parity_torch_numpy():
    """
    Verifies that NumPy preprocessing and PyTorch preprocessing yield identical tensors (< 1e-5).
    """
    import torch
    import torchvision.transforms as T
    
    # Dummy image
    dummy = Image.fromarray((np.random.rand(450, 450, 3) * 255).astype(np.uint8))
    
    # Numpy pipeline
    np_out, _ = preprocess_image_numpy(dummy, target_size=(224, 224))
    
    # PyTorch torchvision pipeline
    squared = aspect_safe_pad_to_square(dummy)
    resized = squared.resize((224, 224), Image.Resampling.BICUBIC)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    torch_tensor = transform(resized).unsqueeze(0).numpy()
    
    diff = np.max(np.abs(np_out - torch_tensor))
    print(f"Torch vs NumPy Preprocess Parity Max Diff: {diff:.8f}")
    assert diff < 1e-5, f"Parity check failed with diff {diff}"
    print("Preprocess parity test PASSED.")
    return True

if __name__ == "__main__":
    check_parity_torch_numpy()
