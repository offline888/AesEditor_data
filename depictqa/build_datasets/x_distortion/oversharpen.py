import cv2
import numpy as np


def oversharpen(img, severity=1):
    """OverSharpening filter."""
    # Unsharp Masking
    assert img.dtype == np.uint8, "Image array should have dtype of np.uint8"
    assert severity in [1, 2, 3, 4, 5], "Severity must be an integer between 1 and 5."

    amount = [2, 2.8, 4, 6, 8][severity - 1]
    # Create a blurred/smoothed version
    blur_radius = 5
    sigmaX = 0
    blurred = cv2.GaussianBlur(img, (blur_radius, blur_radius), sigmaX)
    # (1 + amount) * img - amount * blurred -> enhance high frequency, keep low frequency
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened

def sharpening_decrease(img, severity=1):
    """Sharpening Decrease by detail attenuation with edge-aware base (bilateral)."""
    assert img.dtype == np.uint8, "Image array should have dtype of np.uint8"
    assert severity in [1, 2, 3, 4, 5], "Severity must be an integer between 1 and 5."

    # 较大的 severity 表示更强的“去锐化”（更弱的细节）
    alpha_map = [0.8, 0.65, 0.5, 0.35, 0.2]  # 细节保留比例
    sigmaC_map = [20, 35, 50, 65, 80]        # 双边滤波颜色域
    sigmaS_map = [9, 12, 16, 20, 24]         # 双边滤波空间域
    d = 9

    alpha = alpha_map[severity - 1]
    sigma_color = sigmaC_map[severity - 1]
    sigma_space = sigmaS_map[severity - 1]

    img_f = img.astype(np.float32)
    # 双边滤波作为边缘保持的“基底”，能避免简单高斯带来的过度模糊
    base = cv2.bilateralFilter(img_f, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    detail = img_f - base
    out = base + alpha * detail  # 缩减细节层，从而压制过锐化与光晕
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out