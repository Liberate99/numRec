import numpy as np
import math

# gamma变换
def adjust_gamma(src, gamma=0.25):
    scale = float(np.iinfo(src.dtype).max - np.iinfo(src.dtype).min)
    dst = ((src.astype(np.float32) / scale) ** gamma) * scale
    dst = np.clip(dst, 0, 255).astype(np.uint8)
    return dst