import cv2
import numpy as np
from PIL import Image,ImageEnhance

# Exposure value：+1，光线量翻倍
# Exposure value：-1，光线量减半

# operate on LAB space
def exposure_increase_LAB(img, severity=1):
    ev = [0.5, 1.0, 1.5, 2.0, 2.5][severity - 1]
    gain = 2 ** ev
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_lab = img_lab.astype(np.float32)
    
    # Only adjust the L channel
    img_lab[:, :, 0] = img_lab[:, :, 0] * gain
    
    img_lab = np.clip(img_lab, 0, 255)
    img_lab = img_lab.astype(np.uint8)
    img_lq = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return img_lq

def exposure_decrease_LAB(img,severity=1):
    ev=[-0.5,-1.0,-1.5,-2.0,-2.5][severity-1]
    gain=2**ev

    img_lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    img_lab=img_lab.astype(np.float32)

    img_lab[:,:,0]=img_lab[:,:,0]*gain
    
    img_lab=np.clip(img_lab,0,255)
    img_lab=img_lab.astype(np.uint8)
    img_lq=cv2.cvtColor(img_lab,cv2.COLOR_LAB2RGB)
    return img_lq



