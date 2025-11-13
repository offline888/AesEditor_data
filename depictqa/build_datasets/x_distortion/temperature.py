import cv2
import numpy as np

def temperature_warm_RGB(img,severity=1):
    # Red channel increase
    # Blue channel decrease
    red_factor=[1.08,1.15,1.23,1.32,1.42][severity-1]
    blue_factor=[0.92,0.85,0.77,0.68,0.58][severity-1]

    img = np.float32(np.array(img) / 255.0)
    # adjust RGB channels
    img[:,:,0]=img[:,:,0]*red_factor # red channel
    img[:,:,2]=img[:,:,2]*blue_factor # blue channel

    img_lq=np.clip(img,0,1)
    img_lq=np.uint8(img_lq*255.0)
    return img_lq

def temperature_cool_RGB(img,severity=1):
    # Red channel decrease
    # Blue channel increase
    red_factor=[0.92,0.85,0.77,0.68,0.58][severity-1]
    blue_factor=[1.08,1.15,1.23,1.32,1.42][severity-1]

    img = np.float32(np.array(img) / 255.0)
    # adjust RGB channels
    img[:,:,0]=img[:,:,0]*red_factor # red channel
    img[:,:,2]=img[:,:,2]*blue_factor # blue channel

    img_lq=np.clip(img,0,1)
    img_lq=np.uint8(img_lq*255.0)
    return img_lq
def temperature_warm_LAB(img,severity=1):
    # In LAB space, B channel represents blue-yellow axis
    # Positive B = yellow, Negative B = blue
    b_shift=[5,10,15,20,25][severity-1]

    img_lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    img_lab=img_lab.astype(np.float32)

    # shift B channel to yellow (warm)
    img_lab[:,:,2]=img_lab[:,:,2]+b_shift

    img_lab=np.clip(img_lab,0,255)
    img_lab=img_lab.astype(np.uint8)
    img_lq=cv2.cvtColor(img_lab,cv2.COLOR_LAB2RGB)
    return img_lq

def temperature_cool_LAB(img,severity=1):
    # In LAB space, B channel represents blue-yellow axis
    # Positive B = yellow, Negative B = blue
    b_shift=[-5,-10,-15,-20,-25][severity-1]

    img_lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    img_lab=img_lab.astype(np.float32)

    # shift B channel to blue (cool)
    img_lab[:,:,2]=img_lab[:,:,2]+b_shift

    img_lab=np.clip(img_lab,0,255)
    img_lab=img_lab.astype(np.uint8)
    img_lq=cv2.cvtColor(img_lab,cv2.COLOR_LAB2RGB)
    return img_lq