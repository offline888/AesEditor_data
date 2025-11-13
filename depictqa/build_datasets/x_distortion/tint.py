import cv2
import numpy as np

def tint_green_RGB(img, severity=1):
    # Green channel increase
    # Red channel decrease
    green_factor=[1.08,1.15,1.23,1.32,1.42][severity-1]
    red_factor=[0.92,0.85,0.77,0.68,0.58][severity-1]

    img=np.array(img,dtype=np.float32)/255.0

    # adjust RGB channels
    img[:,:,1]=img[:,:,1]*green_factor # green channel
    img[:,:,0]=img[:,:,0]*red_factor # red channel

    img_lq=np.clip(img,0,1)
    img_lq=np.uint8(img_lq*255.0)
    return img_lq

def tint_magenta_RGB(img, severity=1):
    # Red channel increase
    # Green channel decrease
    red_factor=[1.08,1.15,1.23,1.32,1.42][severity-1]
    green_factor=[0.92,0.85,0.77,0.68,0.58][severity-1]

    img=np.array(img,dtype=np.float32)/255.0

    # adjust RGB channels
    img[:,:,0]=img[:,:,0]*red_factor # red channel
    img[:,:,1]=img[:,:,1]*green_factor # green channel

    img_lq=np.clip(img,0,1)
    img_lq=np.uint8(img_lq*255.0)
    return img_lq

def tint_green_LAB(img, severity=1):
    # A channel shift to green
    # In LAB space, A channel represents green-magenta axis
    # Positive A = magenta, Negative A = green
    a_shift=[-5,-10,-15,-20,-25][severity-1]

    img_lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    img_lab=img_lab.astype(np.float32)

    # shift A channel to green
    img_lab[:,:,1]=img_lab[:,:,1]+a_shift

    img_lab=np.clip(img_lab,0,255)
    img_lab=img_lab.astype(np.uint8)
    img_lq=cv2.cvtColor(img_lab,cv2.COLOR_LAB2RGB)
    return img_lq

def tint_magenta_LAB(img, severity=1):
    # A channel shift to magenta
    # In LAB space, A channel represents green-magenta axis
    # Positive A = magenta, Negative A = green
    a_shift=[5,10,15,20,25][severity-1]

    img_lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    img_lab=img_lab.astype(np.float32)

    # shift A channel to magenta
    img_lab[:,:,1]=img_lab[:,:,1]+a_shift

    img_lab=np.clip(img_lab,0,255)
    img_lab=img_lab.astype(np.uint8)
    img_lq=cv2.cvtColor(img_lab,cv2.COLOR_LAB2RGB)
    return img_lq
