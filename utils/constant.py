DIST_DICT={
    # DISTORTION_CLASS:[DISTORTION_NAME]
    "brighten": [
        "brightness_brighten_shift_HSV",
        "brightness_brighten_shift_RGB",
        "brightness_brighten_gamma_HSV",
        "brightness_brighten_gamma_RGB",
    ],
    "darken":[
        "brightness_darken_shift_HSV",
        "brightness_darken_shift_RGB",
        "brightness_darken_gamma_HSV",
        "brightness_darken_gamma_RGB",
    ],
    "contrast_strengthen": [
        "contrast_strengthen_scale",
        "contrast_strengthen_stretch",
    ],
    "contrast_weaken": [
        "contrast_weaken_scale",
        "contrast_weaken_stretch",
    ],
    "saturate_strengthen": [
        "saturate_strengthen_HSV",
        "saturate_strengthen_YCrCb",
    ],
    "saturate_weaken": [
        "saturate_weaken_HSV",
        "saturate_weaken_YCrCb",
    ],
    "oversharpen": [
        "oversharpen",
    ],
    "temperature_warm": [
        "temperature_warm_RGB",
        "temperature_warm_LAB",
    ],
    "temperature_cool": [
        "temperature_cool_RGB",
        "temperature_cool_LAB",
    ],
    "tint_green": [
        "tint_green_RGB",
        "tint_green_LAB",
    ],
    "tint_magenta": [
        "tint_magenta_RGB",
        "tint_magenta_LAB",
    ],
    "exposure_increase": [
        "exposure_increase_LAB",
    ],
    "exposure_decrease": [
        "exposure_decrease_LAB",
    ],
}
# DISTORTION_CATEGORY:[DISTORTION_CLASS]
CATEGORY_TO_CLASSES = {
    "brightness": ["brighten", "darken"],
    "contrast": ["contrast_strengthen", "contrast_weaken"],
    "exposure": ["exposure_increase", "exposure_decrease"],
    "oversharpen": ["oversharpen"],
    "saturate": ["saturate_strengthen", "saturate_weaken"],
    "temperature": ["temperature_warm", "temperature_cool"],
    "tint": ["tint_green", "tint_magenta"],
}
