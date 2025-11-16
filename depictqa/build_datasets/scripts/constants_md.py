from depictqa.build_datasets.x_distortion import distortions_dict

multi_distortions_dict = {
    # Brightness 类别
    "brighten": [
        "contrast_strengthen", "contrast_weaken",
        "saturate_strengthen", "saturate_weaken",
        "temperature_warm", "temperature_cool",
        "tint_green", "tint_magenta",
        "oversharpen",
    ],
    "darken": [
        "contrast_strengthen", "contrast_weaken",
        "saturate_strengthen", "saturate_weaken",
        "temperature_warm", "temperature_cool",
        "tint_green", "tint_magenta",
        "oversharpen",
    ],
    
    # Contrast 类别
    "contrast_strengthen": [
        "brighten", "darken",
        "saturate_strengthen", "saturate_weaken",
        "exposure_increase", "exposure_decrease",
        "temperature_warm", "temperature_cool",
        "tint_green", "tint_magenta",
        "oversharpen",
    ],
    "contrast_weaken": [
        "brighten", "darken",
        "saturate_strengthen", "saturate_weaken",
        "exposure_increase", "exposure_decrease",
        "temperature_warm", "temperature_cool",
        "tint_green", "tint_magenta",
        "oversharpen",
    ],
    
    # Saturation 类别
    "saturate_strengthen": [
        "brighten", "darken",
        "contrast_strengthen", "contrast_weaken",
        "exposure_increase", "exposure_decrease",
        "temperature_warm", "temperature_cool",
        "tint_green", "tint_magenta",
        "oversharpen",
    ],
    "saturate_weaken": [
        "brighten", "darken",
        "contrast_strengthen", "contrast_weaken",
        "exposure_increase", "exposure_decrease",
        "temperature_warm", "temperature_cool",
        "tint_green", "tint_magenta",
        "oversharpen",
    ],
    
    # Exposure 类别
    "exposure_increase": [
        "contrast_strengthen", "contrast_weaken",
        "saturate_strengthen", "saturate_weaken",
        "temperature_warm", "temperature_cool",
        "tint_green", "tint_magenta",
        "oversharpen",
    ],
    "exposure_decrease": [
        "contrast_strengthen", "contrast_weaken",
        "saturate_strengthen", "saturate_weaken",
        "temperature_warm", "temperature_cool",
        "tint_green", "tint_magenta",
        "oversharpen",
    ],
    
    # Temperature 类别
    "temperature_warm": [
        "brighten", "darken",
        "contrast_strengthen", "contrast_weaken",
        "saturate_strengthen", "saturate_weaken",
        "exposure_increase", "exposure_decrease",
        "oversharpen",
    ],
    "temperature_cool": [
        "brighten", "darken",
        "contrast_strengthen", "contrast_weaken",
        "saturate_strengthen", "saturate_weaken",
        "exposure_increase", "exposure_decrease",
        "oversharpen",
    ],
    
    # Tint 类别
    "tint_green": [
        "brighten", "darken",
        "contrast_strengthen", "contrast_weaken",
        "saturate_strengthen", "saturate_weaken",
        "exposure_increase", "exposure_decrease",
        "oversharpen",
    ],
    "tint_magenta": [
        "brighten", "darken",
        "contrast_strengthen", "contrast_weaken",
        "saturate_strengthen", "saturate_weaken",
        "exposure_increase", "exposure_decrease",
        "oversharpen",
    ],
    # Oversharpen
    "oversharpen": [
        "brighten", "darken",
        "contrast_strengthen", "contrast_weaken",
        "saturate_strengthen", "saturate_weaken",
        "exposure_increase", "exposure_decrease",
        "temperature_warm", "temperature_cool",
        "tint_green", "tint_magenta",
    ],
}

for dist1 in multi_distortions_dict:
    assert dist1 in distortions_dict
    for dist2 in multi_distortions_dict[dist1]:
        assert dist2 in distortions_dict
