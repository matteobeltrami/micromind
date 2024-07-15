from PIL import Image
import numpy as np
import skimage as sk
import albumentations as A
import imgaug.augmenters as iaa

import random
from io import BytesIO
import warnings

warnings.simplefilter("ignore", UserWarning)

IMG_SIZE = 640


def gaussian_noise(x, severity=1):
    vl = [(400, 400), (1000, 1000), (2000, 2000), (3300, 3300), (7000, 7000)][
        severity - 1
    ]

    transform = A.GaussNoise(var_limit=vl, mean=0, noise_scale_factor=1, p=1)
    x = transform(image=x)
    return x


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.0
    x = np.clip(np.random.poisson(x * c) / c, 0, 1) * 255
    x = x.astype(np.uint8)
    return x


def impulse_noise(x, severity=1):
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c)
    x = np.clip(x, 0, 1) * 255
    x = x.astype(np.uint8)
    return x


def speckle_noise(x, severity=1):
    c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.0
    x = np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    x = x.astype(np.uint8)
    return x


def gaussian_blur(x, severity=1):
    b = [(5, 5), (7, 7), (9, 9), (11, 11), (13, 13)][severity - 1]
    s = [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6)][severity - 1]

    transform = A.GaussianBlur(blur_limit=b, sigma_limit=s, p=1)
    x = transform(image=x)
    return x


def glass_blur(x, severity=1):
    s = [0.7, 0.9, 1.0, 1.1, 1.5][severity - 1]
    d = [1, 2, 2, 3, 4][severity - 1]
    i = [2, 1, 3, 2, 2][severity - 1]

    transform = A.GlassBlur(sigma=s, max_delta=d, iterations=i, p=1)
    x = transform(image=x)
    return x


def defocus_blur(x, severity=1):
    r = [(3, 3), (4, 4), (6, 6), (8, 8), (10, 10)][severity - 1]
    a = [(0.1, 0.1), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)][severity - 1]

    transform = A.Defocus(radius=r, alias_blur=a, p=1)
    x = transform(image=x)
    return x


def zoom_blur(x, severity=1):
    m = [(1.11, 1.11), (1.16, 1.16), (1.21, 1.21), (1.26, 1.26), (1.31, 1.31)][
        severity - 1
    ]
    s = [(0.01, 0.01), (0.01, 0.01), (0.02, 0.02), (0.02, 0.02), (0.03, 0.03)][
        severity - 1
    ]

    transform = A.ZoomBlur(max_factor=m, step_factor=s, p=1)

    x = transform(image=x)
    return x


def spatter(x, severity=1):
    m = [(0.65, 0.65), (0.65, 0.65), (0.65, 0.65), (0.65, 0.65), (0.67, 0.67)][
        severity - 1
    ]
    s = [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.4, 0.4)][severity - 1]
    g = [(4, 4), (3, 3), (2, 2), (1, 1), (1, 1)][severity - 1]
    t = [(0.69, 0.69), (0.68, 0.68), (0.68, 0.68), (0.65, 0.65), (0.65, 0.65)][
        severity - 1
    ]
    i = [(0.6, 0.6), (0.6, 0.6), (0.5, 0.5), (1.0, 1.0), (1.0, 1.0)][severity - 1]

    transform = A.Spatter(
        mean=m, std=s, gauss_sigma=g, cutout_threshold=t, intensity=i, p=1
    )

    x = transform(image=x)
    return x


def contrast(x, severity=1):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]

    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    x = np.clip((x - means) * c + means, 0, 1) * 255
    x = x.astype(np.uint8)
    return x


def brightness(x, severity=1):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]

    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    x = np.clip(x, 0, 1) * 255
    x = x.astype(np.uint8)
    return x


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    x = np.clip(x, 0, 1) * 255
    x = x.astype(np.uint8)
    return x


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x = Image.fromarray(np.uint8(x)).convert("RGB")
    x.save(output, "JPEG", quality=c)
    x = Image.open(output)
    x = np.array(x)
    return x


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x = Image.fromarray(np.uint8(x)).convert("RGB")
    x = x.resize((int(IMG_SIZE * c), int(IMG_SIZE * c)), Image.BOX)
    x = x.resize((IMG_SIZE, IMG_SIZE), Image.BOX)
    x = np.array(x)
    return x


def fog(x, severity=1):
    a = [0.1, 0.12, 0.15, 0.18, 0.18][severity - 1]
    f = [(0.5, 0.5), (0.6, 0.6), (0.7, 0.7), (0.8, 0.8), (0.9, 0.9)][severity - 1]

    transform = A.RandomFog(alpha_coef=a, fog_coef_range=f, p=1)
    x = transform(image=x)
    return x


def rain(x, severity=1):
    dl = [15, 18, 20, 20, 20][severity - 1]
    dw = [1, 1, 1, 1, 1][severity - 1]
    bl = [4, 5, 7, 7, 7][severity - 1]
    br = [0.8, 0.75, 0.7, 0.65, 0.6][severity - 1]
    rt = [None, None, None, "heavy", "torrential"][severity - 1]

    transform = A.RandomRain(
        drop_length=dl,
        drop_width=dw,
        blur_value=bl,
        brightness_coefficient=br,
        rain_type=rt,
        p=1,
    )
    x = transform(image=x)
    return x


def snow(x, severity=1):
    b = [2.1, 2.2, 2.3, 2.4, 2.5][severity - 1]
    s = [(0.25, 0.25), (0.27, 0.27), (0.32, 0.32), (0.37, 0.37), (0.45, 0.45)][
        severity - 1
    ]

    transform = A.RandomSnow(brightness_coeff=b, snow_point_range=s, p=1)
    x = transform(image=x)
    return x


def sunflare(x, severity=1):
    r = [250, 325, 400, 500, 600][severity - 1]
    c = [
        (200, 200, 200),
        (220, 220, 220),
        (240, 240, 240),
        (255, 255, 255),
        (255, 255, 255),
    ][severity - 1]

    transform = A.RandomSunFlare(src_radius=r, src_color=c, p=1)
    x = transform(image=x)
    return x


def frost(x, severity=1):
    aug = iaa.imgcorruptlike.Frost(severity=severity)
    x = aug(images=[x])
    return x[0]


class Corruptor:
    def __init__(self, apply=True, severity=1):
        self.apply = apply
        self.severity = severity
        self.transformations = [
            gaussian_noise,
            shot_noise,
            impulse_noise,
            speckle_noise,
            gaussian_blur,
            glass_blur,
            defocus_blur,
            zoom_blur,
            spatter,
            contrast,
            brightness,
            saturate,
            jpeg_compression,
            pixelate,
            fog,
            rain,
            snow,
            sunflare,
            frost,
        ]

    def __call__(self, x):
        if not self.apply or self.severity == 0:
            return x
        func = random.choice(self.transformations)
        image = x["img"]
        image = func(image, self.severity)
        if isinstance(image, dict):
            image = image["image"]
        x["img"] = image
        return x
