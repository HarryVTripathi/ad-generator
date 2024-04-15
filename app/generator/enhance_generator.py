from PIL import Image, ImageEnhance, ImageDraw
import numpy as np
import cv2 as cv
from generator.image_filter import ImageFilter


def enhance_image(image: Image):
    img = np.array(image)

    img_enhanced = img.copy()
    img_enhanced = cv.convertScaleAbs(img_enhanced, alpha=1.1, beta=5)
    img_enhanced = cv.bilateralFilter(img_enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced_b, img_enhanced_g, img_enhanced_r = cv.split(img_enhanced)
    img_enhanced_b = clahe.apply(img_enhanced_b)
    img_enhanced_g = clahe.apply(img_enhanced_g)
    img_enhanced_r = clahe.apply(img_enhanced_r)
    img_enhanced = cv.merge((img_enhanced_b, img_enhanced_g, img_enhanced_r))

    return Image.fromarray(img_enhanced)



def isolate_products(pil_img: Image, classes, boxes):
    for cl, (xmin, ymin, xmax, ymax), in zip(classes, boxes.tolist()):
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle(xy=((xmin, ymin), (xmax, ymax)))
        draw.text((xmin, ymin), cl, (0, 0, 255,), thickness=1.5)

    return pil_img


def PILhance(image: Image):
    background = image.copy()
    background = ImageEnhance.Brightness(background).enhance(0.5)
    mask = background.convert("L")

    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(1.5)

    enhanced_image.putalpha(mask)
    return enhanced_image


def filhance(image: Image):
    classes, bboxes = ImageFilter().filter(image)

    enhanced_imaged =  enhance_image(image) if len(classes) else image
    return isolate_products(enhanced_imaged, classes, bboxes)
    