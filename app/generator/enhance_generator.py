from PIL import Image
import numpy as np
import cv2 as cv


def enhance_image(image: Image):
    img = np.array(image)

    img_enhanced = img.copy()
    img_enhanced = cv.convertScaleAbs(img_enhanced, alpha=1.1, beta=5)
    img_enhanced = cv.bilateralFilter(img_enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced_b, img_enhanced_g, img_enhanced_r = cv2.split(img_enhanced)
    img_enhanced_b = clahe.apply(img_enhanced_b)
    img_enhanced_g = clahe.apply(img_enhanced_g)
    img_enhanced_r = clahe.apply(img_enhanced_r)
    img_enhanced = cv.merge((img_enhanced_b, img_enhanced_g, img_enhanced_r))

    return Image.fromarray(img_enhanced)