import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def check_pdf_blur(pdf_path, threshold=100):
    images = convert_from_path(pdf_path)
    blurry_pages = []

    for i, image in enumerate(images):
        # Convert PIL image to numpy array
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        if is_blurry(image_cv, threshold):
            blurry_pages.append(i + 1)

    return blurry_pages

pdf_path = "res.pdf"
blurry_pages = check_pdf_blur(pdf_path)

if blurry_pages:
    print(f"Blurry pages: {blurry_pages}")
else:
    print("No blurry pages found.")
