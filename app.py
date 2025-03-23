from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configure upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Path to the target logo
LOGO_PATH = os.path.join("static", "logo.png")

def extract_text_from_image(image):
    """
    Extract text from an image using pytesseract.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

def extract_images_from_pdf(pdf_path):
    """
    Extract images from a PDF file.
    """
    pdf_document = fitz.open(pdf_path)
    images = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]  # XREF of the image
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            images.append(image)

    return images

def compare_images_orb(image1, image2, threshold=10):
    """
    Compare two images using ORB feature matching.
    :param image1: Target logo (static/logo.png).
    :param image2: Extracted image from the PDF.
    :param threshold: Minimum number of good matches required to consider the images similar.
    :return: True if the images match, False otherwise.
    """
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # Check if descriptors are valid
    if des1 is None or des2 is None:
        return False  # No descriptors found in one or both images

    # Ensure descriptors have the same type and dimensions
    if des1.dtype != des2.dtype or des1.shape[1] != des2.shape[1]:
        return False

    # Use BFMatcher to match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Check if the number of good matches exceeds the threshold
    return len(matches) >= threshold

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            flash("No file uploaded!", "error")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No file selected!", "error")
            return redirect(request.url)

        if file:
            # Save the file
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Check if the file is a PDF
            if file.filename.endswith(".pdf"):
                # Extract images from the PDF
                images = extract_images_from_pdf(file_path)

                if not images:
                    flash("No images found in the PDF!", "error")
                    return redirect(request.url)

                # Extract text from the first image
                text = extract_text_from_image(images[0])
                print(f"Extracted Text: {text}")

                # Compare images with the target logo using ORB
                target_logo = cv2.imread(LOGO_PATH, cv2.IMREAD_COLOR)
                logo_found = False

                for image in images:
                    if compare_images_orb(target_logo, image):
                        logo_found = True
                        break

                if not logo_found:
                    flash("Logo not found in the PDF!", "error")
                    return redirect(request.url)

                flash("File uploaded successfully! Text extracted and logo verified.", "success")
                return redirect(request.url)

            else:
                flash("Unsupported file type! Please upload a PDF.", "error")
                return redirect(request.url)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)