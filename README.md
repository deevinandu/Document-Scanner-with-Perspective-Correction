# Document Scanner

This project is a Python-based document scanner that processes images to detect, warp, and enhance documents, extracting text using Optical Character Recognition (OCR). It uses OpenCV for image processing, Pytesseract for OCR, and Matplotlib for visualization.

## Features

- Detects document edges and contours.
- Applies perspective transformation to correct document orientation.
- Enhances documents with thresholding and filters.
- Extracts text using OCR.
- Visualizes processing steps.

## Requirements

- Python 3.x
- Google Colab (for file upload)
- Tesseract OCR

## Installation

1. Install Python packages:

```bash
pip install opencv-python opencv-python-headless pytesseract matplotlib pillow numpy
```

2. Install Tesseract OCR (Ubuntu):

```bash
sudo apt install tesseract-ocr
```

## Usage

1. Run in Google Colab.
2. Upload a document image.
3. View processed images (original, edges, scanned, filtered).
4. Check extracted text in the console.
5. Find the scanned document as `scanned_output.png`.

## Notes

- Use clear, well-lit images for best results.
- Modify file upload for non-Colab environments.
