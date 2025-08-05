"""
image_processing.py
====================

This script implements a basic image processing pipeline inspired by the
FastAIPhoto intelligent document processing system described in the
project.  It can automatically deskew, crop, denoise and remove
unwanted borders from scanned document images.  The script processes
all image files in a given input directory and writes the processed
images to an output directory, preserving the original filenames.

The implementation uses OpenCV and NumPy.  The deskew algorithm
computes the minimum‐area bounding rectangle of the text region to
estimate the document’s skew angle.  Cropping functions locate the
bounding box of the foreground region and remove any surrounding
padding or black edges.  Denoising is performed using median filtering
to remove speckle noise without significantly blurring text.  These
techniques are simple yet effective for many archival scanning tasks.

Example usage:

```
python image_processing.py --input input_folder --output output_folder
```

You can adjust individual steps using the provided command line
arguments.  See `python image_processing.py -h` for details.
"""

import argparse
import os
from typing import Tuple

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk using OpenCV.

    Args:
        path: Path to the image file.

    Returns:
        The loaded image in BGR format.
    """
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def save_image(path: str, image: np.ndarray) -> None:
    """Save an image to disk using OpenCV.

    Args:
        path: Destination file path.
        image: Image array in BGR format.
    """
    # Use imencode to support Unicode file paths on Windows
    ext = os.path.splitext(path)[1]
    success, buf = cv2.imencode(ext, image)
    if not success:
        raise ValueError(f"Failed to encode image for saving: {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def deskew_image(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Deskew an image by estimating and correcting its rotation angle.

    The function converts the image to grayscale, thresholds it to
    isolate the foreground, computes the minimum area bounding
    rectangle of the non-zero pixels and uses its angle to correct
    skew.  The corrected image is returned along with the angle in
    degrees.  A positive angle indicates a clockwise rotation was
    applied.

    Args:
        image: Input image (BGR).

    Returns:
        A tuple (rotated, angle) where `rotated` is the deskewed
        image and `angle` is the rotation angle in degrees.
    """
    # Convert to grayscale and invert so text becomes white on black background
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    # Apply Otsu's threshold to binarize the inverted image
    _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Extract coordinates of all non-zero pixels (foreground)
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        # Nothing detected, return original image
        return image.copy(), 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # The angle returned by minAreaRect is in the range [-90, 0);
    # adjust it to ensure a proper rotation direction.  Follow the
    # convention used by PyImageSearch: if the angle is less than -45
    # degrees, add 90 and negate; otherwise simply negate it.  This
    # produces a positive rotation angle (clockwise) that deskews the text.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # Rotate the image around its center to correct skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def crop_foreground(image: np.ndarray, margin: int = 0) -> np.ndarray:
    """Crop the image to the bounding box of its largest foreground contour.

    Many scanned images contain both dark and light borders depending on
    the scanner bed.  A simple threshold and bounding rectangle can
    fail on light backgrounds.  This implementation converts the
    image to grayscale, applies Otsu thresholding and finds the
    largest external contour.  The resulting bounding box is used to
    crop the image, optionally expanding by a margin.  This technique
    is inspired by open‑source tools that trim and deskew photos using
    Canny edges and contours (e.g. the trimpictures project)【57333697039207†L61-L104】.

    Args:
        image: Input image (BGR).
        margin: Extra pixels to include around the detected bounding box.

    Returns:
        Cropped image. If no contour is found, the original image is returned.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Otsu threshold to separate foreground from background.  We use
    # binary inversion so that dark text on light backgrounds becomes
    # foreground.  For dark backgrounds (white text), Otsu still
    # generates a reasonable separation.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find external contours of the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.copy()
    # Choose the largest contour by area; this should correspond to the
    # document region.  Smaller contours are likely noise or specks.
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    # Apply margin and clamp to image bounds
    x_start = max(x - margin, 0)
    y_start = max(y - margin, 0)
    x_end = min(x + w + margin, image.shape[1])
    y_end = min(y + h + margin, image.shape[0])
    return image[y_start:y_end, x_start:x_end]


def remove_black_borders(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Remove dark or light borders from the image.

    Scanned documents can have very dark borders (near 0 intensity) or
    very light borders (near 255 intensity) depending on the scanner
    background.  This function scans from each edge inward and crops
    away rows and columns whose average intensity is within a band of
    "background" values.  The default threshold of 10 means that
    values below 10 are considered black and values above 245 (255 - 10)
    are considered white borders【251769692126785†L0-L55】.

    Args:
        image: Input image (BGR).
        threshold: Intensity threshold for determining border pixels.  Values
            within [0, threshold] or [255 - threshold, 255] are treated as
            background.

    Returns:
        Cropped image without detected borders.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    min_thresh = threshold
    max_thresh = 255 - threshold
    # Scan from the top until a row has a mean intensity outside the background
    top = 0
    while top < h and (np.mean(gray[top, :]) <= min_thresh or np.mean(gray[top, :]) >= max_thresh):
        top += 1
    # Scan from the bottom
    bottom = h - 1
    while bottom > top and (np.mean(gray[bottom, :]) <= min_thresh or np.mean(gray[bottom, :]) >= max_thresh):
        bottom -= 1
    # Scan from the left
    left = 0
    while left < w and (np.mean(gray[:, left]) <= min_thresh or np.mean(gray[:, left]) >= max_thresh):
        left += 1
    # Scan from the right
    right = w - 1
    while right > left and (np.mean(gray[:, right]) <= min_thresh or np.mean(gray[:, right]) >= max_thresh):
        right -= 1
    # Clamp values to ensure we don't exceed bounds
    top = max(top, 0)
    bottom = min(bottom, h - 1)
    left = max(left, 0)
    right = min(right, w - 1)
    # Avoid negative or zero dimensions
    if right <= left or bottom <= top:
        return image.copy()
    return image[top:bottom + 1, left:right + 1]


def denoise_image(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply median filtering to reduce speckle noise.

    Args:
        image: Input image (BGR).
        ksize: Kernel size for the median filter; must be odd.

    Returns:
        Denoised image.
    """
    return cv2.medianBlur(image, ksize)


def auto_orient(image: np.ndarray, orientation: str = "landscape") -> np.ndarray:
    """Rotate the image by 90 degrees if its orientation does not match the desired one.

    Some scanned pages may be oriented incorrectly (e.g., portrait when you expect
    landscape or vice versa).  This helper checks the aspect ratio of the image
    and rotates by 90 degrees clockwise when necessary.  The default target
    orientation is ``landscape`` (width greater than height), but you can pass
    ``portrait`` to enforce the opposite.

    Args:
        image: Input image (BGR).
        orientation: Target orientation: either ``landscape`` or ``portrait``.

    Returns:
        Rotated image if rotation is needed; otherwise the original image.
    """
    h, w = image.shape[:2]
    if orientation == "landscape" and h > w:
        # Rotate 90° clockwise to make width > height
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == "portrait" and w > h:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def process_image(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Process a single image according to command line options.

    This function orchestrates deskewing, border removal, cropping and
    denoising in the order specified by the user.

    Args:
        image: Input image (BGR).
        args: Parsed command line arguments controlling the pipeline.

    Returns:
        The processed image.
    """
    processed = image.copy()
    # Optionally rotate the image to the desired orientation before other
    # operations.  This is useful when scans are accidentally rotated 90°.
    if getattr(args, 'auto_orient', None):
        processed = auto_orient(processed, orientation=args.auto_orient)
    if args.deskew:
        processed, angle = deskew_image(processed)
        if args.verbose:
            print(f"Deskewed by {angle:.2f} degrees")
    if args.remove_borders:
        processed = remove_black_borders(processed, threshold=args.border_threshold)
    if args.crop:
        processed = crop_foreground(processed, margin=args.crop_margin)
    if args.denoise:
        processed = denoise_image(processed, ksize=args.denoise_ksize)
    return processed


def process_directory(input_dir: str, output_dir: str, args: argparse.Namespace) -> None:
    """Process all images in a directory.

    Args:
        input_dir: Directory containing images to process.
        output_dir: Directory to save processed images.
        args: Parsed command line options.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        lower = filename.lower()
        if not any(lower.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            continue
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        if args.verbose:
            print(f"Processing {filename}...")
        image = load_image(input_path)
        processed = process_image(image, args)
        save_image(output_path, processed)
        if args.verbose:
            print(f"Saved result to {output_path}")


def build_argparser() -> argparse.ArgumentParser:
    """Create a command line argument parser."""
    parser = argparse.ArgumentParser(description="Batch process scanned document images with deskewing, cropping and denoising.")
    parser.add_argument('--input', required=True, help='Input directory containing images')
    parser.add_argument('--output', required=True, help='Output directory for processed images')
    parser.add_argument('--deskew', action='store_true', help='Perform automatic deskewing')
    parser.add_argument('--crop', action='store_true', help='Crop to foreground bounding box')
    parser.add_argument('--crop-margin', type=int, default=0, help='Margin (pixels) to add around cropped bounding box')
    parser.add_argument('--remove-borders', action='store_true', help='Remove black scanner borders')
    parser.add_argument('--border-threshold', type=int, default=10, help='Intensity threshold for border removal')
    parser.add_argument('--denoise', action='store_true', help='Apply median filter for denoising')
    parser.add_argument('--denoise-ksize', type=int, default=3, help='Kernel size for median filtering (odd)')
    parser.add_argument('--auto-orient', choices=['landscape', 'portrait'], help='Automatically rotate images to the specified orientation (landscape or portrait)')
    parser.add_argument('--verbose', action='store_true', help='Print progress information')
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    process_directory(args.input, args.output, args)


if __name__ == '__main__':
    main()