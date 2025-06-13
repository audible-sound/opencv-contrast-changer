import cv2
import numpy as np

#Contrast Limited Adaptive Histogram Equalization
def clahe_contrast(frame):
    # Convert to LAB color space for better results
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# Global Histogram Equalization
def histogram_contrast(frame):
    # Convert to YUV and equalize Y channel
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# Linear contrast streching
def linear_contrast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_val, max_val = np.min(gray), np.max(gray)
    
    # Stretch contrast to full range
    stretched = ((frame.astype(np.float32) - min_val) / 
                (max_val - min_val) * 255).astype(np.uint8)
    return stretched

# Gamma correction
def gamma_contrast(frame):
    gamma = 0.5 
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

# Binary colours
def binary_contrast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

contrast_methods = {
    "clahe": clahe_contrast,
    "histogram": histogram_contrast,
    "linear": linear_contrast,
    "gamma": gamma_contrast,
    "binary": binary_contrast
}

# Apply contrast to frame
def apply_contrast(frame, method):
    if method in contrast_methods:
        return contrast_methods[method](frame)
    return frame