import cv2
import numpy as np
import os
from skimage.segmentation import slic
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans


def compress_hex_color(hex_color):
    """Convert hex color to shortest possible representation"""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
        return f'#{r//17:x}{g//17:x}{b//17:x}'
    return hex_color


# Quantization methods
def quantize_kmeans(img_rgb, num_colors=16):
    """Standard K-means quantization"""
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Quantized image
    palette = centers.astype(np.uint8)
    quantized = palette[labels.flatten()].reshape(img_rgb.shape)
    
    return quantized, labels.flatten(), palette

def quantize_slic(img_rgb, num_colors=16, compactness=10):
    """SLIC superpixel-based quantization"""
    # Generate superpixels
    segments = slic(img_rgb, n_segments=num_colors*2, compactness=compactness)
    
    # Create a new image with the mean color of each segment
    result = np.zeros_like(img_rgb)
    palette = []
    labels_map = {}
    
    for i, segment_id in enumerate(np.unique(segments)):
        mask = segments == segment_id
        mean_color = np.mean(img_rgb[mask], axis=0).astype(np.uint8)
        result[mask] = mean_color
        palette.append(mean_color)
        labels_map[segment_id] = i
    
    # Convert segmentation to flat labels array
    labels = np.array([labels_map[s] for s in segments.flatten()])
    palette = np.array(palette)
    
    # Further quantize colors if needed
    if len(palette) > num_colors:
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        color_labels = kmeans.fit_predict(palette)
        new_palette = kmeans.cluster_centers_.astype(np.uint8)
        
        # Map labels through the two-step quantization
        labels = color_labels[labels]
        palette = new_palette
        
        # Update the quantized image
        result = palette[labels].reshape(img_rgb.shape)
    
    return result, labels, palette

def quantize_bilateral_kmeans(img_rgb, num_colors=16, d=15, sigma_color=75, sigma_space=75):
    """Bilateral filtering followed by K-means"""
    # Apply bilateral filter for edge-preserving smoothing
    smoothed = cv2.bilateralFilter(img_rgb, d, sigma_color, sigma_space)
    
    # Apply standard K-means to the smoothed image
    pixels = smoothed.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Quantized image
    palette = centers.astype(np.uint8)
    quantized = palette[labels.flatten()].reshape(img_rgb.shape)
    
    return quantized, labels.flatten(), palette

def quantize_meanshift(img_rgb, num_colors=16, quantile=0.2):
    """Mean Shift clustering for color quantization"""
    # Reshape image for clustering
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    
    # Estimate bandwidth
    bandwidth = estimate_bandwidth(pixels, quantile=quantile, n_samples=1000)
    
    # Apply Mean Shift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(pixels)
    centers = ms.cluster_centers_
    
    # If we have too many clusters, merge similar ones
    if len(centers) > num_colors:
        # Apply K-means to the Mean Shift centers
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        center_labels = kmeans.fit_predict(centers)
        new_centers = kmeans.cluster_centers_.astype(np.uint8)
        
        # Map the labels
        labels = center_labels[labels]
        centers = new_centers
    
    # Quantized image
    palette = centers.astype(np.uint8)
    quantized = palette[labels].reshape(img_rgb.shape)
    
    return quantized, labels, palette

def extract_features_by_scale(img_np, num_colors=16, method='kmeans', method_params=None):
    """
    Extract image features hierarchically by scale
    
    Args:
        img_np (np.ndarray): Input image
        num_colors (int): Number of colors to quantize
        method (str): Quantization method ('kmeans', 'slic', 'bilateral_kmeans', 'meanshift')
        method_params (dict): Additional parameters for the chosen method
    
    Returns:
        list: Hierarchical features sorted by importance
    """
    os.makedirs('debug_display', exist_ok=True)
    
    # Convert to RGB if needed
    if len(img_np.shape) == 3 and img_np.shape[2] > 1:
        img_rgb = img_np
    else:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    cv2.imwrite('debug_display/01_original.jpg', img_rgb)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    
    cv2.imwrite('debug_display/02_grayscale.jpg', gray)
    
    # Initialize default method parameters if not provided
    if method_params is None:
        method_params = {}
    
    # Perform color quantization based on selected method
    if method == 'kmeans':
        quantized, labels, palette = quantize_kmeans(img_rgb, num_colors)
    elif method == 'slic':
        compactness = method_params.get('compactness', 10)
        quantized, labels, palette = quantize_slic(img_rgb, num_colors, compactness)
    elif method == 'bilateral_kmeans':
        d = method_params.get('d', 15)
        sigma_color = method_params.get('sigma_color', 75)
        sigma_space = method_params.get('sigma_space', 75)
        quantized, labels, palette = quantize_bilateral_kmeans(img_rgb, num_colors, d, sigma_color, sigma_space)
    elif method == 'meanshift':
        quantile = method_params.get('quantile', 0.2)
        quantized, labels, palette = quantize_meanshift(img_rgb, num_colors, quantile)
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    cv2.imwrite('debug_display/03_quantized.jpg', quantized)
    
    # Hierarchical feature extraction
    hierarchical_features = []
    
    # Sort colors by frequency
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_colors = [palette[i] for i in sorted_indices]
    
    # Center point for importance calculations
    center_x, center_y = width/2, height/2
    
    contour_img = img_rgb.copy()
    simplified_contour_img = img_rgb.copy()
    
    for idx, color in enumerate(sorted_colors):
        # Create color mask
        color_mask = cv2.inRange(quantized, color, color)
        
        cv2.imwrite(f'debug_display/04_color_mask_{idx}.jpg', color_mask)
        
        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Convert RGB to compressed hex
        hex_color = compress_hex_color(f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')
        
        color_features = []
        for contour_idx, contour in enumerate(contours):
            # Skip tiny contours
            area = cv2.contourArea(contour)
            if area < 20:
                continue
            
            # Calculate contour center
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            
            # Distance from image center (normalized)
            dist_from_center = np.sqrt(((cx - center_x) / width)**2 + ((cy - center_y) / height)**2)
            
            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            cv2.drawContours(simplified_contour_img, [approx], -1, (0, 0, 255), 2)
            
            # Generate points string
            points = " ".join([f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx])
            
            # Calculate importance (area, proximity to center, complexity)
            importance = (
                area * 
                (1 - dist_from_center) * 
                (1 / (len(approx) + 1))
            )
            
            color_features.append({
                'points': points,
                'color': hex_color,
                'area': area,
                'importance': importance,
                'point_count': len(approx),
                'original_contour': approx  # Store original contour for adaptive simplification
            })
        
        # Sort features by importance within this color
        color_features.sort(key=lambda x: x['importance'], reverse=True)
        hierarchical_features.extend(color_features)
    
    cv2.imwrite('debug_display/05_all_contours.jpg', contour_img)
    cv2.imwrite('debug_display/06_simplified_contours.jpg', simplified_contour_img)
    
    # Final sorting by overall importance
    hierarchical_features.sort(key=lambda x: x['importance'], reverse=True)
    
    importance_img = np.zeros_like(img_rgb)
    for idx, feature in enumerate(hierarchical_features[:20]):
        cv2.drawContours(importance_img, [feature['original_contour']], -1, (255, 255, 255), -1)
        cv2.putText(importance_img, str(idx+1), (int(feature['original_contour'][0][0][0]), 
                                              int(feature['original_contour'][0][0][1])), 
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite('debug_display/07_importance_ranking.jpg', importance_img)
    
    return hierarchical_features


if __name__ == "__main__":
    img = cv2.imread("/home/mpf/Downloads/r1.jpg")
    img = cv2.resize(img, (384, 384))
    
    # Example usage with different methods:
    # Default K-means
    # res = extract_features_by_scale(img, num_colors=17)
    
    # SLIC superpixels
    # res = extract_features_by_scale(img, num_colors=17, method='slic', method_params={'compactness': 20})
    
    # Bilateral + K-means
    # res = extract_features_by_scale(img, num_colors=17, method='bilateral_kmeans', 
    #                              method_params={'d': 15, 'sigma_color': 75, 'sigma_space': 75})
    
    # Mean Shift
    res = extract_features_by_scale(img, num_colors=16, method='bilateral_kmeans', method_params={'quantile': 0.15})


