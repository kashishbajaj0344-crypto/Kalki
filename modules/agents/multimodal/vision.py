from typing import Dict, Any
import logging
import os
from PIL import Image
import numpy as np

from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = logging.getLogger(__name__)


class VisionAgent(BaseAgent):
    """Visual processing and analysis agent using PIL and rule-based analysis"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="VisionAgent",
            capabilities=[AgentCapability.VISION],
            description="Processes and analyzes visual information using PIL",
            config=config or {}
        )

    async def initialize(self) -> bool:
        try:
            # Test PIL import
            from PIL import Image
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        action = task.get("action")
        params = task.get("params", {})

        if action == "analyze":
            return await self._analyze_image(params)
        elif action == "detect":
            return await self._detect_objects(params)
        elif action == "classify":
            return await self._classify_image(params)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _analyze_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image_path = params.get("image_path", "")
            if not os.path.exists(image_path):
                return {"status": "error", "error": f"Image file not found: {image_path}"}

            # Load and analyze image
            img = Image.open(image_path)
            img_array = np.array(img)

            # Basic analysis
            analysis = self._rule_based_image_analysis(img, img_array)

            return {"status": "success", "image_path": image_path, "analysis": analysis}
        except Exception as e:
            self.logger.exception(f"Image analysis error: {e}")
            return {"status": "error", "error": str(e)}

    def _rule_based_image_analysis(self, img: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Rule-based image analysis using basic image properties"""
        width, height = img.size
        mode = img.mode

        # Color analysis
        if mode == 'RGB':
            dominant_colors = self._extract_dominant_colors(img_array)
        else:
            dominant_colors = ["grayscale"]

        # Composition analysis based on aspect ratio and basic heuristics
        aspect_ratio = width / height
        if aspect_ratio > 1.5:
            composition = "wide landscape"
        elif aspect_ratio < 0.67:
            composition = "tall portrait"
        else:
            composition = "balanced"

        # Simple feature detection based on color variance and edges
        features_detected = []
        if self._has_high_contrast(img_array):
            features_detected.append("high_contrast")
        if self._has_color_variance(img_array):
            features_detected.append("colorful")
        if self._detects_edges(img_array):
            features_detected.append("structured")

        # Brightness analysis
        brightness = np.mean(img_array) / 255.0
        if brightness > 0.7:
            lighting = "bright"
        elif brightness < 0.3:
            lighting = "dark"
        else:
            lighting = "moderate"

        return {
            "description": f"{mode} image, {width}x{height}, {composition} composition",
            "dominant_colors": dominant_colors,
            "composition": composition,
            "features_detected": features_detected,
            "lighting": lighting,
            "dimensions": {"width": width, "height": height},
            "aspect_ratio": round(aspect_ratio, 2)
        }

    def _extract_dominant_colors(self, img_array: np.ndarray, num_colors: int = 3) -> list:
        """Extract dominant colors using k-means clustering approximation"""
        # Simple approach: sample pixels and find most common colors
        pixels = img_array.reshape(-1, 3)
        # Sample a subset of pixels for performance
        sample_size = min(1000, len(pixels))
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[indices]

        # Simple color quantization
        colors = []
        for i in range(num_colors):
            # Find the most frequent color in remaining pixels
            unique_colors, counts = np.unique(sample_pixels.view(dtype=[('r', 'u1'), ('g', 'u1'), ('b', 'u1')]),
                                            return_counts=True)
            if len(unique_colors) > 0:
                most_common = unique_colors[np.argmax(counts)]
                colors.append(f"rgb({most_common['r']},{most_common['g']},{most_common['b']})")
                # Remove similar colors for next iteration
                mask = np.sum(np.abs(sample_pixels - [most_common['r'], most_common['g'], most_common['b']]), axis=1) > 30
                sample_pixels = sample_pixels[mask]
                if len(sample_pixels) < 10:
                    break

        return colors if colors else ["unknown"]

    def _has_high_contrast(self, img_array: np.ndarray) -> bool:
        """Detect high contrast based on standard deviation"""
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        return np.std(gray) > 50

    def _has_color_variance(self, img_array: np.ndarray) -> bool:
        """Detect color variance in RGB images"""
        if len(img_array.shape) == 3:
            std_per_channel = np.std(img_array, axis=(0, 1))
            return np.mean(std_per_channel) > 30
        return False

    def _detects_edges(self, img_array: np.ndarray) -> bool:
        """Simple edge detection using gradient magnitude"""
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Simple sobel-like gradient
        dx = np.abs(np.gradient(gray, axis=1))
        dy = np.abs(np.gradient(gray, axis=0))
        gradient_magnitude = np.sqrt(dx**2 + dy**2)

        return np.mean(gradient_magnitude) > 20

    async def _detect_objects(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image_path = params.get("image_path", "")
            if not os.path.exists(image_path):
                return {"status": "error", "error": f"Image file not found: {image_path}"}

            # Load image
            img = Image.open(image_path)
            img_array = np.array(img)

            # Rule-based object detection using color and shape heuristics
            detections = self._rule_based_object_detection(img_array)

            return {"status": "success", "image_path": image_path, "detections": detections, "count": len(detections)}
        except Exception as e:
            self.logger.exception(f"Object detection error: {e}")
            return {"status": "error", "error": str(e)}

    def _rule_based_object_detection(self, img_array: np.ndarray) -> list:
        """Rule-based object detection using color and shape analysis"""
        detections = []

        # Convert to HSV for better color analysis
        if len(img_array.shape) == 3:
            hsv = self._rgb_to_hsv(img_array)

            # Detect skin-colored regions (simple face detection heuristic)
            skin_mask = self._detect_skin_regions(hsv)
            if np.sum(skin_mask) > 100:  # Minimum area threshold
                detections.append({
                    "class": "person",
                    "confidence": 0.7,
                    "bbox": self._mask_to_bbox(skin_mask)
                })

            # Detect blue regions (possible cars, sky, water)
            blue_mask = (hsv[:, :, 0] > 0.55) & (hsv[:, :, 0] < 0.7) & (hsv[:, :, 1] > 0.3)
            if np.sum(blue_mask) > 200:
                detections.append({
                    "class": "vehicle",
                    "confidence": 0.6,
                    "bbox": self._mask_to_bbox(blue_mask)
                })

            # Detect green regions (vegetation)
            green_mask = (hsv[:, :, 0] > 0.2) & (hsv[:, :, 0] < 0.4) & (hsv[:, :, 1] > 0.2)
            if np.sum(green_mask) > 300:
                detections.append({
                    "class": "vegetation",
                    "confidence": 0.8,
                    "bbox": self._mask_to_bbox(green_mask)
                })

        return detections

    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV color space"""
        rgb_norm = rgb.astype(float) / 255.0
        hsv = np.zeros_like(rgb_norm)

        for i in range(rgb_norm.shape[0]):
            for j in range(rgb_norm.shape[1]):
                r, g, b = rgb_norm[i, j]
                max_val = max(r, g, b)
                min_val = min(r, g, b)
                diff = max_val - min_val

                # Hue
                if diff == 0:
                    h = 0
                elif max_val == r:
                    h = (60 * ((g - b) / diff) + 360) % 360
                elif max_val == g:
                    h = (60 * ((b - r) / diff) + 120) % 360
                else:
                    h = (60 * ((r - g) / diff) + 240) % 360
                hsv[i, j, 0] = h / 360.0  # Normalize to 0-1

                # Saturation
                if max_val == 0:
                    s = 0
                else:
                    s = diff / max_val
                hsv[i, j, 1] = s

                # Value
                hsv[i, j, 2] = max_val

        return hsv

    def _detect_skin_regions(self, hsv: np.ndarray) -> np.ndarray:
        """Detect skin-colored regions in HSV space"""
        # Skin color ranges in HSV (rough approximation)
        h_mask = (hsv[:, :, 0] > 0.05) & (hsv[:, :, 0] < 0.15)  # Hue range for skin
        s_mask = hsv[:, :, 1] > 0.2  # Minimum saturation
        v_mask = (hsv[:, :, 2] > 0.3) & (hsv[:, :, 2] < 0.9)  # Value range

        return h_mask & s_mask & v_mask

    def _mask_to_bbox(self, mask: np.ndarray) -> list:
        """Convert binary mask to bounding box [x1, y1, x2, y2]"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return [0, 0, 0, 0]

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return [int(x1), int(y1), int(x2), int(y2)]

    async def _classify_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image_path = params.get("image_path", "")
            if not os.path.exists(image_path):
                return {"status": "error", "error": f"Image file not found: {image_path}"}

            # Load image
            img = Image.open(image_path)
            img_array = np.array(img)

            # Rule-based classification
            classification = self._rule_based_classification(img, img_array)

            return {"status": "success", "image_path": image_path, "classification": classification}
        except Exception as e:
            self.logger.exception(f"Image classification error: {e}")
            return {"status": "error", "error": str(e)}

    def _rule_based_classification(self, img: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Rule-based image classification using heuristics"""
        width, height = img.size
        aspect_ratio = width / height

        # Initialize scores for different categories
        scores = {
            "landscape": 0,
            "portrait": 0,
            "document": 0,
            "indoor": 0,
            "nature": 0,
            "urban": 0
        }

        # Aspect ratio hints
        if aspect_ratio > 1.5:
            scores["landscape"] += 2
        elif aspect_ratio < 0.67:
            scores["portrait"] += 2

        # Color analysis
        if len(img_array.shape) == 3:
            hsv = self._rgb_to_hsv(img_array)

            # High green content suggests nature
            green_pixels = np.sum((hsv[:, :, 0] > 0.2) & (hsv[:, :, 0] < 0.4) & (hsv[:, :, 1] > 0.2))
            if green_pixels > img_array.shape[0] * img_array.shape[1] * 0.3:
                scores["nature"] += 3
                scores["landscape"] += 1

            # High blue content might be sky/water
            blue_pixels = np.sum((hsv[:, :, 0] > 0.55) & (hsv[:, :, 0] < 0.7) & (hsv[:, :, 1] > 0.3))
            if blue_pixels > img_array.shape[0] * img_array.shape[1] * 0.2:
                scores["landscape"] += 2

            # Low saturation and high value might indicate document/scanned image
            low_sat_pixels = np.sum(hsv[:, :, 1] < 0.2)
            if low_sat_pixels > img_array.shape[0] * img_array.shape[1] * 0.8:
                scores["document"] += 3

        # Contrast analysis
        if self._has_high_contrast(img_array):
            scores["landscape"] += 1  # High contrast often in outdoor scenes
        else:
            scores["indoor"] += 1  # Low contrast might be indoor

        # Find top categories
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_5 = [{"class": cls, "confidence": min(score / 5.0, 1.0)} for cls, score in sorted_scores[:5]]

        return {
            "primary_class": top_5[0]["class"] if top_5 else "unknown",
            "confidence": top_5[0]["confidence"] if top_5 else 0.0,
            "top_5": top_5
        }

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True
