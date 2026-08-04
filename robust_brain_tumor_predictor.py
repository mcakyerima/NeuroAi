
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import json
from PIL import Image
import io

class RobustBrainTumorPredictor:
    def __init__(self, model_path='best_brain_tumor_model.keras', metadata_path='enhanced_model_metadata.json'):
        """
        Robust brain tumor predictor with advanced preprocessing
        Handles real-world images, camera captures, and web images
        """
        self.model = tf.keras.models.load_model(model_path)

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.class_names = self.metadata['class_names']
        self.confidence_threshold = 0.7  # Minimum confidence for reliable prediction

    def advanced_preprocess(self, image_input, target_size=(224, 224)):
        """
        Advanced preprocessing pipeline for robust prediction
        Handles various image sources and qualities
        """
        # Handle different input types
        if isinstance(image_input, str):
            # File path
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image from {image_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            # Numpy array
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image = image_input.copy()
            else:
                raise ValueError("Invalid image array shape")
        else:
            # PIL Image or file-like object
            if hasattr(image_input, 'read'):
                image_input.seek(0)
                pil_image = Image.open(image_input).convert('RGB')
            else:
                pil_image = image_input.convert('RGB')
            image = np.array(pil_image)

        # Enhanced preprocessing pipeline
        original_shape = image.shape[:2]

        # 1. Quality enhancement
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # 2. Noise reduction (especially important for camera/phone images)
        image = cv2.bilateralFilter(image, 9, 75, 75)

        # 3. Contrast enhancement using CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        # 4. Smart cropping to focus on brain region
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Find the largest contour (likely the brain)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get bounding box of largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)

            # Crop to brain region
            if w > 50 and h > 50:  # Ensure minimum size
                image = image[y:y+h, x:x+w]

        # 5. Resize with aspect ratio preservation
        h, w = image.shape[:2]
        if h != w:
            # Make square by padding
            size = max(h, w)
            delta_w = size - w
            delta_h = size - h
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

        # 6. Resize to target size with high-quality interpolation
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

        # 7. Normalize
        image = image.astype(np.float32) / 255.0

        return image

    def predict(self, image_input, return_all_probs=False):
        """
        Make prediction with confidence assessment
        """
        try:
            # Preprocess image
            processed_image = self.advanced_preprocess(image_input)
            img_array = np.expand_dims(processed_image, axis=0)

            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)

            # Extract results
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]

            # Confidence assessment
            is_reliable = confidence >= self.confidence_threshold

            # All probabilities
            all_probabilities = {name: float(prob) for name, prob in
                               zip(self.class_names, predictions[0])}

            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'is_reliable': is_reliable,
                'confidence_level': self._get_confidence_level(confidence),
                'all_probabilities': all_probabilities
            }

            if return_all_probs:
                result['raw_predictions'] = predictions[0]

            return result

        except Exception as e:
            return {
                'error': str(e),
                'predicted_class': 'error',
                'confidence': 0.0,
                'is_reliable': False
            }

    def _get_confidence_level(self, confidence):
        """Categorize confidence level"""
        if confidence >= 0.95:
            return "Very High"
        elif confidence >= 0.85:
            return "High"
        elif confidence >= 0.70:
            return "Medium"
        elif confidence >= 0.50:
            return "Low"
        else:
            return "Very Low"

    def batch_predict(self, image_paths):
        """Predict on multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append(result)
        return results

# Usage examples:
# predictor = RobustBrainTumorPredictor()
#
# # For file path
# result = predictor.predict('brain_scan.jpg')
#
# # For uploaded file in Flask
# result = predictor.predict(request.files['image'])
#
# # For numpy array (camera capture)
# result = predictor.predict(camera_image_array)
#
# print(f"Prediction: {result['predicted_class']}")
# print(f"Confidence: {result['confidence']*100:.1f}% ({result['confidence_level']})")
# print(f"Reliable: {result['is_reliable']}")
