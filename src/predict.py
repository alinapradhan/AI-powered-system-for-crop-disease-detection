"""
Prediction and inference for crop disease detection
"""

import os
import sys
import numpy as np
import cv2
import json
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from config import DISEASE_CLASSES, MODEL_CONFIG


class DiseasePredictor:
    """Handle prediction and inference for crop disease detection."""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or 'models/crop_disease_model.h5'
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.disease_classes = DISEASE_CLASSES
        
    def load_model(self):
        """Load the trained model."""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
                return True
            else:
                print(f"Model file not found at {self.model_path}")
                print("Please train the model first using train.py")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_image(self, image_path, top_k=3):
        """Predict disease for a single image."""
        if self.model is None:
            if not self.load_model():
                return None
        
        # Preprocess the image
        processed_image = self.preprocessor.preprocess_image_for_prediction(image_path)
        if processed_image is None:
            print(f"Error preprocessing image: {image_path}")
            return None
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get top-k predictions
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            disease_name = self.disease_classes[idx]
            confidence = float(predictions[0][idx])
            results.append({
                'disease': disease_name,
                'confidence': confidence,
                'percentage': confidence * 100
            })
        
        return results
    
    def predict_batch(self, image_paths):
        """Predict diseases for multiple images."""
        if self.model is None:
            if not self.load_model():
                return None
        
        results = {}
        for image_path in image_paths:
            prediction = self.predict_image(image_path)
            results[image_path] = prediction
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction results."""
        # Get prediction
        prediction = self.predict_image(image_path)
        if prediction is None:
            return
        
        # Load and display image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        # Display predictions
        plt.subplot(1, 2, 2)
        diseases = [p['disease'].replace('___', ' - ').replace('_', ' ') for p in prediction]
        confidences = [p['percentage'] for p in prediction]
        
        bars = plt.barh(diseases, confidences)
        plt.xlabel('Confidence (%)')
        plt.title('Top 3 Predictions')
        plt.xlim(0, 100)
        
        # Color bars based on confidence
        for i, bar in enumerate(bars):
            if confidences[i] > 70:
                bar.set_color('green')
            elif confidences[i] > 50:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add percentage labels
        for i, (disease, conf) in enumerate(zip(diseases, confidences)):
            plt.text(conf + 1, i, f'{conf:.1f}%', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def get_disease_info(self, disease_name):
        """Get information about a specific disease."""
        # Parse disease name
        parts = disease_name.split('___')
        if len(parts) == 2:
            plant, condition = parts
            plant = plant.replace('_', ' ').title()
            condition = condition.replace('_', ' ').title()
            
            if condition.lower() == 'healthy':
                return {
                    'plant': plant,
                    'condition': 'Healthy',
                    'description': f'The {plant.lower()} plant appears to be healthy with no visible signs of disease.',
                    'recommendations': [
                        'Continue regular monitoring',
                        'Maintain proper watering schedule',
                        'Ensure adequate nutrition',
                        'Monitor for early signs of disease'
                    ]
                }
            else:
                return {
                    'plant': plant,
                    'condition': condition,
                    'description': f'{condition} detected in {plant.lower()} plant.',
                    'recommendations': [
                        'Consult with agricultural extension services',
                        'Consider appropriate fungicide/pesticide treatment',
                        'Improve ventilation and reduce humidity',
                        'Remove affected plant parts if necessary',
                        'Monitor surrounding plants for spread'
                    ]
                }
        
        return {
            'plant': 'Unknown',
            'condition': disease_name,
            'description': 'Disease information not available.',
            'recommendations': ['Consult with agricultural experts']
        }
    
    def generate_report(self, image_path, output_path=None):
        """Generate a detailed prediction report."""
        prediction = self.predict_image(image_path)
        if prediction is None:
            return None
        
        # Get the top prediction
        top_prediction = prediction[0]
        disease_info = self.get_disease_info(top_prediction['disease'])
        
        report = {
            'image_path': image_path,
            'timestamp': str(np.datetime64('now')),
            'top_prediction': {
                'disease': top_prediction['disease'],
                'confidence': top_prediction['confidence'],
                'percentage': top_prediction['percentage']
            },
            'all_predictions': prediction,
            'disease_info': disease_info,
            'recommendations': disease_info['recommendations']
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_path}")
        
        return report


def main():
    """Main prediction function for testing."""
    predictor = DiseasePredictor()
    
    # Test with sample image (if exists)
    sample_image = "data/sample/test_image.jpg"
    
    if os.path.exists(sample_image):
        print(f"Testing prediction with {sample_image}")
        
        # Make prediction
        prediction = predictor.predict_image(sample_image)
        if prediction:
            print("\nPrediction Results:")
            for i, pred in enumerate(prediction, 1):
                print(f"{i}. {pred['disease']} - {pred['percentage']:.2f}%")
            
            # Generate visualization
            predictor.visualize_prediction(sample_image, "results/prediction_result.png")
            
            # Generate report
            report = predictor.generate_report(sample_image, "results/prediction_report.json")
            print("\nDetailed report generated.")
    else:
        print(f"Sample image not found at {sample_image}")
        print("Please provide an image path to test prediction.")
        print("\nUsage example:")
        print("python predict.py path/to/your/image.jpg")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            predictor = DiseasePredictor()
            predictor.visualize_prediction(image_path)
        else:
            print(f"Image file not found: {image_path}")
    else:
        main()