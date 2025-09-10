"""
CNN Model for Crop Disease Detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import MODEL_CONFIG


class CropDiseaseModel:
    """Convolutional Neural Network for crop disease classification."""
    
    def __init__(self):
        self.model = None
        self.input_shape = MODEL_CONFIG['input_shape']
        self.num_classes = MODEL_CONFIG['num_classes']
        
    def build_model(self):
        """Build the CNN architecture."""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Data augmentation layers
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Rescaling
            layers.Rescaling(1./255),
            
            # First convolutional block
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=None):
        """Compile the model with optimizer, loss, and metrics."""
        if learning_rate is None:
            learning_rate = MODEL_CONFIG['learning_rate']
            
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
    def get_model_summary(self):
        """Get model summary."""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Please build and train the model first.")
    
    def load_model(self, filepath):
        """Load a pre-trained model."""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


def create_transfer_learning_model():
    """Create a model using transfer learning with MobileNetV2."""
    base_model = keras.applications.MobileNetV2(
        input_shape=MODEL_CONFIG['input_shape'],
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(MODEL_CONFIG['num_classes'], activation='softmax')
    ])
    
    return model