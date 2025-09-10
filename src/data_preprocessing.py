"""
Data preprocessing utilities for crop disease detection
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from config import DATA_CONFIG, DISEASE_CLASSES


class DataPreprocessor:
    """Handle data loading and preprocessing for crop disease detection."""
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_CONFIG['data_dir']
        self.img_height = DATA_CONFIG['img_height']
        self.img_width = DATA_CONFIG['img_width']
        self.batch_size = DATA_CONFIG['batch_size']
        
    def load_image(self, image_path):
        """Load and preprocess a single image."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, (self.img_width, self.img_height))
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def create_dataset_from_directory(self, data_dir=None, validation_split=0.2):
        """Create TensorFlow datasets from directory structure."""
        if data_dir is None:
            data_dir = self.data_dir
            
        # Create training dataset
        train_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        
        # Create validation dataset
        val_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        
        # Get class names
        class_names = train_ds.class_names
        
        # Optimize datasets for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds, class_names
    
    def preprocess_image_for_prediction(self, image_path):
        """Preprocess a single image for prediction."""
        image = self.load_image(image_path)
        if image is not None:
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            return image
        return None
    
    def augment_image(self, image):
        """Apply data augmentation to an image."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        return image
    
    def load_and_prepare_data(self, data_dir=None):
        """Load data from directory structure and prepare for training."""
        if data_dir is None:
            data_dir = self.data_dir
            
        images = []
        labels = []
        
        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} does not exist.")
            print("Please download the PlantVillage dataset from:")
            print("https://www.kaggle.com/datasets/emmarex/plantdisease")
            return None, None, None
        
        # Iterate through each class directory
        for class_idx, class_name in enumerate(DISEASE_CLASSES):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                print(f"Loading images from {class_name}...")
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_file)
                        image = self.load_image(img_path)
                        if image is not None:
                            images.append(image)
                            labels.append(class_idx)
        
        if len(images) == 0:
            print("No images found in the specified directory structure.")
            return None, None, None
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Convert labels to categorical
        labels = keras.utils.to_categorical(labels, num_classes=len(DISEASE_CLASSES))
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        return X_train, X_val, y_train, y_val


def create_sample_data_structure():
    """Create sample directory structure for testing."""
    base_dir = "data/sample"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a few sample class directories
    sample_classes = ['Apple___healthy', 'Tomato___healthy', 'Potato___healthy']
    
    for class_name in sample_classes:
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Created directory: {class_dir}")
    
    print(f"Sample data structure created at {base_dir}")
    print("Please add your plant disease images to the respective directories.")