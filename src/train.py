"""
Training script for crop disease detection model
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import CropDiseaseModel, create_transfer_learning_model
from data_preprocessing import DataPreprocessor
from config import MODEL_CONFIG, TRAINING_CONFIG, DISEASE_CLASSES


class ModelTrainer:
    """Handle model training and evaluation."""
    
    def __init__(self, use_transfer_learning=False):
        self.use_transfer_learning = use_transfer_learning
        self.model = None
        self.history = None
        self.preprocessor = DataPreprocessor()
        
    def setup_model(self):
        """Initialize and compile the model."""
        if self.use_transfer_learning:
            print("Using transfer learning with MobileNetV2...")
            self.model = create_transfer_learning_model()
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            print("Using custom CNN architecture...")
            cnn_model = CropDiseaseModel()
            self.model = cnn_model.build_model()
            cnn_model.compile_model()
            self.model = cnn_model.model
    
    def setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=TRAINING_CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                monitor='val_loss'
            ),
            keras.callbacks.ReduceLROnPlateau(
                patience=TRAINING_CONFIG['reduce_lr_patience'],
                factor=0.5,
                monitor='val_loss'
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=TRAINING_CONFIG['checkpoint_path'] + '/model_{epoch:02d}_{val_accuracy:.2f}.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Create checkpoint directory
        os.makedirs(TRAINING_CONFIG['checkpoint_path'], exist_ok=True)
        
        return callbacks
    
    def train_with_datasets(self, train_ds, val_ds):
        """Train model using TensorFlow datasets."""
        print("Starting training...")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train the model
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=MODEL_CONFIG['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        self.model.save(TRAINING_CONFIG['model_save_path'])
        print(f"Model saved to {TRAINING_CONFIG['model_save_path']}")
        
        return self.history
    
    def train_with_arrays(self, X_train, y_train, X_val, y_val):
        """Train model using numpy arrays."""
        print("Starting training with arrays...")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=MODEL_CONFIG['epochs'],
            batch_size=MODEL_CONFIG['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        os.makedirs(os.path.dirname(TRAINING_CONFIG['model_save_path']), exist_ok=True)
        self.model.save(TRAINING_CONFIG['model_save_path'])
        print(f"Model saved to {TRAINING_CONFIG['model_save_path']}")
        
        return self.history
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        else:
            plt.savefig('results/training_history.png')
            print("Training history plot saved to results/training_history.png")
        
        plt.show()
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance."""
        if self.model is None:
            print("No model available for evaluation.")
            return
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=DISEASE_CLASSES))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png')
        print("Confusion matrix saved to results/confusion_matrix.png")
        plt.show()


def main():
    """Main training function."""
    print("Crop Disease Detection - Model Training")
    print("=" * 50)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer(use_transfer_learning=True)
    trainer.setup_model()
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    
    # Try to load data from directory structure first
    try:
        train_ds, val_ds, class_names = preprocessor.create_dataset_from_directory()
        print(f"Found {len(class_names)} classes: {class_names}")
        
        # Train the model
        history = trainer.train_with_datasets(train_ds, val_ds)
        
        # Plot training history
        trainer.plot_training_history()
        
    except Exception as e:
        print(f"Error loading data from directory: {e}")
        print("\nPlease ensure the data is organized in the following structure:")
        print("data/plantvillage/")
        print("├── Apple___Apple_scab/")
        print("├── Apple___Black_rot/")
        print("├── Apple___Cedar_apple_rust/")
        print("└── ...")
        print("\nYou can download the dataset from:")
        print("https://www.kaggle.com/datasets/emmarex/plantdisease")
        
        # Create sample structure for testing
        from data_preprocessing import create_sample_data_structure
        create_sample_data_structure()


if __name__ == "__main__":
    main()