#!/usr/bin/env python3
"""
Command Line Interface for Crop Disease Detection System
This tool helps farmers detect diseases in their crops using AI-powered computer vision.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import numpy
        import tensorflow
        import cv2
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("\n🔧 Installation required:")
        print("pip install -r requirements.txt")
        return False

# Try to import modules, but handle gracefully if dependencies are missing
try:
    from predict import DiseasePredictor
    from train import ModelTrainer
    from data_preprocessing import DataPreprocessor, create_sample_data_structure
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Some modules unavailable: {e}")
    MODULES_AVAILABLE = False
    
    # Define dummy functions for setup command
    def create_sample_data_structure():
        """Create sample directory structure for testing."""
        import os
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


def predict_single_image(image_path, model_path=None, save_results=False):
    """Predict disease for a single image."""
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available. Please install dependencies first:")
        print("pip install -r requirements.txt")
        return
    
    if not check_dependencies():
        return
        
    predictor = DiseasePredictor(model_path)
    
    if not predictor.load_model():
        print("Failed to load model. Please train the model first using 'python main.py train'")
        return
    
    print(f"Analyzing image: {image_path}")
    print("-" * 50)
    
    # Make prediction
    prediction = predictor.predict_image(image_path)
    if prediction is None:
        print("Failed to make prediction. Please check the image file.")
        return
    
    # Display results
    print("🔍 Disease Detection Results:")
    print()
    
    top_prediction = prediction[0]
    disease_info = predictor.get_disease_info(top_prediction['disease'])
    
    print(f"📱 Plant: {disease_info['plant']}")
    print(f"🏥 Condition: {disease_info['condition']}")
    print(f"🎯 Confidence: {top_prediction['percentage']:.1f}%")
    print()
    
    print("📊 Top 3 Predictions:")
    for i, pred in enumerate(prediction, 1):
        disease_display = pred['disease'].replace('___', ' - ').replace('_', ' ')
        print(f"  {i}. {disease_display}: {pred['percentage']:.1f}%")
    
    print()
    print("💡 Recommendations:")
    for rec in disease_info['recommendations']:
        print(f"  • {rec}")
    
    if save_results:
        # Save visualization and report
        os.makedirs('results', exist_ok=True)
        predictor.visualize_prediction(image_path, 'results/latest_prediction.png')
        predictor.generate_report(image_path, 'results/latest_report.json')
        print("\n💾 Results saved to 'results/' directory")


def train_model(data_dir=None, use_transfer_learning=True):
    """Train the crop disease detection model."""
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available. Please install dependencies first:")
        print("pip install -r requirements.txt")
        return
    
    if not check_dependencies():
        return
        
    print("🚀 Starting Model Training")
    print("=" * 50)
    
    if data_dir and not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    try:
        trainer = ModelTrainer(use_transfer_learning=use_transfer_learning)
        trainer.setup_model()
        
        preprocessor = DataPreprocessor(data_dir)
        
        # Try to load data
        train_ds, val_ds, class_names = preprocessor.create_dataset_from_directory()
        print(f"✅ Found {len(class_names)} disease classes")
        
        # Train the model
        print("🏋️ Training in progress...")
        history = trainer.train_with_datasets(train_ds, val_ds)
        
        # Plot results
        trainer.plot_training_history()
        
        print("✅ Training completed successfully!")
        print(f"📁 Model saved to: models/crop_disease_model.h5")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n🔧 Setup Instructions:")
        print("1. Download the PlantVillage dataset from:")
        print("   https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("2. Extract to 'data/plantvillage/' directory")
        print("3. Run training again")


def setup_environment():
    """Setup the project environment."""
    print("🛠️  Setting up Crop Disease Detection System")
    print("=" * 50)
    
    # Create necessary directories
    directories = ['data', 'models', 'results', 'data/sample']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create sample data structure
    create_sample_data_structure()
    
    print("\n📚 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download dataset from: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("3. Extract dataset to 'data/plantvillage/' directory")
    print("4. Train model: python main.py train")
    print("5. Test prediction: python main.py predict path/to/image.jpg")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Crop Disease Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup                          # Setup project environment
  python main.py train                          # Train the model
  python main.py predict image.jpg              # Predict disease for an image
  python main.py predict image.jpg --save       # Predict and save results
  python main.py train --data data/custom       # Train with custom data directory
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup project environment')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', type=str, default='data/plantvillage',
                             help='Path to training data directory')
    train_parser.add_argument('--no-transfer', action='store_true',
                             help='Train from scratch instead of using transfer learning')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict disease for an image')
    predict_parser.add_argument('image', type=str, help='Path to image file')
    predict_parser.add_argument('--model', type=str, default=None,
                               help='Path to model file (default: models/crop_disease_model.h5)')
    predict_parser.add_argument('--save', action='store_true',
                               help='Save prediction results and visualization')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_environment()
    
    elif args.command == 'train':
        train_model(args.data, use_transfer_learning=not args.no_transfer)
    
    elif args.command == 'predict':
        if not MODULES_AVAILABLE:
            print("❌ Required modules not available. Please install dependencies first:")
            print("pip install -r requirements.txt")
            return
            
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            return
        
        predict_single_image(args.image, args.model, args.save)
    
    else:
        parser.print_help()
        print("\n🌱 Welcome to the AI-Powered Crop Disease Detection System!")
        print("This tool helps farmers identify diseases in their crops using computer vision.")
        print("Use --help with any command for more information.")


if __name__ == "__main__":
    main()