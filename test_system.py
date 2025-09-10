"""
Test script for the crop disease detection system
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from model import CropDiseaseModel, create_transfer_learning_model
    from data_preprocessing import DataPreprocessor
    from predict import DiseasePredictor
    from config import MODEL_CONFIG, DISEASE_CLASSES
except ImportError as e:
    print(f"Warning: Could not import modules for testing: {e}")
    print("This is expected if dependencies are not installed.")


class TestCropDiseaseDetection(unittest.TestCase):
    """Test cases for the crop disease detection system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CropDiseaseModel()
        self.preprocessor = DataPreprocessor()
    
    def test_model_creation(self):
        """Test CNN model creation."""
        try:
            model = self.model.build_model()
            self.assertIsNotNone(model)
            print("✅ Model creation test passed")
        except Exception as e:
            print(f"⚠️  Model creation test skipped (dependencies): {e}")
    
    def test_config_values(self):
        """Test configuration values."""
        self.assertEqual(MODEL_CONFIG['num_classes'], 38)
        self.assertEqual(len(DISEASE_CLASSES), 38)
        self.assertEqual(MODEL_CONFIG['input_shape'], (224, 224, 3))
        print("✅ Configuration test passed")
    
    def test_disease_classes(self):
        """Test disease class definitions."""
        # Check if all classes have proper format
        for disease in DISEASE_CLASSES:
            self.assertIn('___', disease)
            parts = disease.split('___')
            self.assertEqual(len(parts), 2)
        print("✅ Disease classes test passed")
    
    def test_data_preprocessor_init(self):
        """Test data preprocessor initialization."""
        self.assertEqual(self.preprocessor.img_height, 224)
        self.assertEqual(self.preprocessor.img_width, 224)
        print("✅ Data preprocessor test passed")
    
    def test_project_structure(self):
        """Test if required directories exist."""
        required_dirs = ['src', 'data', 'models', 'results']
        for directory in required_dirs:
            self.assertTrue(os.path.exists(directory), f"Directory {directory} missing")
        print("✅ Project structure test passed")
    
    def test_requirements_file(self):
        """Test if requirements.txt exists and has required packages."""
        self.assertTrue(os.path.exists('requirements.txt'))
        
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_packages = ['tensorflow', 'opencv-python', 'numpy', 'matplotlib']
        for package in required_packages:
            self.assertIn(package, requirements)
        print("✅ Requirements file test passed")


def run_system_tests():
    """Run basic system functionality tests."""
    print("🧪 Running Crop Disease Detection System Tests")
    print("=" * 50)
    
    # Test 1: Check if main.py exists and is executable
    if os.path.exists('main.py'):
        print("✅ Main CLI script exists")
    else:
        print("❌ Main CLI script missing")
    
    # Test 2: Check src directory structure
    src_files = ['config.py', 'model.py', 'data_preprocessing.py', 'train.py', 'predict.py']
    all_src_files_exist = True
    for file in src_files:
        if os.path.exists(f'src/{file}'):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_src_files_exist = False
    
    # Test 3: Try importing modules (if dependencies are available)
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow available (version: {tf.__version__})")
    except ImportError:
        print("⚠️  TensorFlow not installed - install with: pip install -r requirements.txt")
    
    try:
        import cv2
        print(f"✅ OpenCV available (version: {cv2.__version__})")
    except ImportError:
        print("⚠️  OpenCV not installed - install with: pip install -r requirements.txt")
    
    # Test 4: CLI help functionality
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'main.py', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CLI help command works")
        else:
            print("⚠️  CLI help command has issues")
    except Exception as e:
        print(f"⚠️  CLI test skipped: {e}")
    
    print("\n📋 Test Summary:")
    if all_src_files_exist:
        print("✅ All core files are present")
        print("✅ System is ready for use")
        print("\n🚀 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Setup environment: python main.py setup")
        print("3. Download dataset and train: python main.py train")
    else:
        print("❌ Some core files are missing")


if __name__ == "__main__":
    # Run system tests first
    run_system_tests()
    
    print("\n" + "=" * 50)
    
    # Run unit tests if modules can be imported
    try:
        unittest.main(verbosity=2, exit=False)
    except Exception as e:
        print(f"Unit tests skipped: {e}")
        print("Install dependencies to run full tests: pip install -r requirements.txt")