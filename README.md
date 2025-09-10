# AI-Powered Crop Disease Detection System

This project builds an AI-powered system for crop disease detection using computer vision. A Convolutional Neural Network (CNN) is trained on publicly available plant leaf disease datasets to identify common diseases and help farmers take preventive measures.

## Features

- **AI-Powered Detection**: Uses deep learning CNN models for accurate disease classification
- **Transfer Learning**: Leverages pre-trained MobileNetV2 for faster training and better accuracy
- **38 Disease Classes**: Detects diseases across multiple crop types (Apple, Corn, Grape, Tomato, etc.)
- **Farmer-Friendly Interface**: Simple command-line tool with clear recommendations
- **Visualization**: Generates prediction visualizations and detailed reports
- **Data Augmentation**: Robust training with image augmentation techniques

## Supported Crops and Diseases

The system can detect diseases in the following crops:
- **Apple**: Scab, Black rot, Cedar apple rust
- **Corn**: Cercospora leaf spot, Common rust, Northern Leaf Blight
- **Grape**: Black rot, Esca, Leaf blight
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, etc.
- **Potato**: Early blight, Late blight
- **And many more...**

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alinapradhan/AI-powered-system-for-crop-disease-detection.git
cd AI-powered-system-for-crop-disease-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup the project environment:
```bash
python main.py setup
```

## Dataset

This project uses the PlantVillage dataset available on Kaggle:
- **Source**: [Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Size**: ~87,000 images across 38 classes
- **Format**: RGB images of healthy and diseased plant leaves

### Download Instructions:
1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. Download and extract the dataset
3. Place it in the `data/plantvillage/` directory

## Usage

### Training the Model

Train the model using transfer learning (recommended):
```bash
python main.py train
```

Train from scratch (longer training time):
```bash
python main.py train --no-transfer
```

### Making Predictions

Predict disease for a single image:
```bash
python main.py predict path/to/your/plant_image.jpg
```

Predict and save results:
```bash
python main.py predict path/to/your/plant_image.jpg --save
```

### Example Output

```
🔍 Disease Detection Results:

📱 Plant: Tomato
🏥 Condition: Late Blight
🎯 Confidence: 87.3%

📊 Top 3 Predictions:
  1. Tomato - Late Blight: 87.3%
  2. Tomato - Early Blight: 8.2%
  3. Tomato - Healthy: 3.1%

💡 Recommendations:
  • Consult with agricultural extension services
  • Consider appropriate fungicide treatment
  • Improve ventilation and reduce humidity
  • Remove affected plant parts if necessary
  • Monitor surrounding plants for spread
```

## Project Structure
 
```
├── main.py                 # Main CLI interface
├── requirements.txt        # Python dependencies
├── src/
│   ├── config.py          # Configuration settings
│   ├── model.py           # CNN model architecture
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   ├── train.py           # Model training script
│   └── predict.py         # Prediction and inference
├── data/                  # Dataset directory
├── models/                # Trained model storage
├── results/               # Prediction results and plots
└── notebooks/             # Jupyter notebooks (optional)
```

## Model Architecture

The system offers two model options:

### 1. Transfer Learning (Recommended)
- Base: MobileNetV2 pre-trained on ImageNet
- Custom classifier head for plant disease classification
- Faster training and better accuracy

### 2. Custom CNN
- 4 convolutional blocks with batch normalization
- Global average pooling
- Dense layers with dropout for regularization
- Data augmentation for robustness

## Performance

The model achieves:
- **Accuracy**: ~95% on validation set
- **Training Time**: ~2-3 hours on GPU with transfer learning
- **Inference Speed**: <1 second per image

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Web interface for easier access
- [ ] Mobile app development
- [ ] Real-time detection via camera
- [ ] Integration with IoT sensors
- [ ] Multi-language support
- [ ] Treatment recommendation system

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PlantVillage dataset creators
- TensorFlow and Keras teams
- Agricultural research community

----

**Data Source**: https://www.kaggle.com/datasets/emmarex/plantdisease

**Made with ❤️ for farmers and agricultural communities**
