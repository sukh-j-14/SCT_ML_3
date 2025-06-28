# Dogs vs Cats Classification using Support Vector Machine (SVM)

## 📋 Project Overview

This project implements a Support Vector Machine (SVM) classifier to distinguish between images of dogs and cats using the popular Kaggle Dogs vs Cats dataset. The model achieves binary classification with competitive accuracy on this challenging computer vision task.

## 🎯 Objective

The primary goal is to develop a machine learning model that can accurately classify images as either containing a dog (label: 1) or a cat (label: 0) using Support Vector Machine algorithms.

## 📊 Dataset

**Dataset Source:** [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)

**Dataset Details:**
- **Training Set:** 25,000 images (12,500 cats, 12,500 dogs)
- **Test Set:** 12,500 images (unlabeled)
- **Image Format:** JPG files
- **Image Resolution:** Variable (resized to 64x64 pixels for processing)

## 🏗️ Methodology

### Data Preprocessing
1. **Image Loading:** Images are loaded using OpenCV
2. **Resizing:** All images are resized to 64x64 pixels for consistency
3. **Flattening:** Images are flattened into 1D vectors (12,288 features per image)
4. **Normalization:** Pixel values are normalized to range [0, 1] by dividing by 255
5. **Label Encoding:** 
   - Cats: 0
   - Dogs: 1

### Model Architecture
- **Algorithm:** Support Vector Machine (SVM)
- **Kernel:** Default RBF kernel
- **Implementation:** scikit-learn SVC class

### Training Process
- **Train-Test Split:** 80% training, 20% validation
- **Training Samples:** 20,000 images
- **Validation Samples:** 5,000 images
- **Random State:** 42 (for reproducibility)

## 📈 Results

### Model Performance
- **Accuracy:** 68.90% on validation set
- **Model Size:** ~1.5GB (saved as pickle file)

### Key Findings
- The SVM model demonstrates reasonable performance on the binary classification task
- The model successfully generalizes to unseen test data
- Performance could be improved with feature engineering or hyperparameter tuning

## 🚀 Usage

### Prerequisites
```bash
pip install numpy pandas scikit-learn opencv-python joblib
```

### Running the Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/sukh-j-14/DogsVsCats.git
   cd DogsVsCats
   ```

2. **Download the dataset:**
   - Download from [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
   - Extract the zip files to the project directory

3. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook dogsvscats.ipynb
   ```

### Model Loading
```python
import joblib

# Load the trained model
model = joblib.load('svm_cat_dog_model.pkl')

# Make predictions
predictions = model.predict(test_data)

## 🔧 Technical Details

### Libraries Used
- **NumPy:** Numerical computations and array operations
- **Pandas:** Data manipulation and CSV handling
- **OpenCV:** Image processing and loading
- **scikit-learn:** SVM implementation and model evaluation
- **joblib:** Model serialization

### Model Specifications
- **Input Shape:** (n_samples, 12288) - flattened 64x64x3 images
- **Output:** Binary classification (0: cat, 1: dog)
- **Training Time:** ~47 minutes (on Kaggle environment)
- **Prediction Time:** ~82 minutes for 12,500 test images

## 📊 Model Performance Analysis

### Strengths
- Simple and interpretable model
- Good baseline performance for binary classification
- Handles high-dimensional data effectively

### Limitations
- Limited accuracy compared to deep learning approaches
- Large model size (1.5GB)
- Long training and prediction times
- No feature extraction - uses raw pixel values

### Potential Improvements
- Feature engineering (HOG, SIFT, etc.)
- Hyperparameter tuning (C, gamma, kernel selection)
- Data augmentation
- Ensemble methods
- Transfer learning with pre-trained models

## 🔗 Links

### Model Download
**Trained Model:** [Download SVM Model](https://drive.google.com/file/d/152sagrX2_gP4aQwjAlekOLmtzWP1N2Jj/view?usp=drive_link)

### Dataset
**Kaggle Dataset:** [Dogs vs Cats Competition](https://www.kaggle.com/c/dogs-vs-cats/data)

## 👨‍💻 Author

This project was implemented as part of a machine learning assignment focusing on Support Vector Machine classification for computer vision tasks.

## 📝 License

This project is for educational purposes. The dataset is provided by Kaggle and subject to their terms of use.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---
