# Automated Segmentation of Cardiac Chambers in MRI Using Deep Learning for Clinical Assessment

##  Overview

This repository contains an advanced deep learning framework for automated segmentation of cardiac chambers in MRI scans, specifically designed for the ACDC (Automatic Cardiac Diagnosis Challenge) dataset. The system employs a sophisticated Deep U-Net architecture with meta-learning capabilities to achieve state-of-the-art performance in multi-class cardiac tissue segmentation.

The model accurately delineates four critical cardiac structures:
- **Right Ventricle (RV)**
- **Left Ventricle (LV)**
- **Myocardium**
- **Background**

This tool is particularly valuable for clinical applications such as ventricular volume estimation, ejection fraction calculation, and myocardial mass assessment - essential metrics in cardiology.

##  Key Features

- **Deep U-Net Architecture**: Custom-built neural network with encoder-decoder structure for precise segmentation
- **Multi-class Segmentation**: Simultaneous identification of RV, LV, myocardium, and background
- **Data Augmentation**: Advanced augmentation techniques for improved generalization
- **Meta-Learning Integration**: Enhanced performance through ensemble learning strategies
- **Comprehensive Evaluation**: Extensive metrics including Dice coefficient, accuracy, and confusion matrices
- **Clinical Readiness**: Designed with clinical deployment considerations

##  Performance Results

The proposed deep learning framework was evaluated on the independent test set of the ACDC dataset, comprising 20 patient volumes not seen during training or validation. The model achieved an overall accuracy of **0.9913** and a mean Dice coefficient of **0.8968**, demonstrating strong performance across all cardiac structures.

### Table 1: Dice Scores and Overall Accuracy
| Structure | Dice Score |
|-----------|------------|
| Background | 0.9965 |
| Right Ventricle (RV) | 0.8148 |
| Myocardium | 0.8454 |
| Left Ventricle (LV) | 0.9304 |
| **Mean Dice** | **0.8968** |
| **Accuracy** | **0.9913** |

*Table 1: Dice scores and overall accuracy for multi-class cardiac MRI segmentation on the ACDC test set.*

### Visual Results
- **Figure 1**: Example segmentation outputs for RV, LV, and myocardium on test slices. Left: MRI input; Middle: Ground truth; Right: Model predictions.
- **Figure 2**: Comprehensive visualization of performance metrics: (Top-left) Dice scores by class; (Top-right) Dice score distribution as pie chart; (Bottom-left) sample input image; (Bottom-right) summary of quantitative metrics.

##  Quick Start

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.0+
- GPU with CUDA support (recommended for training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AlirezaRahi/Automated-Segmentation-of-Cardiac-Chambers-in-MRI-Using-Deep-Learning-for-Clinical-Assessment.git
cd Automated-Segmentation-of-Cardiac-Chambers-in-MRI-Using-Deep-Learning-for-Clinical-Assessment
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)
```
tensorflow>=2.0.0
numpy>=1.19.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
h5py>=3.1.0
joblib>=1.0.0
```

##  Dataset Setup

### ACDC Dataset Preparation

1. **Download the ACDC dataset** from the official challenge website:
   - [ACDC Challenge Dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)

2. **Organize the dataset structure**:
```
data/
├── training/
│   ├── patient001/
│   ├── patient002/
│   └── ...
└── testing/
    ├── patient101/
    ├── patient102/
    └── ...
```

3. **Update the data path** in the main script:
```python
# Update this path in acdc_deep_unet_meta_dice_fixed_single.py
sys.path.append('path/to/your/data/directory')
```

## 🏃‍♂️ Usage Instructions

### Training the Model

1. **Configure training parameters** in `main()` function:
```python
# Adjust these parameters as needed
target_size = (192, 192)
batch_size = 4
learning_rate = 1e-3
epochs = 100
```

2. **Run the training script**:
```bash
python acdc_deep_unet_meta_dice_fixed_single.py
```

### Model Training Process

The training process includes:
- **Data loading and preprocessing** (resizing, normalization, augmentation)
- **Model construction** (Deep U-Net architecture)
- **Training with callbacks** (early stopping, learning rate reduction, model checkpointing)
- **Validation and testing**
- **Results visualization and saving**

### Evaluation

The model automatically evaluates on the test set and generates:
- **Performance metrics** (Dice scores, accuracy)
- **Visualization plots** (confusion matrices, ROC curves, segmentation examples)
- **Detailed reports** (JSON and text formats)

### Output Structure

After training, results are saved in a timestamped directory:
```
acdc_evaluation_YYYYMMDD_HHMMSS/
├── results.json              # Detailed metrics in JSON format
├── report.txt               # Text report of results
├── u-net_v1_evaluation_plots.png      # Main evaluation plots
└── u-net_v1_detailed_analysis.png    # Detailed analysis plots
```

## 🏗️ Model Architecture

### Deep U-Net Design

The model follows an encoder-decoder architecture:

**Encoder (Contracting Path):**
- 4 downsampling blocks with convolution + pooling
- Feature maps: 64 → 128 → 256 → 512 → 1024
- Max pooling for spatial dimension reduction

**Bottleneck:**
- Deep feature extraction with 1024 filters

**Decoder (Expansive Path):**
- 4 upsampling blocks with transposed convolution + skip connections
- Feature concatenation from corresponding encoder layers
- Feature maps: 1024 → 512 → 256 → 128 → 64

**Output:**
- Final convolution with 4 filters (one per class)
- Softmax activation for multi-class segmentation

### Key Technical Features
- **Skip Connections**: Preserve spatial information from encoder to decoder
- **Batch Normalization**: Improved training stability
- **Dropout**: Regularization to prevent overfitting
- **Deep Supervision**: Enhanced gradient flow

##  Performance Analysis

### Quantitative Results

The model demonstrates excellent performance metrics:

1. **Overall Accuracy**: 99.13%
2. **Mean Dice Coefficient**: 89.68%
3. **Class-wise Performance**:
   - **Background**: 99.65% Dice - Near-perfect segmentation
   - **Left Ventricle**: 93.04% Dice - Excellent performance
   - **Myocardium**: 84.54% Dice - Strong segmentation
   - **Right Ventricle**: 81.48% Dice - Good performance with room for improvement

### Clinical Significance

The high Dice scores for LV and myocardium suggest the model's suitability for clinical applications:

- **Ventricular Volume Estimation**: Accurate for ejection fraction calculation
- **Myocardial Mass Assessment**: Precise for cardiac function evaluation
- **Disease Classification**: Reliable for diagnostic support
- **Treatment Planning**: Useful for intervention planning

##  Technical Details

### Data Preprocessing Pipeline

1. **Image Loading**: HDF5 format reading with proper key identification
2. **Resizing**: Standardization to 192×192 pixels
3. **Normalization**: Min-max scaling to [0, 1] range
4. **Augmentation** (training only):
   - Random flips (horizontal/vertical)
   - Random rotations (0-360 degrees)
   - Brightness adjustments

### Training Strategy

1. **Loss Function**: Sparse Categorical Crossentropy
2. **Optimizer**: Adam with learning rate scheduling
3. **Regularization**:
   - Early stopping (patience: 15 epochs)
   - Learning rate reduction on plateau
   - Model checkpointing for best weights
4. **Validation**: 30% of data for validation, 20% for testing

### Evaluation Metrics

1. **Dice Coefficient (F1-Score)**:
   ```python
   Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
   ```
2. **Accuracy**: Percentage of correctly classified pixels
3. **Confusion Matrix**: Per-class performance visualization
4. **ROC Curves**: Trade-off between sensitivity and specificity

##  Applications

### Clinical Use Cases

1. **Cardiac Function Assessment**:
   - Ejection fraction calculation
   - Stroke volume estimation
   - Cardiac output measurement

2. **Disease Diagnosis**:
   - Cardiomyopathy detection
   - Ventricular hypertrophy identification
   - Wall motion abnormality assessment

3. **Treatment Monitoring**:
   - Pre- and post-treatment comparison
   - Surgical outcome evaluation
   - Medication effect assessment

4. **Research Applications**:
   - Population studies
   - Longitudinal analysis
   - Comparative effectiveness research

### Integration into Clinical Workflow

The model can be integrated into:
- **PACS systems** for automated analysis
- **Clinical decision support systems**
- **Telemedicine platforms**
- **Research databases**

##  Citation

If you use this code in your research, please cite:

**Rahi, A. (2025). Automated Segmentation of Cardiac Chambers in MRI Using Deep Learning for Clinical Assessment [Computer model]. GitHub. https://github.com/AlirezaRahi/Automated-Segmentation-of-Cardiac-Chambers-in-MRI-Using-Deep-Learning-for-Clinical-Assessment**

###  Publication

**Title:** Automated Segmentation of Cardiac Chambers in MRI Using Deep Learning for Clinical Assessment  
**Journal:** *[]*  
**DOI:** *[]*  
**Publication Date:** *[]*  
**URL:** *[]*

##  License

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).

**Summary of License Terms**:

This work may be read and downloaded for personal use only. It may be shared in its complete and unaltered form for non-commercial purposes, provided that the author's name, the title of the work, and a link to the original source are clearly cited. Any modification, adaptation, commercial use, or distribution for profit is strictly prohibited. For permissions beyond this license, please contact the author directly.

**Full License Details**:

Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made.

**NonCommercial**: You may not use the material for commercial purposes.

NoDerivatives: If you remix, transform, or build upon the material, you may not distribute the modified material.

**To view a copy of this license, visit**:
https://creativecommons.org/licenses/by-nc-nd/4.0/

**Commercial Use**:
For commercial licensing or collaboration inquiries, please contact the author directly.

**Academic Use**:
Researchers are encouraged to use, cite, and build upon this work for non-commercial research purposes with proper attribution.

### Areas for Contribution
- Model architecture improvements
- Additional evaluation metrics
- Support for other cardiac MRI datasets
- Docker containerization
- Web interface development
- Documentation enhancements

## ❓ Frequently Asked Questions

### Q1: What GPU requirements are needed?
**A**: Minimum 8GB GPU memory for training, 4GB for inference. RTX 2080 or higher recommended.

### Q2: How long does training take?
**A**: Approximately 4-6 hours for 100 epochs on a modern GPU with the ACDC dataset.

### Q3: Can I use this with other cardiac MRI datasets?
**A**: Yes, with appropriate data preprocessing and format adjustments.

### Q4: What's the inference time per image?
**A**: Approximately 0.1-0.2 seconds per slice on GPU.

### Q5: How accurate is the model compared to manual segmentation?
**A**: The model achieves expert-level performance with Dice scores comparable to inter-observer variability among clinicians.

##  Troubleshooting

### Common Issues and Solutions

1. **Out of Memory Error**:
   - Reduce batch size
   - Use smaller input images
   - Enable mixed precision training

2. **Slow Training**:
   - Ensure GPU is being utilized
   - Check for CPU bottlenecks in data loading
   - Use TFRecord format for faster I/O

3. **Poor Performance**:
   - Check data preprocessing
   - Verify label consistency
   - Adjust learning rate
   - Increase training data

4. **Installation Issues**:
   - Use exact version numbers from requirements.txt
   - Create fresh virtual environment
   - Check CUDA/cuDNN compatibility

##  Future Work

Planned enhancements include:

1. **Architecture Improvements**:
   - Attention mechanisms
   - Transformer-based models
   - Multi-scale feature fusion

2. **Functionality Extensions**:
   - 3D segmentation support
   - Real-time inference
   - Uncertainty quantification
   - Few-shot learning capabilities

3. **Clinical Integration**:
   - DICOM standard support
   - Hospital PACS integration
   - Regulatory compliance (FDA/CE)

4. **Additional Features**:
   - Automated report generation
   - Longitudinal analysis tools
   - Multi-modal fusion (MRI+CT)

## 🙏 Acknowledgments

- **ACDC Challenge Organizers** for providing the dataset
- **TensorFlow Team** for the deep learning framework
- **Medical Imaging Research Community** for valuable resources and tools
- **Clinical Collaborators** for domain expertise and validation

## 📚 References

1. Bernard, O., Lalande, A., Zotti, C., et al. (2018). Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved? IEEE Transactions on Medical Imaging.
2. Isensee, F., Jaeger, P. F., Kohl, S. A., et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods.
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
4. Litjens, G., Kooi, T., Bejnordi, B. E., et al. (2017). A survey on deep learning in medical image analysis. Medical Image Analysis.
5. Chen, C., Qin, C., Qiu, H., et al. (2020). Deep learning for cardiac image segmentation: A review. Frontiers in Cardiovascular Medicine.

## 📞 Contact

**Alireza Rahi**  
- 📧 Email: alireza.rahi@outlook.com  
- 💼 LinkedIn: [Alireza Rahi](https://www.linkedin.com/in/alireza-rahi-6938b4154/)  
- 💻 GitHub: [AlirezaRahi](https://github.com/AlirezaRahi)  

For questions, collaborations, or support, please feel free to reach out.

---

**Disclaimer**: This tool is intended for research purposes only. Clinical decisions should not be made solely based on the output of this software without verification by qualified medical professionals.

---

*Last Updated: December 2025*  
*Version: 1.0.0*
