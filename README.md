# MSFF-Net
Multimodal Deep Learning: Tumor and Visceral Fat Impact on Colorectal Cancer Occult Peritoneal Metastasis
## Requirements
* python3.11.4
* pytorch2.1.0+cu121
* tensorboard 2.14.0
## Usage
### 1.dataset
* Tumor CT images, L3 level CT images, and clinical information in 535 patients with CRC.  
* **PS:** The data **cannot be shared publicly** due to the privacy of individuals that participated in the study and because the data is intended for future research purposes.
### 2.Train the MSFF-Net
* You need to train the MSFF-Net with the following commands:  
`$ python train.py`  
* You can modify the training hyperparameters in `$ config.py`.
### 4.Predict PM
* If you wish to see predictions for MSFF-Net or other base models, you should run the following file:  
`$ python predict.py`
* After this operation, a feature file will be generated for feature fusion and machine learning model classification.
### 5.Machine learning model classification
* The `$ svm.py` file provides code for SVM, RF and GBDT to predict PM. You can run this file to obtain prediction results.
### 6.run tensorboard
`$ tensorboard --logdir=./logs/`
