\# Brain Tumor MRI Classification



This repository contains the source code, experimental workflows, and selected results for a brain MRI tumor classification research project.



\## Proposed Model



The proposed framework combines:



\- Convolutional Neural Network (CNN)

\- Bidirectional Long Short-Term Memory (BiLSTM)

\- Attention mechanism



The model performs four-class brain MRI classification:



1\. Glioma

2\. Meningioma

3\. No Tumor

4\. Pituitary Tumor



\## Main Experiments



The repository contains code and experimental results for:



\- Proposed Hybrid CNN-BiLSTM-Attention model

\- VGG16

\- ResNet50

\- EfficientNet-B3

\- DenseNet121

\- Five-fold cross-validation

\- Sequence-ordering ablation experiments

\- Cross-dataset validation

\- Statistical significance testing

\- LIME explainability

\- Grad-CAM++ explainability

\- Dataset overlap and duplicate analysis



\## Dataset A



Dataset A is a publicly available brain MRI dataset containing 7,023 images from four classes.



Source:



https://data.mendeley.com/datasets/w4sw3s9f59/1



The dataset combines images originating from Figshare, SARTAJ, and Br35H.



\## Dataset B



Dataset B was used for additional external and cross-dataset validation.



Source:



https://data.mendeley.com/datasets/zwr4ntf94j/4



\## Repository Contents



`all\_codes/` contains the main preprocessing, training, evaluation, cross-validation, explainability, ablation, and statistical-analysis scripts.



Experimental result directories contain CSV, JSON, and text outputs used for analysis and manuscript preparation.



\## Important Note



The MRI image datasets and large trained model checkpoint files are not included in this repository because of their size. Dataset links are provided above so that the datasets can be obtained from their original public sources.



\## Research Use



This repository is intended for academic and research purposes.

