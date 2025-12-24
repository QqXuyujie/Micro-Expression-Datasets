# Micro-Expression-Datasets

This repository contains a unified and extensible codebase for preprocessing and modeling micro-expression datasets, including **SMIC, SAMM, CASME II, CASME III, and DFME**.

---

## Missing Files

The following pretrained weight files are **not included** in this repository due to licensing restrictions:

- **FlowNet2_checkpoint.pth.tar** (FlowNet2.0 pretrained weights)
- **shape_predictor_68_face_landmarks.dat** (dlib 68-point facial landmark model)

Please download them manually or contact the repository author.

---

## FlowNet2.0 Installation (cp39)

The `cp39` directory contains `.whl` files for installing **FlowNet2.0**.

### Supported Environment

- **OS**: Windows  
- **Python**: 3.9  
- **PyTorch**: 1.12  
- **CUDA**: 11.3 or 11.4 (cu11.3 / cu11.4)

---

## Dataset Details

### CASME II Dataset

#### 5-Class Classification
- **Other**: 99  
- **Disgust**: 63  
- **Happiness**: 32  
- **Repression**: 27  
- **Surprise**: 25  

#### 3-Class Classification
- **Positive**: Happiness (32)  
- **Negative**: Disgust (63), Repression (27)  
- **Surprise**: Surprise (25)  

---

### CASME III Dataset

#### 4-Class Classification
- **Negative**: Disgust (250), Fear (86), Anger (64), Sadness (57)  
- **Positive**: Happy (57)  
- **Surprise**: Surprise (187)  
- **Others**: Others (161)  

#### 3-Class Classification
- **Negative**: Disgust (250), Fear (86), Anger (64), Sadness (57)  
- **Positive**: Happy (57)  
- **Surprise**: Surprise (187)  

---

### SMIC Dataset

#### 3-Class Classification
- **Positive**: Happiness (26)  
- **Negative**: Sadness, Disgust, Contempt, Fear, Anger (92)  
- **Surprise**: Surprise (15)  

#### 5-Class Classification
- **Anger**: 57  
- **Contempt**: 12  
- **Happiness**: 26  
- **Surprise**: 15  
- **Other**: 26  

---

### SAMM Dataset

#### 3-Class Classification
- **Positive**: Happiness (26)  
- **Negative**: Sadness, Disgust, Contempt, Fear, Anger (92)  
- **Surprise**: Surprise (15)  

#### 5-Class Classification
- **Anger**: 57  
- **Contempt**: 12  
- **Happiness**: 26  
- **Surprise**: 15  
- **Other**: 26  

---

## Notes

- Dataset labels follow the original dataset annotations.
- The codebase is designed to be easily extensible to new micro-expression datasets.
