# Snowdrift Detection and Characterization – Workflow Repository

## Overview
This repository contains the full workflow of the thesis:  
**_“Exploring the Potential of Synthetic Aperture Radar (SAR) Imagery for Snowdrift Detection and Characterization: A Case Study in Atka Bay, Antarctica.”_**

It includes preprocessing of Sentinel-1 SAR data and meteorological datasets, segmentation and trajectory extraction workflows, and mixed-effects modeling for environmental interpretation.  
Both **Python** and **R** environments are used.

---

## 1. Jupyter Notebooks

### **`S1_pre_processing.ipynb`**
Preprocessing workflow for **Sentinel-1 GRD** products:
- Radiometric calibration  
- Speckle filtering  
- Subsetting and terrain correction  
- Export of processed backscatter images  

Executed in **Jupyter Notebook**.

---

### **`ERA5_NM_daily_processing.ipynb`**
Processing of meteorological datasets (**ERA5** reanalysis and **Neumayer III** station data):
- Harmonization and temporal aggregation  
- Daily statistics computation  
- Visualization of meteorological parameters  
- Export of variables for modeling  

Executed in **Jupyter Notebook**.

---

## 2. Python Modules

### **`polygon_seg.py`**
Encapsulated functions for:
- Three-class Otsu threshold segmentation  
- ROI polygon marking  
- Middle-class mask generation  

---

### **`SnowDrift.py`**
Contains core functions for:
- Anchor-point identification  
- Snowdrift trajectory detection  
- Drift-trail length computation  

These modules are used by the execution scripts (`run_poly.py`, `run_trace.py`).

---

## 3. Execution Scripts

### **`run_poly.py`**
Runs segmentation and ROI extraction on preprocessed Sentinel-1 backscatter images using functions defined in `polygon_seg.py`.

---

### **`run_trace.py`**
Executes snowdrift trajectory detection and trail-length computation using methods in `SnowDrift.py`.

---

## 4. R Script

### **`LMM_SnowDrift.R`**
R script for performing linear mixed-effects modeling (LMM):
- Evaluates the influence of wind and snowfall on drift-trail geometry  
- Implements random-intercept and random-slope structures  

---

## 5. Environment

### **`snowdrift-env.yml`**
Conda environment specification for running all Python components.

Create and activate the environment:

---

## 6. Trajectory-Length Results (Not Included in Repo)

All ROI-level drift-trajectory result images are too large to upload.  
They are available via Google Drive:

🔗 **Google Drive folder:** [Yi_Qin_MSc_Thesis][https://drive.google.com/drive/folders/1Lju1HAf-U_xfHYduQVoOsQ-ZxSzuP92e?usp=drive_link]


This folder contains:
- **Trajectory_Results** — All ROI-level trajectory-length output images  
- **Yi_Qin_MA_thesis_revised.pdf** — Final thesis document  
- **Yi_Qin_1445674_Presentation_MSc._Thesis_Defense.pdf** — Defense presentation slides  
