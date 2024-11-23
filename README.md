# ExerciseLLM
Python code and dataset repository for paper "Leveraging Large Language Models for Rehabilitation Exercise Quality
Assessment and Feedback Generation"
KITE Research Institute 2024

[[Project Page]](insert_link) [[Paper]](insert_link) 

(add image of architecture/visualizations)

(add citations block)

## Table of Content
* [1. Results](#1-results)
* [2. Datasets](#2-datasets)
* [3. Evaluation](#3-evaluation)
* [4. Acknowledgements](#4-acknowledgements)


## 1. Results 
(insert results images, text, etc)
 
## 2. Installation

### 2.1. Environment
create a virtual env to install the requirements:
```bash
pip install -r requirements.txt
```

### 2.2. Datasets
**ExerciseLLM** is a categorized and described rehabilitation exercise movement dataset that originates from [UI-PRMD](https://webpages.uidaho.edu/ui-prmd/) and [IRDS](https://www.mdpi.com/2306-5729/6/5/46) dataset, which can be downloaded directly from the linked sites. 

Currently for UI-PRMD, only the segmented files from Kinect are used. Ensure the downloaded dataset follows the format below:
```
./dataset/
├── UI-PRMD
    ├── correct
        ├── kinect
            ├── angles/
            ├── positions
                ├── m01_s01_e01_positions.txt
                ├── m01_s01_e02_positions.txt
                ├── ...
                └── m10_s10_e10_positions.txt
    └── incorrect/
└── UI-PRMD_visualization.ipynb
```

## 3. Generate Our Dataset
Run the following Python scripts to generate the corresponding data files

**Absolute Coordinates**
Position to absolute coordinates: `generate_coordinates.py`

**Features**
Feature extractors: `generate_features_set.py`

**Chain of Thought**
Feature extractors: `generate_cot_set.py`

**The final directory tree follows:**
```
./dataset/
├── IRDS # need to expand later
├── UI-PRMD
    ├── correct
        ├── kinect
            ├── angles/
            ├── positions
                ├── m01_s01_e01_positions.txt
                ├── m01_s01_e02_positions.txt
                ├── ...
                └── m10_s10_e10_positions.txt
    └── incorrect/
├── UI-PRMD_generated
    ├── correct
        ├── features
            ├── m01_s01_e01_features.csv
            ├── m01_s01_e02_features.csv
            ├── ...
            └── m10_s10_e10_features.csv
        ├── coordinates/
        └── chainofthought/ #idk yet for this, need api 
    └── incorrect/
└── UI-PRMD_visualization.ipynb # also move this somewhere else i feel like
```

## 4. Evaluation 
Automatic prompt generation:
Prompt creator (rand demo/test splits)

## 5. Visualization
Visualize UI-PRMD movements by running `visualization/UI-PRMD_visualization.ipynb`

### 6. Acknowledgements

* public code 
* (add contributor names)
