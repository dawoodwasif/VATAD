# VATAD
VATAD: Video Augmentation Technique for Action Detection

### 1. Install required dependencies.
`pip install -r requirements.txt`

### 2. Download sam weights and place it in weights folder.
`mkdir weights`
`cd weights`
`wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`

### 3. Update config.yaml file to change parameters

### 4. Execute main file to start video processing
`python main.py`
