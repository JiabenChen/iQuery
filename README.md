# iQuery: Instruments as Queries for Audio-Visual Sound Separation
Official implementation of [Instruments as Queries for Audio-Visual Sound Separation](https://arxiv.org/abs/2212.03814).

[CVPR Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_iQuery_Instruments_As_Queries_for_Audio-Visual_Sound_Separation_CVPR_2023_paper.pdf) [Project Page](https://jiabenchen.github.io/iQuery/) [Video](https://www.youtube.com/watch?v=EZ9CgknV9Z4)

This paper has been accepted by **CVPR 2023**.

## Introduction
We re-formulate visual-sound separation task and propose Instrument as Query (**iQuery**) with a flexible query expansion mechanism. Our approach ensures cross-modal consistency and cross-instrument disentanglement. We utilize "visually named" queries to initiate the learning of audio queries and use cross-modal attention to remove potential sound source interference at the estimated waveforms. To generalize to a new instrument or event class, drawing inspiration from the text-prompt design, we insert an additional query as an audio prompt while freezing the attention mechanism.

<center><img src="figures/pipeline.png" width="100%"></center>

## Requirements
### Installation
Create a conda environment and install dependencies:
```bash
git clone https://github.com/JiabenChen/iQuery.git
cd iQuery

conda create --name iQuery python=3.8
conda activate iQuery

# Install the according versions of torch and torchvision
pip install torch==1.12.0+cu102 torchvision==0.13.0+cu102 torchaudio==0.12.0 --extra-index-url 

# Install detectron2
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

# Install required packages
pip install -U opencv-python
pip install -r requirements.txt
```

### Dataset
1. Download datasets.

    a. Download MUSIC and MUSIC-21 dataset from: https://github.com/roudimit/MUSIC_dataset (The original MUSIC datasets provides .json files, you can refer to iQuery/scripts/download_videos.py to download videos)

    b. Download AVE dataset from: https://github.com/YapengTian/AVE-ECCV18

2. Preprocess videos. You can do it in your own way as long as the index files are similar.

    a. Extract frames and waveforms (11025Hz for MUSIC/MUSIC-21, and 22000Hz for AVE). (You can refer to iQuery/scripts/extract_audio.py and iQuery/scripts/extract_frames.py)

    b. Detect objects in video frames. For MUSIC dataset, we used object detector trained by Ruohan used in his Cosep project (see [CoSep repo](https://github.com/rhgao/co-separation)). For MUSIC-21 and AVE, we used a pre-trained Detic detector (see [Detic repo](https://github.com/facebookresearch/Detic/tree/main)) to detect the 10 more instruments in MUSIC-21 dataset and 28 event classes in AVE dataset. The detected objects for each video are stored in a .npy file.

    c. Extract motion features. We adopt the pretrained I3D video encoder used in [FAME repo](https://github.com/Mark12Ding/FAME). The extracted motion features are stored in a .npy file. (You could refer to iQuery/scripts/extract_motion.py)

3. Data splits. We created index files as .csv for training and testing. Index files for MUSIC/MUSIC-21/AVE datasets can be found at iQuery/data/MUSIC, iQuery/data/MUSIC_21 and iQuery/data/AVE, respectively.

4. Directory structure. We have following directory structure for data:
    ```
    data
    ├── audio
    |   ├── acoustic_guitar
    │   |   ├── M3dekVSwNjY.wav
    │   |   ├── ...
    │   ├── trumpet
    │   |   ├── STKXyBGSGyE.wav
    │   |   ├── ...
    │   ├── ...
    |
    └── frames
    |   ├── acoustic_guitar
    │   |   ├── M3dekVSwNjY.mp4
    │   |   |   ├── 000001.jpg
    │   |   |   ├── ...
    │   |   ├── ...
    │   ├── trumpet
    │   |   ├── STKXyBGSGyE.mp4
    │   |   |   ├── 000001.jpg
    │   |   |   ├── ...
    │   |   ├── ...
    │   ├── ...
    |
    └── detection_results
    |   ├── acoustic_guitar
    │   |   ├── M3dekVSwNjY.mp4.npy
    │   |   ├── ...
    │   ├── trumpet
    │   |   ├── STKXyBGSGyE.mp4.npy
    │   |   ├── ...
    │   ├── ...
    |
    └── motion_features
    |   ├── acoustic_guitar
    │   |   ├── M3dekVSwNjY.mp4.npy
    │   |   ├── ...
    │   ├── trumpet
    │   |   ├── STKXyBGSGyE.mp4.npy
    │   |   ├── ...
    │   ├── ...
    ```

## Training
1. Train the full iQuery model 
    a. on MUSIC dataset
    ```bash
    cd code
    bash scripts/train_music.sh
    ```  
    b. on MUSIC-21 dataset
    ```bash
    cd code
    bash scripts/train_music21.sh
    ```     
    c. on AVE dataset
    ```bash
    cd code
    bash scripts/train_ave.sh
    ```     
2. Train iQuery model without motion
    ```bash
    cd code
    bash scripts/train_nomotion.sh
    ```     

## Evaluation
Please use the following script to evaluate iQuery's performance, you should only modify the pretrained model id and main file name in the script:
```bash
cd code
bash scripts/evaluate.sh
```     

## Citation
```bash
@inproceedings{chen2023iquery,
  title={iQuery: Instruments as Queries for Audio-Visual Sound Separation},
  author={Chen, Jiaben and Zhang, Renrui and Lian, Dongze and Yang, Jiaqi and Zeng, Ziyao and Shi, Jianbo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14675--14686},
  year={2023}
}
```

### Acknowledgement
Our code is based on the implementation of [Sound-of-Pixels](https://github.com/hangzhaomit/Sound-of-Pixels), [CCoL](https://github.com/YapengTian/CCOL-CVPR21), and [AVE](https://github.com/ly-zhu/self-supervised-motion-representations). We sincerely thanks those authors for their great works. If you use our codes, please also consider cite their nice works.

## Contact
If you have any question about this project, please feel free to contact jic088@ucsd.edu. 
