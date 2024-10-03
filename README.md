# CARe_experiments
This repository contains our experiments under openmask3d setting reported in the paper "Context-Aware Replanning with Pre-Explored Semantic Map for Object Navigation".

---
**Context-Aware Replanning with Pre-Explored Semantic Map for Object Navigation**

[Hung-Ting Su*](https://htsucml.github.io/), [Ching-Yuan Chen*](https://care-maps.github.io/), [Po-Chen Ko*](https://pochen-ko.github.io/), [Jia-Fong Yeh](https://www.cmlab.csie.ntu.edu.tw/~jiafongyeh), [Min Sun](https://aliensunmin.github.io/), [Winston H. Hsu](https://winstonhsu.info/)

---

## Setup
To setup the openmask3d enviroment, first follow [their Setup instructions](https://github.com/OpenMask3D/openmask3d):

Then, update openmask3d to our version by

```bash
conda activate openmask3d
pip install -r requirements.txt
```

## Obtaining Map

### Preprocessing
We preprocess the data to put everything in the correct position for OpenMask3d. First, put the MatterPort3D dataset under `./datset/`

The resulting dataset structure should be 
```
dataset/
    |--scans/
        |---{scan_ID_1}/
        |   |---region_segmentations/
        |   |---undistorted_camera_parameters/
        |   |---undistorted_color_images/
        |   |---indistorted_depth_images/
        |        
        |---{scan_ID_2}/

        ...
```
Next, run `preprocess_mp3d.py`
```
python preprocess_mp3d.py
```

### Running OpenMask3D
Before running OM3D, you need to download their pretrained models for 3D mask proposal. For your convenience, we provide `download_on3d_models.sh` which downloads the pretrained model checkpoints pnd puts them in the correct place (under `./resources/`).
```
bash download_om3d_models.sh
```


To obtain OpenMask3D semantic map, run `extract_mp3d_feature.py`
```
python extract_mp3d_feature.py
```

## Experiment
Finally, the OpenMask3D experiment with the introduced replanning strategies can be exucuted by 
```
python evaluate_mp3d_top_category.py
python evaluate_mp3d_top_confidence.py
```

The results will be saved as "topk_confidence_replanning_raw.json" and "topk_categoty_replanning_raw.json", respectively.




## Acknowledgement
This project is developed from the following repositories:
- [openmask3d](https://github.com/OpenMask3D/openmask3d)
