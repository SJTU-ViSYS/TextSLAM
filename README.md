# TextSLAM: Visual SLAM with Semantic Planar Text Features

**Authors**: [Boying Li](https://leeby68.github.io/), [Danping Zou](https://drone.sjtu.edu.cn/dpzou/), Yuan Huang, Xinghan Niu, Ling Pei and Wenxian Yu.

🏠 [[Project]](https://leeby68.github.io/TextSLAM/)
&emsp;
📝 [[Paper]](https://ieeexplore.ieee.org/abstract/document/10285400)
&emsp;
➡️ [[Dataset]](https://github.com/SJTU-ViSYS/TextSLAM-Dataset/)
&emsp;
🔧 [[Extra Evaluation Tool]](https://github.com/SJTU-ViSYS/SLAM_Evaluation/)


**Motivation:**

:star: TextSLAM is a novel visual Simultaneous Localization and Mapping system (SLAM) tightly coupled with semantic text objects.

:bulb: Humans can read texts and navigate complex environments using scene texts, such as road markings and room names. why not robots?

:star: TextSLAM explores scene texts as the basic feature both geometrically and semantically. It achieves superior performance even under challenging environments, such as image blurring, large viewpoint changes, and significant illumination variations (day and night).

This repository provides **C++ implementation of TextSLAM system**.

<img src="./pic/Image_TextSLAM.png"  width ="800" align = "center" /> 
<em><div align='center'>Overview of TextSLAM</div></em>
&emsp;   

Our **accompanying videos** are now available on YouTube (click below images to open) and Bilibili<sup>[1-outdoor](https://www.bilibili.com/video/BV1pe411B7kx/?spm_id_from=333.999.0.0&vd_source=404d99588f2e4c0ce1cca75ed492e620), [2-night](https://www.bilibili.com/video/BV1kC4y1M7tk/?spm_id_from=333.999.0.0&vd_source=404d99588f2e4c0ce1cca75ed492e620), [3-rapid](https://www.bilibili.com/video/BV1Au4y1T7DE/?spm_id_from=333.999.0.0&vd_source=404d99588f2e4c0ce1cca75ed492e620)</sup>.
<div align="center">
<a href="https://youtu.be/ug-FvJKTXJY" target="_blank"><img src="https://github.com/SJTU-ViSYS/TextSLAM/blob/main/pic/TextSLAM-frontPage.png" alt="video" width="32%" /></a>
<a href="https://youtu.be/PYrZ5kiIC0Q" target="_blank"><img src="https://github.com/SJTU-ViSYS/TextSLAM/blob/main/pic/TextSLAM-frontPage.png" alt="video" width="32%" /></a>
  <a href="https://youtu.be/3Ml6070Hgd8" target="_blank"><img src="https://github.com/SJTU-ViSYS/TextSLAM/blob/main/pic/TextSLAM-frontPage.png" alt="video" width="32%" /></a>
</div>

:star: Please consider citing the following papers in your publications if the project helps your work.
```
@article{li2023textslam,
  title={TextSLAM: Visual SLAM with Semantic Planar Text Features},
  author={Li, Boying and Zou, Danping and Huang, Yuan and Niu, Xinghan and Pei, Ling and Yu, Wenxian},
  booktitle={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2023}
}

@inproceedings{li2020textslam,
  title={TextSLAM: Visual SLAM with Planar Text Features},
  author={Li, Boying and Zou, Danping and Sartori, Daniele and Pei, Ling and Yu, Wenxian},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2020}
}
```

# Getting Start

## Dataset Download

Download the dataset from [**TextSLAM Dataset**](https://github.com/SJTU-ViSYS/TextSLAM-Dataset/).

## 1. Prerequisites

TextSLAM is run in Ubuntu 16.04. It should be easy to compile in other Linux system versions.

**1.1 Ceres & Eigen3** 

Refer to [**Ceres**](http://ceres-solver.org/installation.html#linux) for installing it in Linux.

During the above process, **Eigen3** is also installed at the same time.

**1.2 OpenCV**

We use **OpenCV** 3.3.1 for image processing.

You can use the **OpenCV library** provided by ROS. Remember to set OpenCV_DIR in CMakeLists.txt using `set(OpenCV_DIR [ros_direction]/share/OpenCV-3.3.1-dev)`.

You can also refer to [**OpenCV**](http://opencv.org) to download and install the library.

**1.3 EVO (Evaluation)**

EVO is used for SLAM results evaluation. Refer to [**EVO**](https://github.com/MichaelGrupp/evo) to install this evaluation tool.

<!-- 2.2 Build-->
<!-- git clone & build -->
<!-- 2.3 Run -->
<!-- yaml, how to run, results meanings -->
## 2. Build and Run

**2.1 Clone the repository and build the project:**
```
git clone https://github.com/SJTU-ViSYS/TextSLAM.git
mkdir build
cd build
cmake ..
make -j
```
<!-- cmake .. -DCMAKE_BUILD_TYPE=Release -->
Above procedure will create an executable named TextSLAM.

**2.2 Run TextSLAM with:**
```
./TextSLAM [yaml_path]/[yaml_name].yaml
```
We provide yaml files (`GeneralMotion.yaml`,`AIndoorLoop.yaml`, `LIndoorLoop.yaml`, `Outdoor.yaml`) for our 4 kinds of experiments. 
Write your sequence save path in 'Exp read path:' of the yaml file.

Refer to [TextSLAM Dataset](https://github.com/SJTU-ViSYS/TextSLAM-Dataset/) for a detail yaml file structure.

**2.3 Output:**
`keyframe_latest.txt` will output to record each keyframe pose estimation results in the current station.
`keyframe.txt` will output when finishing a sequence.
Both `keyframe_latest.txt` and `keyframe.txt` are in TUM format with ` timestamp tx ty tz qx qy qz qw `.

## 3. Evaluation

We use EVO to evaluate the SLAM performance.

**For APE evaluation:**
```
evo_ape tum gt.txt text.txt -va -s
```

**For RPE evaluation at the uint of 1.0 m:** 
```
evo_rpe tum gt.txt text.txt -va -s --pose_relation trans_part -d 1.0 -u m
```

For the loop tests in a large indoor scene, add ` --n_to_align XX ` to align the first XX pose of the whole trajectory.
Because GT for this sequence is only at the beginning and the end, using the alignment for the first poses will get the more correct results.
```
evo_ape tum gt.txt text.txt -va -s --n_to_align XX
evo_rpe tum gt.txt text.txt -va -s --pose_relation trans_part -d 1.0 -u m --n_to_align XX
```

:heavy_exclamation_mark: **ATTENTION for RPE evaluation:**  :heavy_exclamation_mark:

EVO does not automatically rectify the misalignment between the SLAM body frame and the ground-truth body, which influences RPE results.

To solve this problem, we provide an [extra Evaluation tool](https://github.com/SJTU-ViSYS/SLAM_Evaluation) for TextSLAM dataset, which also served as **a supplement for EVO**.

Following the instruction of the [extra Evaluation tool](https://github.com/SJTU-ViSYS/SLAM_Evaluation) to first obtain the **updated pose ground truth** file, and then use the updated GT file to evaluate the RPE results.

This step is necessary for **all data except outdoor sequences**. 
We use COLMAP to generate outdoor sequences' ground truth, which generates the same ground truth frame as the SLAM estimated body frame.

# Acknowledgement

The authors thank [ORB-SLAM](https://github.com/raulmur/ORB_SLAM2), [DSO](https://github.com/JakobEngel/dso), and [AttentionOCR](https://github.com/zhang0jhon/AttentionOCR) for their excellent works.
The authors thank [EVO](https://github.com/MichaelGrupp/evo) for providing this convenient evaluation tool.
The authors thank [Ceres](http://ceres-solver.org) for providing this powerful optimization library.
