# [ECCV'24] ProGraph：Temporal-alignable Probability Guided Graph Topological Modeling for 3D Human Reconstruction
### [Project Page](http://124.70.164.141:8081/) | [Model Checkpoint](https://huggingface.co/Yanggenfan/ProGraph/tree/main)


![figure4](./assert/first.png)




---

## Overview
Current 3D human motion reconstruction methods from monocular videos rely on features within the current reconstruction window, leading to distortion and deformations in the human structure under local occlusions or blurriness in video frames. To estimate realistic 3D human mesh sequences based on incomplete features, we propose Temporally-alignable Probability Guided Graph Topological Modeling for 3D Human Reconstruction (ProGraph). For missing parts recovery, we exploit the explicit topological-aware probability distribution across the entire motion sequence. To restore the complete human, Graph Topological Modeling (GTM) learns the underlying topological structure, focusing on the relationships inherent in the individual parts. Next, to generate blurred motion parts, Temporal-alignable Probability Distribution (TPDist) utilizes the GTM to predict features based on distribution. This interactive mechanism facilitates motion consistency, allowing the restoration of human parts. Furthermore, Hierarchical Human Loss (HHLoss) constrains the probability distribution errors of inter-frame features during topological structure variation. Our Method achieves superior results than other SOTA methods in addressing occlusions and blurriness on 3DPW.

![overall_architecture](./assert/pipeline.png)


---

## Graph Topological Modeling (GTM)
![mapping_architecture](./assert/mapping.png)


---
## Temporally-alignable Probability Distribution (TPDist)
![TPDist_architecture](./assert/TPDist.png)

---

## Installation
We provide two ways to install conda environments depending on CUDA versions. 

```bash
git clone https://github.com/3DHumanRehab/ProGraph.git
cd Prograph
pip install -r requirements.txt
```

---

## Download
We provide guidelines to download pre-trained models.

Download pre-trained model and put it into the models folder [Prograph-checkpoint.zip](https://huggingface.co/Yanggenfan/P-BET/resolve/main/P-BET-checkpoint.zip?download=true). 

Download all models and put them into the models folder    [Downloads_folder_models](https://huggingface.co/Yanggenfan/ProGraph/tree/main/models)


<a name="model_checkpoint"></a>

- Model checkpoints were obtained in Conda Environment (CUDA 11.7)

---

## Demo
We provide guidelines to run end-to-end inference on test video.

The following command will run ProGraph on video in the specified `--video_file_or_path`.

```bash
python demo.py  --video video/video.mp4
```

---

## Experiments
We provide guidelines to train and evaluate our model on Human3.6M, 3DPW and FreiHAND. 

## Ablation Study
![ablation_study](./assert/ablation_study.png)


Ablation in TPDist and HHLoss.
## Comparative Study
![Comparative_study](./assert/compare_study.png)


Comparison with Fastmetro, GLoT, and PyMAF.


---

## Results
This repository provides several experimental results:
<div style="display: flex; justify-content: space-between;">
    <img src="./assert/1_pro -original-original1.gif" width="32%" height="auto">
    <img src="./assert/2_pro -original-original (2).gif" width="32%" height="auto">
    <img src="./assert/5_pro -original-original.gif" width="32%" height="auto">
</div>
<div style="display: flex; justify-content: space-between;">
    <img src="./assert/2_pro -original-original (1).gif" width="32%" height="auto">
    <img src="./assert/3_pro -original-original (1).gif" width="32%" height="auto">
    <img src="./assert/4_pro -original-original.gif" width="32%" height="auto">
</div>
<div style="display: flex; justify-content: space-between;">
    <img src="./assert/4_1_pro -original-original.gif" width="32%" height="auto">
    <img src="./assert/2_pro -original-original.gif" width="32%" height="auto">
    <img src="./assert/3_pro -original-original.gif" width="32%" height="auto">
</div>

![table2](./assert/compare.png)
Comparison of intra-frame prediction results
![table2](./assert/compare1.png)
omparison of inter-frame prediction results
![figure1](./assert/human4d_hightlight.png)
Comparison with 4DHumans.
![intro1](./assert/occur.png)
The reconstruction results and probability distribution of human body parts during the prediction process.


---

## Acknowledgments

Our repository is modified and adapted from these amazing repositories. If you find their work useful for your research, please also consider citing them:
- [METRO](https://github.com/microsoft/MeshTransformer)          
- [MeshGraphormer](https://github.com/microsoft/MeshGraphormer)
- [HMR](https://github.com/akanazawa/hmr)
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)
- [GLOT](https://github.com/sxl142/GLoT)
