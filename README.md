# SAAVN
# SAAVN Code release for paper "Sound Adversarial Audio-Visual Navigation" (In PyTorch)

## Usage
This repo supports AudioGoal Task on Replica and Matterport3D datasets.

Below we show the commands for training and evaluating AudioGoal with Depth sensor on Replica, 
but it applies to Matterport dataset as well. 
1. Training
```
python main.py --default av_nav --run-type train --exp-config [exp_config_file] --model-dir data/models/replica/av_nav/e0000/audiogoal_depth --tag-config [tag_config_file] TORCH_GPU_ID 0 SIMULATOR_GPU_ID 0
```
2. Validation (evaluate each checkpoint and generate a validation curve)
```
python main.py --default av_nav --run-type eval --exp-config [exp_config_file] --model-dir data/models/replica/av_nav/e0000/audiogoal_depth --tag-config [tag_config_file] TORCH_GPU_ID 0 SIMULATOR_GPU_ID 0
```
3. Test the best validation checkpoint based on validation curve
```
python main.py --default av_nav --run-type eval --exp-config [exp_config_file] --model-dir data/models/replica/av_nav/e0000/audiogoal_depth --tag-config [tag_config_file] TORCH_GPU_ID 0 SIMULATOR_GPU_ID 0
```
4. Generate demo video with audio
```
python main.py --default av_nav --run-type eval --exp-config [exp_config_file] --model-dir data/models/replica/av_nav/e0000/audiogoal_depth --tag-config [tag_config_file] TORCH_GPU_ID 0 SIMULATOR_GPU_ID 0
```

Note: [exp_config_file] is the main parameter configuration file of the experiment, while [tag_config_file] is special parameter configuration file for abalation experiments.

# Cite

@inproceedings{YinfengICLR2022saavn,
	title = {Sound Adversarial Audio-Visual Navigation},
	author = {Yinfeng Yu, Wenbing Huang, Fuchun Sun, Changan Chen, Yikai Wang, Xiaohong Liu},
	year = {2022},
    booktitle={ICLR},
}
