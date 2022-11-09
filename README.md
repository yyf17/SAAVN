# SAAVN
# SAAVN Code release for paper "Sound Adversarial Audio-Visual Navigation,ICLR2022" (In PyTorch)

## What we do?

### Motivation

Contribution of SoundSpaces:
- Build an audio simulation platform SoundSpaces[1] to enable audio-visual navigation for two visually realistic 3D environments: Replica[2] and Matterport3D[3].
- Proposed AudioGoal navigation Task:This task requires a robot equipped with a camera and microphones to interact with the environment and navigate to a sounding object. 
- SoundSpaces dataset: SoundSpaces is a first-of-its-kind dataset of audio renderings based on geometrical acoustic simulations for two sets of publicly available 3D environments: Replica[2] and Matterport3D[3].
### Characteristic of SoundSpaces

Sumary:SoundSpaces is focus on audio-visual navigation problem in the acoustically clean or simple environment:
- The number of target sound sources is one. 
- The position of the target sound source is fixed in an episode of a scene. 
- The volume of the target sound source is the same in all episodes of all scenes, and there is no change.

All in all, the sound in the setting of SoundSpaces is  acoustically clean or simple.

### Challenge

However,there are many situations different from the setting of SoundSpaces , which there are some non-target sounding objects in the scene:
For example, a kettle in the kitchen beeps to tell the robotthat the water is boiling, and the robot in the living room needs to navigate to the kitchen and turnoff the stove; while in the living room, two children are playing a game, chuckling loudly fromtime to time.

#### Challenge 1: 
Can an agent still find its way to the destination without being distracted by all non-target sounds around the agent? 

non-target sounding objects:
- not deliberately embarrassing the robot: someone walking and chatting past the robot
- deliberately embarrassing the robot: someone blocking the robot forwarding

#### Challenge 2: 

How to model non-target sounding objects in simulator or in reality?  There are no such setting existed!

### Solution policy

- Worst case strategy: Regard non-target sounding objects as deliberately embarrassing the robot,we called them as sound attacker.
- Simplify:Only consider the simplest situation,one sound attacker.
- Zero sum game:One agent,one sound attacker.



![SAAVN](saavn.png)
---------------------------------------------------------------------------------------------------

## These code are under cleaning! Some of bugs maybe happen, please tell me if you have any trouble.

## Thanks

These codes are based on the [SoundSpaces](https://github.com/facebookresearch/sound-spaces) code base.

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

## Citation
If you use this model in your research, please cite the following paper:
```
@inproceedings{YinfengICLR2022saavn,
  author    = {Yinfeng Yu and
               Wenbing Huang and
               Fuchun Sun and
               Changan Chen and
               Yikai Wang and
               Xiaohong Liu},
  title     = {Sound Adversarial Audio-Visual Navigation},
  booktitle = {The Tenth International Conference on Learning Representations, {ICLR}
               2022, Virtual Event, April 25-29, 2022},
  publisher = {OpenReview.net},
  year      = {2022},
  url       = {https://openreview.net/forum?id=NkZq4OEYN-},
  timestamp = {Thu, 18 Aug 2022 18:42:35 +0200},
  biburl    = {https://dblp.org/rec/conf/iclr/Yu00C0L22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
