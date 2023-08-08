# SSWMNetwork
Source code of paper "SSWMNet: Solving The Problem Of Speech Separation While Wearing a Mask".

# Model

![](C:\Users\mfm\Desktop\b300047216e2fa4ca6821184425ed0c.png)

# Dataset

## SSWM Dataset

Due to copyright issues, we can not directly disclose our dataset, but we will publish the features extracted using our model and some samples later.

## Prepare Data

Due to copyright issues, we can not directly disclose our dataset, but we will publish the features extracted using our model and some samples later.

We hope the data to be categorized by person, that is, the videos of a certain speaker are in the same directory. First, using ffmpeg to cut the videos into video frames and audios, and storing the audios into audio directory and storing the video frames into frames directory, still categorized by person. The directory structure is as follows.

```
-dataset
	-audio
		-speaker1
		-speaker2
		-speaker3
		...
	-frames
		-speaker1
		-speaker2
		-speaker3
		...
```

Second, through `python scripts/create_index_files_nannv.py` creates a csv file contaning addresses which store speaker visual frames and audio. 

# Training

```
python main.py 
```

All parameters are included in `main.py` and can be changed according to demand.

