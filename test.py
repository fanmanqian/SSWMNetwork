# @Author : cheertt
# @Time   : 19-11-14 下午3:51
# @Remark :

from pystoi.stoi import stoi
import librosa


if __name__ == '__main__':
    gt1, sr = librosa.load('/home/cheertt/codes/instrument/motion_unet_film/ckpt/1114-debugTime-2mix-LogFreq-resnet18dilated-unet6-linear-frames3stride16-maxpool-binary-channels32-epoch40-step40_60/visualization/guguoning-20180717_110+guguoning-20180717_227/gt1.wav', sr=16000)
    gt2, _ = librosa.load('/home/cheertt/codes/instrument/motion_unet_film/ckpt/1114-debugTime-2mix-LogFreq-resnet18dilated-unet6-linear-frames3stride16-maxpool-binary-channels32-epoch40-step40_60/visualization/guguoning-20180717_110+guguoning-20180717_227/pred1.wav', sr=16000)

    d = stoi_score = stoi(gt1, gt2, sr, extended=False)
    print(d)

