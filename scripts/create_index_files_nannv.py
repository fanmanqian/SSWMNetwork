import os
import glob
import argparse
import random
import fnmatch


def find_recursive(root_dir, ext='.wav'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_audio', default='/E/cheertt-data/host_video_3/audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='/E/cheertt-data/host_video_3/frames224',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=8, type=int,
                        help="fps of video frames")
    parser.add_argument('--path_output', default='./data',
                        help="path to output index files")
    parser.add_argument('--trainset_ratio', default=0.8, type=float,
                        help="80% for training, 20% for validation")
    args = parser.parse_args()

    male = ["guguoning", "heyanke", "huangfeng", "mayaguang", "mingercha", "pantao", "sangchaohui", "zhengxuan",
            "zhuguangquan"]
    female = ["baoxiaofeng", "haixia", "hejia", "hudie", "huwancong", "jingyi", "jinyi", "kali",
              "lizimeng", "mulinshan", "nv", "ouyangxiadan", "shiyu", "weixian", "xuke",
              "zhengtianliang"]

    audio_files_male = []
    audio_files_female = []
    audio_files = []
    infos_male = []
    infos_female = []

    for host in os.listdir(args.root_audio):
        if host in male:
            audio_files_male += glob.glob(os.path.join(args.root_audio, host, '*.wav'))
        elif host in female:
            audio_files_female += glob.glob(os.path.join(args.root_audio, host, '*.wav'))

    print('audio_files_male', len(audio_files_male))
    print('audio_files_female', len(audio_files_female))

    for audio_path in audio_files_male:
        frame_path = audio_path.replace(args.root_audio, args.root_frame).replace('.wav', '')
        frame_files = glob.glob(frame_path + '/*.jpg')
        if len(frame_files) == 0:
            continue
        # if len(frame_files) > args.fps * 20:
        infos_male.append(','.join([audio_path, frame_path, str(len(frame_files))]))
    print('{} male audio/frames pairs found.'.format(len(infos_male)))

    for audio_path in audio_files_female:
        frame_path = audio_path.replace(args.root_audio, args.root_frame).replace('.wav', '')
        frame_files = glob.glob(frame_path + '/*.jpg')
        if len(frame_files) == 0:
            continue
        # if len(frame_files) > args.fps * 20:
        infos_female.append(','.join([audio_path, frame_path, str(len(frame_files))]))
    print('{} female audio/frames pairs found.'.format(len(infos_female)))


    infos = []

    for index, value in enumerate(range((len(infos_male) + len(infos_female)) * 5)):
        random.seed(index)
        index1 = random.randint(0, len(infos_male) - 1)
        index2 = random.randint(0, len(infos_female) - 1)
        # print(int(infos_male[index_male][-1]))
        # print(infos_male[index_male])
        # print(infos_female[index_female][-1])
        if int(infos_male[index1].split(',')[-1]) > args.fps * 4 and int(
                infos_female[index2].split(',')[-1]) > args.fps * 4:
            tmp = infos_male[index1] + '|' + infos_female[index2]
            infos.append(tmp)

    # find all audio/frames pairs
    # infos = []
    # audio_files = find_recursive(args.root_audio, ext='.wav')
    # for audio_path in audio_files:
    #     host = audio_path.split(args.root_audio)[1].split('/')[1]
    #     if host in female:
    #         frame_path = audio_path.replace(args.root_audio, args.root_frame) \
    #                                .replace('.wav', '')
    #         frame_files = glob.glob(frame_path + '/*.png')
    #         if len(frame_files) == 0:
    #             continue
    #         if len(frame_files) > args.fps * 20:
    #             infos.append(','.join([audio_path, frame_path, str(len(frame_files))]))
    # print('{} audio/frames pairs found.'.format(len(infos)))

    # split train/val
    n_train = int(len(infos) * 0.8)
    random.shuffle(infos)
    trainset = infos[0:n_train]
    valset = infos[n_train:]
    for name, subset in zip(['train_nannv', 'val_nannv'], [trainset, valset]):
        if not os.path.exists(args.path_output):
            os.makedirs(args.path_output)
        filename = '{}.csv'.format(os.path.join(args.path_output, name))
        with open(filename, 'w') as f:
            for item in subset:
                f.write(item + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))

    print('Done!')
