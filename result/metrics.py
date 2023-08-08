import glob
import numpy as np
from mir_eval.separation import bss_eval_sources
from pesq import pesq
from pystoi.stoi import stoi


if __name__ == '__main__':
    name = r'test' #实验名称
    root_path = r'./' + name + '/' #模型训练过程中保存的语音
    npys = glob.glob(root_path + '*.npy')

    idxs = [10] #？只计算gt0和pred0
    for idx in idxs:
        gt_file = root_path + 'gt' + str(10 * idx) + '.npy'
        pred_file = root_path + 'pred' + str(10 * idx) + '.npy'
        gt = np.load(gt_file)
        pred = np.load(pred_file)

        smixs = []
        sdrs = []
        sirs = []
        sars = []
        pesqs = []
        stois = []
        cnt_error = 0
        start_idx = 0
        for i in range(start_idx, gt.shape[0], 1):
            # if i == 50:
            #     continue
            # if (idx == 0):
            #     sdr_mix, _, _, _ = bss_eval_sources(gt[i], pred[i], False)
            #     smixs.append(sdr_mix)
            #try:
            sdr, sir, sar, _ = bss_eval_sources(gt[i], pred[i], True)
            sdrs.append(np.mean(sdr))
            sirs.append(np.mean(sir))
            sars.append(np.mean(sar))

            for gt_wav, pred_wav in zip(gt[i], pred[i]):
                pesqs.append(pesq(16000, gt_wav, pred_wav, 'nb'))
                stois.append((float)(stoi(gt_wav, pred_wav, 16000, extended=False)))
            print(str(idx) + ',' + str(i) + ':' + str(sdrs[i - start_idx]))
            with open(root_path + name + '_detail.txt', 'a+') as f:
                    f.write(str(idx) + ',' + str(i) + ':' + str(sdrs[i - start_idx]) + '\n')
            #except:
            #    #cnt_error += 1
            #    print('error')

        with open(root_path + name + '.txt', 'a+') as f:
            f.write('{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(
                idx,
                np.mean(sdrs),
                np.mean(sirs),
                np.mean(sars),
                np.mean(pesqs),
                np.mean(stois)))
        print(str(idx) + ' fininshed!\n')
    print('\n')
