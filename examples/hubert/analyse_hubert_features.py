# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import numpy as np
import fairseq
import torch
import torch.nn.functional as F
import librosa
#from fairseq.data.audio.audio_utils import get_features_or_waveform
import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("analyse_hubert_features")


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.ret_conv = (layer == 0)
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        #wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=self.task.cfg.sample_rate)
        wav, sr = librosa.load(path, sr=16000)
        if sr != 16000:
            wav = librosa.resample(wav, sr, 16000)
        if wav.ndim > 1:
            wav = wav.mean(-1)
        assert wav.ndim == 1
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.cuda().view(1, -1)
            padding_mask = None

            # #for debug
            # waves_padded = torch.zeros(size=(1, 220960), dtype=torch.float, requires_grad=False)
            # waves_padded[0, :x.size(1)] = x
            # #padding_mask = torch.BoolTensor(waves_padded.shape).fill_(False).cuda()
            # #padding_mask[0, x.size(1):] = True
            # x = waves_padded.cuda()
            # #for debug

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=padding_mask,
                    mask=False,
                    ret_conv=self.ret_conv,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)

def cal_canonical_correlation(feat1, feat2):
    if(feat1.shape[1] != feat2.shape[1]):
        min_samples = min(feat1.shape[1], feat2.shape[1])
        feat1 = feat1[:,:min_samples]
        feat2 = feat2[:,:min_samples]

    # cca = CCA(n_components=1)  # 若想计算第二主成分对应的相关系数，则令cca = CCA(n_components=2)
    # cca.fit(feat1, feat2)
    # # 降维操作
    # X_train_r, Y_train_r = cca.transform(feat1, feat2)
    # # 输出相关系数
    # c = np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])
    # numpy_cca = np.mean(c)

    import cca_core
    google_cca = cca_core.robust_cca_similarity(feat1, feat2, epsilon=1e-10,
                                                verbose=True, compute_dirns=False)
    cca = np.mean(google_cca["cca_coef1"])
    return cca

def cal_cosine_sim(vec1, vec2):
    '''
    输入为1维向量
    '''
    cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim

def extract_feature_and_compare(reader, wav_list1, wav_list2):
    feat1_cat = []
    feat2_cat = []
    for i, (wav_path1, wav_path2) in enumerate(zip(wav_list1, wav_list2)):
        feat1 = reader.get_feats(wav_path1).cpu().numpy().T
        feat2 = reader.get_feats(wav_path2).cpu().numpy().T
        feat1_cat.append(feat1)
        feat2_cat.append(feat2)
    feat1 = np.concatenate(feat1_cat, axis=1)
    feat2 = np.concatenate(feat2_cat, axis=1)
    cca = cal_canonical_correlation(feat1, feat2)
    return cca


def main(tsv_dir1, tsv_dir2, ckpt_path, output_dir, text_dependent, max_chunk):
    os.makedirs(output_dir, exist_ok=True)
    f1 = open(tsv_dir1, 'r')
    wav_list1 = [x.strip() for x in f1.readlines()]
    f2 = open(tsv_dir2, 'r')
    wav_list2 = [x.strip() for x in f2.readlines()]
    if not text_dependent:
        import random
        random.seed(777)
        random.shuffle(wav_list2)
    reader = HubertFeatureReader(ckpt_path, 1, max_chunk)
    cca_layers = []
    for layer in range(1, 13):
        reader.layer = layer
        cca_mean = extract_feature_and_compare(reader, wav_list1, wav_list2)
        cca_layers.append(cca_mean)
    suffix = '' if text_dependent else '_shuffle'
    save_name = output_dir + f'/cca{suffix}.png'
    plt.figure()
    time = np.arange(0, 12)
    plt.plot(time, cca_layers)
    plt.savefig(save_name)
    plt.close()

def main_compare_different_layers(tsv_dir1, tsv_dir2, ckpt_path, output_dir, text_dependent, max_chunk):
    os.makedirs(output_dir, exist_ok=True)
    f1 = open(tsv_dir1, 'r')
    wav_list1 = [x.strip() for x in f1.readlines()]
    reader = HubertFeatureReader(ckpt_path, 0, max_chunk)
    cca_layers = []
    cnn_feats = []
    for i, wav_path in enumerate(wav_list1):
        cnn_feat = reader.get_feats(wav_path).cpu().numpy().T
        cnn_feats.append(cnn_feat)
    cnn_feat = np.concatenate(cnn_feats, axis=1)
    for layer in range(1, 13):
        reader.ret_conv = False
        reader.layer = layer
        feats = []
        for i, wav_path in enumerate(wav_list1):
            feat = reader.get_feats(wav_path).cpu().numpy().T
            feats.append(feat)
        feat = np.concatenate(feats, axis=1)
        cca_mean = cal_canonical_correlation(cnn_feat, feat)
        cca_layers.append(cca_mean)
    suffix = '_layers'
    save_name = output_dir + f'/cca{suffix}.png'
    plt.figure()
    time = np.arange(0, 12)
    plt.plot(time, cca_layers)
    plt.savefig(save_name)
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir1")
    parser.add_argument("tsv_dir2")
    parser.add_argument("ckpt_path")
    parser.add_argument("output_dir")
    parser.add_argument("text_dependent", type=int, default=1)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)
    if args.tsv_dir1 != args.tsv_dir2:
        main(**vars(args))
    else:
        main_compare_different_layers(**vars(args))
