# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np

import joblib
import torch
import tqdm

import torch.nn.functional as F
import fairseq
from feature_utils import get_path_iterator
from fairseq.data.audio.audio_utils import get_features_or_waveform


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("forward_hubert_dump_km_label")

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
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=self.task.cfg.sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)

class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_feat_iterator(feat_dir, split, nshard, rank):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    def iterate():
        feat = np.load(feat_path, mmap_mode="r")
        assert feat.shape[0] == (offsets[-1] + lengs[-1])
        for offset, leng in zip(offsets, lengs):
            yield feat[offset: offset + leng]

    return iterate, len(lengs)


def dump_label(ckpt_path, layer, tsv_dir, split, km_path, nshard, rank, lab_dir, max_chunk):
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    apply_kmeans = ApplyKmeans(km_path)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    iterator = generator()

    lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.km"
    os.makedirs(lab_dir, exist_ok=True)
    with open(lab_path, "w") as f:
        for path, nsample in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path, nsample)
            if torch.cuda.is_available():
                feat = torch.from_numpy(feat).cuda()
            lab = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, lab)) + "\n")
            f.flush()
    logger.info("finished successfully")

def thread_dump_label(ckpt_path, layer, tsv_dir, split, km_path, nshard, rank, lab_dir, max_chunk):
    import threading
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    apply_kmeans = ApplyKmeans(km_path)

    def sing_job(tsv_dir, split, nshard, rank, lab_dir):
        generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
        iterator = generator()

        lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.km"
        os.makedirs(lab_dir, exist_ok=True)
        with open(lab_path, "w") as f:
            for path, nsample in tqdm.tqdm(iterator, total=num):
                feat = reader.get_feats(path, nsample)
                # feat = torch.from_numpy(feat).cuda()
                lab = apply_kmeans(feat).tolist()
                f.write(" ".join(map(str, lab)) + "\n")
                f.flush()

    threads = []
    for rank in range(nshard):
        threads.append(threading.Thread(target=sing_job, args=(tsv_dir, split, nshard, rank, lab_dir)))

    for rank in range(nshard):
        threads[rank].start()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path")
    parser.add_argument("layer", type=int)
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("km_path")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("lab_dir")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logging.info(str(args))

    #dump_label(**vars(args))
    thread_dump_label(**vars(args))
