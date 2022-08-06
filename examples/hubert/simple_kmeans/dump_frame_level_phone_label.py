# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np
import tgt
import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_frame_level_phone_label")

def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]
        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)
    return iterate, len(lines)

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

def get_grids(filename, tier_name=None, include_empty_intervals=True, short_format=False):
    g = tgt.io.read_textgrid(filename, include_empty_intervals=include_empty_intervals)

    if short_format:
        tier = g.tiers[0]
    else:
        assert tier_name is not None
        tier = g.get_tier_by_name(tier_name)

    grids = []
    for i in tier.intervals:
        grids.append([i.start_time, i.end_time, i.text])
    return grids

def dump_grids(textgrid_filename, word_durs=None, phone_durs=None):

    textgrid = tgt.core.TextGrid()

    if word_durs is not None:
        interval_tier = tgt.core.IntervalTier(name="words")
        intervals = []
        for word_dur in word_durs:
            interval = tgt.core.Interval(word_dur[0], word_dur[1], word_dur[2])
            intervals.append(interval)
        interval_tier.add_intervals(intervals)
        textgrid.add_tier(interval_tier)

    if phone_durs is not None:
        interval_tier = tgt.core.IntervalTier(name="phones")
        intervals = []
        for phone_dur in phone_durs:
            interval = tgt.core.Interval(phone_dur[0], phone_dur[1], phone_dur[2])
            intervals.append(interval)
        interval_tier.add_intervals(intervals)
        textgrid.add_tier(interval_tier)

    tgt.io.write_to_file(textgrid, textgrid_filename, format='long')

def sample_frame_label(gridpath, end, win_len, hop_len, sample_rate):
    default_phoneme = "sil"
    grids = get_grids(gridpath, 'phones')
    if end == -1:
        start = grids[0][0]
        end = grids[-1][1]

    j, phonemes = 0, []

    current_time = 0.5 * win_len / sample_rate
    move_time = 1.0 * hop_len / sample_rate    # hop_len / sample_rate 每次移动的时长

    while current_time <= end:
        t = current_time

        if t >= start:
            if j >= len(grids):
                phonemes.append(default_phoneme)  # sil

            while j < len(grids):
                if grids[j][0] <= t and t < grids[j][1]:
                    if grids[j][2] != "":
                        phonemes.append(grids[j][2])
                    else:
                        phonemes.append(default_phoneme)
                    break
                j += 1

        current_time += move_time
    return phonemes

def gen_label(gridpath):
    grids = get_grids(gridpath, 'phones')
    phonemes = []
    for g in grids:
        phonemes.append(g[-1])
    return phonemes

def dump_label(tsv_dir, split, label, down_sample_rate, nshard, rank, lab_dir, frame_wise):

    generator, num = get_path_iterator(f"{tsv_dir}/{split}.pho", nshard, rank)
    iterator = generator()

    lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.{label}"
    os.makedirs(lab_dir, exist_ok=True)
    with open(lab_path, "w") as f:
        for path, nsample in tqdm.tqdm(iterator, total=num):
            # 由于音频采样率可能不固定，这里不使用采样点数得到音频结束时间，而是直接使用TextGrid中的结束时间
            # '[(512,10,5)] + [(512,3,5)] + [(512,3,2)] + [(512,2,2)] * 2'
            # win_len = 220 （ F_i =（F_{i+1} - 1）* strid_i + width_i）, hop_len = 200 (stride累乘)
            if frame_wise:
                lab = sample_frame_label(path, end=-1, win_len=220, hop_len=down_sample_rate, sample_rate=16000)
            else:
                lab = gen_label(path)
            f.write(" ".join(map(str, lab)) + "\n")
            f.flush()
    logger.info("finished successfully")


def thread_dump_label(tsv_dir, split, label, down_sample_rate, nshard, lab_dir, frame_wise):
    import threading
    threads = []
    for rank in range(nshard):
        threads.append(threading.Thread(target=dump_label, args=(tsv_dir, split, label, down_sample_rate, nshard, rank, lab_dir, frame_wise)))

    for rank in range(nshard):
        threads[rank].start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("down_sample_rate", type=float)  #降采样倍数
    parser.add_argument("nshard", type=int)
    parser.add_argument("lab_dir")
    parser.add_argument("frame_wise", type=int, default=1)  #1表示帧级别特征
    parser.add_argument("label", type=str, default="ph")    #标签类型后缀

    args = parser.parse_args()
    logging.info(str(args))

    #dump_label(**vars(args))
    thread_dump_label(**vars(args))
