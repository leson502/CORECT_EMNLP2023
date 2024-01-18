import argparse
from numpy.lib.twodim_base import diag

from tqdm import tqdm
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import corect


log = corect.utils.get_logger()
sbert_model = SentenceTransformer("paraphrase-distilroberta-base-v1")

def get_iemocap():
    corect.utils.set_seed(args.seed)

    if args.dataset == "iemocap":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
        ) = pickle.load(
            open(args.data_root + "/data/iemocap/IEMOCAP_features.pkl", "rb"), encoding="latin1"
        )
    elif args.dataset == "iemocap_4":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
        ) = pickle.load(
            open(args.data_root + "/data/iemocap_4/IEMOCAP_features_4.pkl", "rb"), encoding="latin1"
        )

    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        train.append(
            {
                "vid" : vid,
                "speakers" : video_speakers[vid],
                "labels" : video_labels[vid],
                "audio" : video_audio[vid],
                "visual" : video_visual[vid],
                "text": sbert_model.encode(video_sentence[vid]),
                "sentence" : video_sentence[vid],
            }
        )
    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(
            {
                "vid" : vid,
                "speakers" : video_speakers[vid],
                "labels" : video_labels[vid],
                "audio" : video_audio[vid],
                "visual" : video_visual[vid],
                "text": sbert_model.encode(video_sentence[vid]),
                "sentence" : video_sentence[vid],
            }
        )
    for vid in tqdm(test_vids, desc="test"):
        test.append(
            {
                "vid" : vid,
                "speakers" : video_speakers[vid],
                "labels" : video_labels[vid],
                "audio" : video_audio[vid],
                "visual" : video_visual[vid],
                "text": sbert_model.encode(video_sentence[vid]),
                "sentence" : video_sentence[vid],
            }
        )
    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test

def get_mosei():

    mosei_path = args.data_dir
    corect.utils.set_seed(args.seed)

    feature_path = "categorical.p"
    path = os.path.join(mosei_path, feature_path)
    (
        video_ids,
        video_speakers,
        video_labels,
        video_text,
        video_audio,
        video_visual,
        video_sentence,
        trainVids,
        test_vids,
    ) = pickle.load(open(path, "rb"), encoding="latin1")

    label_count = []
    len_count = []
    trainVids = np.array(trainVids)
    test_vids = np.array(test_vids)
    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        train.append(
            corect.Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
        label_count.extend(video_labels[vid])
        len_count.append(len(video_speakers[vid]))

    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(
            corect.Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
    for vid in tqdm(test_vids, desc="test"):
        test.append(
            corect.Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )

    total = len(label_count)
    pos = sum(label_count)
    neg = total - pos

    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test



def main(args):
    if args.dataset == "iemocap":
        train, dev, test = get_iemocap()
        data = {"train": train, "dev": dev, "test": test}
        corect.utils.save_pkl(data, args.data_root + "/data/iemocap/data_iemocap.pkl")
    if args.dataset == "iemocap_4":
        train, dev, test = get_iemocap()
        data = {"train": train, "dev": dev, "test": test}
        corect.utils.save_pkl(data, args.data_root + "/data/iemocap_4/data_iemocap_4.pkl")


    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["iemocap", "iemocap_4"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Dataset directory"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
    )
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    args = parser.parse_args()

    main(args)
