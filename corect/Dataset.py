import math
import random
import torch

import numpy as np


class Dataset:
    def __init__(self, samples, args) -> None:
        self.samples = samples
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(len(self.samples) / args.batch_size)
        self.dataset = args.dataset
        self.speaker_to_idx = {"M": 0, "F": 1}

        self.embedding_dim = args.dataset_embedding_dims[args.dataset]

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size : (index + 1) * self.batch_size]

        return batch

    def padding(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s["text"]) for s in samples]).long()
        mx = torch.max(text_len_tensor).item()
        
        text_tensor = torch.zeros((batch_size, mx, self.embedding_dim['t']))
        audio_tensor = torch.zeros((batch_size, mx, self.embedding_dim['a']))
        visual_tensor = torch.zeros((batch_size, mx, self.embedding_dim['v']))

        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        utterances = []
        for i, s in enumerate(samples):
            cur_len = len(s["text"])
            utterances.append(s["sentence"])
            tmp_t = []
            tmp_a = []
            tmp_v = []
            for t, a, v in zip(s["text"], s["audio"], s["visual"]):
                tmp_t.append(torch.tensor(t))
                tmp_a.append(torch.tensor(a))
                tmp_v.append(torch.tensor(v))
                
            tmp_a = torch.stack(tmp_a)
            tmp_t = torch.stack(tmp_t)
            tmp_v = torch.stack(tmp_v)

            text_tensor[i, :cur_len, :] = tmp_t
            audio_tensor[i, :cur_len, :] = tmp_a
            visual_tensor[i, :cur_len, :] = tmp_v

            speaker_tensor[i, :cur_len] = torch.tensor(
                    [self.speaker_to_idx[c] for c in s["speakers"]])

            labels.extend(s["labels"])

        label_tensor = torch.tensor(labels).long()
        

        data = {
            "text_len_tensor": text_len_tensor,
            "text_tensor": text_tensor,
            "audio_tensor": audio_tensor,
            "visual_tensor": visual_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "utterance_texts": utterances,
        }
        return data

    def shuffle(self):
        random.shuffle(self.samples)
