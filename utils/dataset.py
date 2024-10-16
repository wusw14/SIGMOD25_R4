import random
from collections import defaultdict
from copy import deepcopy
from typing import List

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils import data
from transformers import AutoTokenizer
from utils.predict import PairData

# map lm name to huggingface's pre-trained model names
lm_map = {
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "sent-bert": "sentence-transformers/stsb-roberta-base",
}


def get_tokenizer(lm: str):
    if lm in lm_map:
        lm = lm_map[lm]
    return AutoTokenizer.from_pretrained(lm)


def attr2text(attr_list, entity):
    text = []
    for i, e in enumerate(entity):
        if len(str(e)) > 0 and (
            type(e) == str or type(e) == float and np.isnan(e) == False
        ):
            text.append("COL %s VAL %s" % (attr_list[i], str(e)))
    text = " ".join(text)
    return text


def augment(attr_list, feat_list, aug_type="identical"):
    def token_len(feat_list):
        feat_tokens, feat_len = [], []
        for v in feat_list:
            feat_tokens.append(v.split(" "))
            feat_len.append(len(feat_tokens[-1]))
        return feat_tokens, feat_len

    # ['identical', 'swap_token', 'del_token', 'swap_col', 'del_col', 'shuffle_token', 'shuffle_col']
    if aug_type == "del_token":
        span_len = random.randint(1, 2)
        feat_tokens, feat_len = token_len(feat_list)
        try_time = 0
        while True:
            try_time += 1
            if try_time > 5:
                break
            idx = random.randint(0, len(feat_tokens) - 1)
            if feat_len[idx] > span_len:
                pos = random.randint(0, feat_len[idx] - span_len)
                feat_tokens[idx] = (
                    feat_tokens[idx][:pos] + feat_tokens[idx][pos + span_len :]
                )
                break
        feat_list = [" ".join(tokens) for tokens in feat_tokens]
    elif aug_type == "swap_token":
        span_len = random.randint(2, 4)
        feat_tokens, feat_len = token_len(feat_list)
        try_time = 0
        while True:
            try_time += 1
            if try_time > 5:
                break
            idx = random.randint(0, len(feat_tokens) - 1)
            if feat_len[idx] >= span_len:
                pos = random.randint(0, feat_len[idx] - span_len)
                subattr = feat_tokens[idx][pos : pos + span_len]
                np.random.shuffle(subattr)
                feat_tokens[idx] = (
                    feat_tokens[idx][:pos]
                    + subattr
                    + feat_tokens[idx][pos + span_len :]
                )
                break
        feat_list = [" ".join(tokens) for tokens in feat_tokens]
    elif aug_type == "swap_col":
        idx1 = random.randint(0, len(feat_list) - 1)
        idx2 = random.randint(0, len(feat_list) - 1)
        feat_list[idx1], feat_list[idx2] = feat_list[idx2], feat_list[idx1]
        attr_list[idx1], attr_list[idx2] = attr_list[idx2], attr_list[idx1]
    elif aug_type == "del_col":
        idx = random.randint(0, len(feat_list) - 1)
        del feat_list[idx]
        del attr_list[idx]
    elif aug_type == "shuffle_col":
        shuffled_idx = np.array(list(range(len(attr_list))))
        np.random.shuffle(shuffled_idx)
        attr_list = list(np.array(attr_list)[shuffled_idx])
        feat_list = list(np.array(feat_list)[shuffled_idx])
    elif aug_type == "shuffle_token":
        feat_tokens, feat_len = token_len(feat_list)
        idx = random.randint(0, len(feat_tokens) - 1)
        np.random.shuffle(feat_tokens[idx])
        feat_list = [" ".join(tokens) for tokens in feat_tokens]

    return attr_list, feat_list


def gen_augmented_data(attrs, left_entries, right_entries, num=100):
    examples = []
    for idx in np.random.choice(range(len(left_entries)), num):
        data_aug = augment(attrs, left_entries[idx], "shuffle_token")[1]
        examples.append([left_entries[idx], data_aug, 1])
        examples.append(
            [left_entries[idx], right_entries[np.random.choice(len(right_entries))], 0]
        )
    return examples


class SingleEntityDataset(data.Dataset):
    def __init__(self, data, attr_list, lm="roberta", max_len=256, add_token=True):
        self.tokenizer = get_tokenizer(lm)
        self.data = data
        self.attrs = attr_list
        self.max_len = max_len
        self.add_token = add_token
        self.entitytext = self.combine_token_feature_text()

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.data)

    def combine_token_feature_text(self):
        entity_text = []
        for entity in self.data:
            entity = list(entity)  # list of attribute values
            entity = [str(e) for e in entity]
            entity_text.append(attr2text(list(deepcopy(self.attrs)), list(entity)))
        return entity_text

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the entity
            List of int: mask of the entity
        """
        text = self.entitytext[idx]
        x1 = self.tokenizer(text=text, max_length=self.max_len, truncation=True)
        return x1["input_ids"], x1["attention_mask"]

    @staticmethod
    def pad(batch):
        x, x_mask = zip(*batch)
        maxlen = max([len(xi) for xi in x])
        x = [xi + [0] * (maxlen - len(xi)) for xi in x]
        x_mask = [xi + [0] * (maxlen - len(xi)) for xi in x_mask]
        return torch.LongTensor(x), torch.LongTensor(x_mask)


class AugDatasetWithLabel(data.Dataset):
    """
    EM dataset: generate augmented text pair
    Keeping the positive labeled pair
    return (x1, x2)
    """

    def __init__(
        self,
        samples: List[PairData],
        attrs,
        left_entries,
        right_entries,
        lm="sent-bert",
        max_len=128,
        add_token=True,
        concat=False,
        shuffle=False,
        aug_type="random",
    ):
        self.tokenizer = get_tokenizer(lm)
        self.left_entries = left_entries
        self.right_entries = right_entries
        self.attrs = attrs
        self.max_len = max_len
        self.add_token = add_token
        self.concat = concat
        self.samples = samples  # only positive pairs
        self.shuffle = shuffle
        self.aug_type = aug_type
        self.paired_pos()

    def paired_pos(self):
        self.A_pos = defaultdict(list)
        self.B_pos = defaultdict(list)
        for d in self.samples:
            y = int(d.prob > 0.5)
            if y == 1:
                self.A_pos[d.lid].append([d.rid, d.w])
                self.B_pos[d.rid].append([d.lid, d.w])

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.left_entries) + len(self.right_entries)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the original entity
            List of int: token ID's of the augmented entity
        """
        if idx >= len(self.left_entries):
            e2 = idx - len(self.left_entries)
            entity = self.right_entries[e2]
            entity = [str(e) for e in entity]

            cand = self.B_pos.get(e2, [])
            if len(cand) > 0:
                idx = np.random.choice(range(len(cand)))
                pos_entity = self.left_entries[int(cand[idx][0])]
                w = cand[idx][1]
            else:
                pos_entity = None
                w = 1
        else:
            e1 = idx
            entity = self.left_entries[idx]
            entity = [str(e) for e in entity]

            cand = self.A_pos.get(e1, [])
            if len(cand) > 0:
                # cand = sorted(cand, key=lambda x: x[1], reverse=True)
                idx = np.random.choice(range(len(cand)))
                # idx = 0
                pos_entity = self.right_entries[int(cand[idx][0])]
                w = cand[idx][1]
            else:
                pos_entity = None
                w = 1

        if pos_entity is not None:
            pos_attr = list(deepcopy(self.attrs))
        else:
            if len(self.attrs) > 1:
                aug_type = random.choice(["shuffle_token", "shuffle_col", "del_token"])
            else:
                aug_type = random.choice(["del_token", "shuffle_token"])
            pos_attr, pos_entity = augment(
                list(deepcopy(self.attrs)), list(deepcopy(entity)), aug_type
            )

        org_text = attr2text(list(deepcopy(self.attrs)), list(entity))
        x1 = self.tokenizer(text=org_text, max_length=self.max_len, truncation=True)
        pos_text = attr2text(pos_attr, pos_entity)
        x2 = self.tokenizer(text=pos_text, max_length=self.max_len, truncation=True)

        return (x1["input_ids"], x2["input_ids"], w)

    @staticmethod
    def pad(batch):
        x1, x2, w = zip(*batch)
        maxlen1 = max([len(x) for x in x1])
        maxlen2 = max([len(x) for x in x2])
        x1 = [xi + [0] * (maxlen1 - len(xi)) for xi in x1]
        x2 = [xi + [0] * (maxlen2 - len(xi)) for xi in x2]
        return (torch.LongTensor(x1), torch.LongTensor(x2), torch.FloatTensor(w))


class PairDatasetWithLabel(data.Dataset):
    """
    for supervised learning
    EM dataset: load the labeled data
    return (x, y, e1, e2)
    """

    def __init__(
        self,
        samples: List[PairData],
        attrs,
        left_entries,
        right_entries,
        lm="roberta",
        max_len=512,
        concat=False,
    ):
        self.tokenizer = get_tokenizer(lm)
        self.left_entries = left_entries
        self.right_entries = right_entries
        self.attrs = attrs
        self.max_len = max_len
        self.samples = self.get_text_samples(samples)
        self.org_samples = samples
        self.concat = concat

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.samples)

    def get_text_samples(self, samples: List[PairData]):
        samples_text = []
        for d in samples:
            entry1 = self.left_entries[int(d.lid)]
            entry2 = self.right_entries[int(d.rid)]
            text1 = attr2text(self.attrs, entry1)
            text2 = attr2text(self.attrs, entry2)
            y = int(d.prob > 0.5)
            samples_text.append([text1, text2, y, d.lid, d.rid])
        return samples_text

    def __getitem__(self, idx):
        text1, text2, y, e1, e2 = self.samples[idx]
        if self.concat:
            x = self.tokenizer(
                text=text1,
                text_pair=text2,
                max_length=self.max_len,
                add_special_tokens=True,
                truncation=True,
            )
            return x["input_ids"], y, e1, e2
        else:
            x1 = self.tokenizer(text=text1, max_length=self.max_len, truncation=True)
            x2 = self.tokenizer(text=text2, max_length=self.max_len, truncation=True)
            return x1["input_ids"], x2["input_ids"], y, e1, e2

    @staticmethod
    def pad(batch):
        if len(batch[0]) == 4:
            x, y, e1, e2 = zip(*batch)
            maxlen = max([len(x) for x in x])
            x = [xi + [0] * (maxlen - len(xi)) for xi in x]
            return (torch.LongTensor(x), torch.LongTensor(y), e1, e2)
        else:
            x1, x2, y, e1, e2 = zip(*batch)
            maxlen1 = max([len(x) for x in x1])
            maxlen2 = max([len(x) for x in x2])
            x1 = [xi + [0] * (maxlen1 - len(xi)) for xi in x1]
            x2 = [xi + [0] * (maxlen2 - len(xi)) for xi in x2]
            return (
                torch.LongTensor(x1),
                torch.LongTensor(x2),
                torch.LongTensor(y),
                e1,
                e2,
            )


class WeightedPairDatasetWithLabel(PairDatasetWithLabel):
    def get_text_samples(self, samples: List[PairData]):
        samples_text = []
        for d in samples:
            entry1 = self.left_entries[int(d.lid)]
            entry2 = self.right_entries[int(d.rid)]
            text1 = attr2text(self.attrs, entry1)
            text2 = attr2text(self.attrs, entry2)
            y = int(d.prob > 0.5)
            w = d.w
            samples_text.append([text1, text2, y, w, d.lid, d.rid])
        return samples_text

    def __getitem__(self, idx):
        text1, text2, y, w, e1, e2 = self.samples[idx]
        x = self.tokenizer(
            text=text1,
            text_pair=text2,
            max_length=self.max_len,
            add_special_tokens=True,
            truncation=True,
        )
        return x["input_ids"], y, w, e1, e2

    @staticmethod
    def pad(batch):
        x, y, w, e1, e2 = zip(*batch)
        maxlen = max([len(x) for x in x])
        x = [xi + [0] * (maxlen - len(xi)) for xi in x]
        return (torch.LongTensor(x), torch.LongTensor(y), torch.FloatTensor(w), e1, e2)


class TripleDatasetWithLabel(data.Dataset):
    def __init__(
        self,
        samples: List[List[PairData]],
        attrs,
        left_entries,
        right_entries,
        lm="roberta",
        max_len=512,
    ):
        self.tokenizer = get_tokenizer(lm)
        self.left_entries = left_entries
        self.right_entries = right_entries
        self.attrs = attrs
        self.max_len = max_len
        self.samples = self.get_text_samples(samples)
        self.org_samples = samples

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.samples)

    def get_text_samples(self, samples: List[List[PairData]]):
        samples_text = []
        for triple in samples:
            assert len(triple) == 3
            triple_text = []
            for i, d in enumerate(triple):
                if i < 2:
                    entry1 = self.left_entries[d.lid]
                else:
                    entry1 = self.right_entries[d.lid]
                entry2 = self.right_entries[d.rid]
                text1 = attr2text(self.attrs, entry1)
                text2 = attr2text(self.attrs, entry2)
                triple_text.append([text1, text2, d.prob, d.w])
            samples_text.append(triple_text)
        return samples_text

    def __getitem__(self, idx):
        data = []
        for d in self.samples[idx]:
            x = self.tokenizer(
                text=d[0],
                text_pair=d[1],
                max_length=self.max_len,
                add_special_tokens=True,
                truncation=True,
            )
            y = int(d[2] > 0.5)
            # w = max(d[2], 1 - d[2])
            # w = max(1, np.exp((w - 0.8) * 5))
            w = d[-1]
            data.extend([x["input_ids"], y, w])
        return data

    @staticmethod
    def pad(batch):
        x1, y1, w1, x2, y2, w2, x3, y3, w3 = zip(*batch)
        maxlen1 = max([len(x) for x in x1])
        maxlen2 = max([len(x) for x in x2])
        maxlen3 = max([len(x) for x in x3])
        x1 = [xi + [0] * (maxlen1 - len(xi)) for xi in x1]
        x2 = [xi + [0] * (maxlen2 - len(xi)) for xi in x2]
        x3 = [xi + [0] * (maxlen3 - len(xi)) for xi in x3]
        return (
            torch.LongTensor(x1),
            torch.LongTensor(y1),
            torch.FloatTensor(w1),
            torch.LongTensor(x2),
            torch.LongTensor(y2),
            torch.FloatTensor(w2),
            torch.LongTensor(x3),
            torch.LongTensor(y3),
            torch.FloatTensor(w3),
        )


def prepare_bk_train_data(bk_train_data, input_data):
    bk_train = AugDatasetWithLabel(
        bk_train_data,
        input_data.attrs,
        input_data.left_entries,
        input_data.right_entries,
        lm="sent-bert",
        add_token=True,
        concat=False,
        shuffle=False,
        aug_type="random",
    )
    return bk_train


def prepare_bk_valid_data(bk_valid_data, input_data):
    bk_valid = PairDatasetWithLabel(
        bk_valid_data,
        input_data.attrs,
        input_data.left_entries,
        input_data.right_entries,
        concat=False,
    )
    return bk_valid


def prepare_mc_train_data(mc_train_data_org, input_data, sample=False):
    if sample:
        mc_train_data = []
        pos_num, neg_num = 0, 0
        for d in mc_train_data_org:
            if d.prob > 0.5 and d.w < 1:
                pos_num += 1
            elif d.prob <= 0.5 and d.w < 1:
                neg_num += 1
        sample_ratio = pos_num * 3.0 / neg_num
        for d in mc_train_data_org:
            if d.prob > 0.5:
                mc_train_data.append(d)
            elif d.prob <= 0.5:
                if d.w == 1:
                    mc_train_data.append(d)
                elif np.random.rand() < sample_ratio:
                    mc_train_data.append(d)
    else:
        mc_train_data = mc_train_data_org
    mc_train = WeightedPairDatasetWithLabel(
        mc_train_data,
        input_data.attrs,
        input_data.left_entries,
        input_data.right_entries,
        concat=True,
    )
    return mc_train


def prepare_mc_valid_data(mc_valid_data, input_data, cross_table=True):
    mc_valid = PairDatasetWithLabel(
        mc_valid_data,
        input_data.attrs,
        input_data.left_entries if cross_table else input_data.right_entries,
        input_data.right_entries,
        concat=True,
    )
    return mc_valid


def prepare_mc_predict_data(
    input_data, lid_topk=None, lid_bk={}, lids=None, cross_table=True
):
    if lids is None:
        lids = list(range(len(lid_topk)))

    pairs = []
    if cross_table:
        for lid in lids:
            rids = lid_topk[lid]
            for rid in rids:
                if rid in lid_bk.get(lid, []):
                    continue
                pairs.append(PairData(lid, rid, 0.5))
    else:
        for lid in lids:
            rids = lid_bk.get(lid, [])
            for j1 in range(len(rids)):
                for j2 in range(j1 + 1, len(rids)):
                    pairs.append(PairData(rids[j1], rids[j2], 0.5))
    pairs = prepare_mc_valid_data(pairs, input_data, cross_table=cross_table)
    return pairs
