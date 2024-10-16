import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, AutoModel

from apex import amp

lm_mp = {
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "sent-bert": "sentence-transformers/stsb-roberta-base",
}


class DittoConcatModel(nn.Module):
    """A base model for EM."""

    def __init__(self, device="cuda", lm="roberta"):
        super().__init__()

        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
            print(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)
            print(lm)

        self.device = device

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device)  # (batch_size, seq_len)
        h1 = self.bert(x1)[0][:, 0, :]  # [B, K]
        pred = self.fc(h1)

        # return pred
        return pred  # [B, 2]


class CLSepModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device="cuda", lm="roberta", pooling="cls"):
        super().__init__()

        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
            print(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)
            print(lm)

        self.device = device
        self.pooling = pooling

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x1):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device)  # (batch_size, seq_len)
        h1 = self.bert(x1)[0]  # [B, L, K]
        emb1 = self.fc(h1[:, 0, :])

        emb1 = F.normalize(emb1)

        return emb1

    def get_emb(self, x):
        x = x.to(self.device)  # (batch_size, seq_len)
        # x_mask = x_mask.to(self.device)
        h = self.bert(x)[0]
        emb = h[:, 0, :]
        # emb = self.fc(emb)
        emb = F.normalize(emb)
        return emb
