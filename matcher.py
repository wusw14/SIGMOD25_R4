import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AdamW
from utils.model import DittoConcatModel
from utils.utils import weighted_f1_score

from apex import amp


def transform_logits(logits, prob_type="prob"):
    probs = F.softmax(logits, dim=1)[:, 1]
    if prob_type == "prob":
        return probs
    elif prob_type == "log_prob":
        probs = torch.log(probs)
    elif prob_type == "pred":
        probs = (probs > 0.5).float()
    else:
        raise ValueError("Invalid prob_type")
    return probs


class Matcher:
    def __init__(self, lm, model_path, lr=1e-5):
        self.lm = lm
        self.model_path = os.path.join(model_path, "MC_model.pt")
        self.model = DittoConcatModel(lm=lm).cuda()
        self.get_optimizer(lr)

    def get_optimizer(self, lr):
        opt = AdamW(self.model.parameters(), lr=lr)
        self.model, self.opt = amp.initialize(self.model, opt, opt_level="O2")

    def train(
        self,
        train_set,
        batch_size,
        add_trans_loss=False,
        weighted_train=False,
        w_annot=np.exp(1),
    ):
        """
        Train the matcher
        """
        iterator = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=train_set.pad,
        )
        criterion = nn.CrossEntropyLoss(reduction="none")
        self.model.train()
        train_loss, num = 0.0, 0
        for batch in iterator:
            x, y, w, e1, e2 = batch
            self.opt.zero_grad()
            logits = self.model(x)
            ce_loss = criterion(logits, y.cuda())
            # annot_mask = (w >= 1).float()
            # w = annot_mask + (1 - annot_mask) / w_annot
            w = w.float()
            if weighted_train:
                loss = (ce_loss * w.cuda() * (w > 0).cuda()).mean()
            else:
                loss = (ce_loss * (w > 0).cuda()).mean()
            with amp.scale_loss(loss, self.opt) as scaled_loss:
                scaled_loss.backward()
            self.opt.step()
            train_loss += loss.item() * len(w)
            num += len(w)
        return train_loss / num

    def evaluate(
        self, valid_set, add_trans_loss=False, add_weights=False, w_annot=np.exp(1)
    ):
        """
        Evaluate the matcher
        """
        iterator = DataLoader(
            dataset=valid_set, batch_size=128, collate_fn=valid_set.pad
        )
        self.model.eval()
        y_truth, y_pre, weights = [], [], []
        trans_loss_all, num = 0.0, 0
        with torch.no_grad():
            for batch in iterator:
                x, y, w, e1, e2 = batch
                self.opt.zero_grad()
                logits = self.model(x)
                preds = transform_logits(logits, "pred")
                preds = preds.detach().cpu().numpy().tolist()
                # annot_mask = (w >= 1).float()
                # w = annot_mask + (1 - annot_mask) / w_annot
                w = w.float()
                y_truth.extend(y.numpy().tolist())
                y_pre.extend(preds)
                weights.extend(w.numpy().tolist())
                num += len(w)
        if add_weights:
            score, pre, rec = weighted_f1_score(
                y_true=y_truth, y_pred=y_pre, weights=weights
            )
        else:
            score, pre, rec = weighted_f1_score(y_true=y_truth, y_pred=y_pre)
        score2 = f1_score(y_true=y_truth, y_pred=y_pre)
        return score, score2, pre, rec

    def predict(self, test_set):
        """
        Predict the entry pairs
        """
        iterator = DataLoader(dataset=test_set, batch_size=128, collate_fn=test_set.pad)
        self.model.eval()
        prob_dict = {}
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                x, y, e1, e2 = batch
                logits = self.model(x)
                probs = transform_logits(logits, "prob")
                probs = probs.detach().cpu().numpy()
                for i, (e1, e2) in enumerate(zip(e1, e2)):
                    prob_dict[(int(e1), int(e2))] = probs[i]
        return prob_dict

    def save(self, iter_num=0):
        """
        Save the model
        """
        torch.save(self.model.state_dict(), self.model_path)
        torch.save(
            self.model.state_dict(), self.model_path.replace(".pt", f"_{iter_num}.pt")
        )

    def load(self, iter_num=None):
        """
        Load the model
        """
        if iter_num is not None:
            self.model.load_state_dict(
                torch.load(self.model_path.replace(".pt", f"_{iter_num}.pt"))
            )
        elif os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            print("No model found")
            raise FileNotFoundError
