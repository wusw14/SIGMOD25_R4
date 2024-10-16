import os

import numpy as np
import torch
from numpy import ndarray
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AdamW
from utils.dataset import SingleEntityDataset
from utils.io import InputData
from utils.model import CLSepModel
from utils.predict import TopkSimData
from sklearn.metrics import ndcg_score

from apex import amp


def l2_norm(embeddings):
    """
    L2 normalization
    """
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


class Blocker:
    def __init__(self, lm, model_path, lr):
        self.lm = lm
        self.model_path = os.path.join(model_path, "BK_model.pt")
        self.model = CLSepModel(lm=lm).cuda()
        self.get_optimizer(lr)

    def get_optimizer(self, lr):
        opt = AdamW(self.model.parameters(), lr=lr)
        self.model, self.opt = amp.initialize(self.model, opt, opt_level="O2")

    def get_init_topk(self, input_data: InputData, topk=5) -> TopkSimData:
        """
        Get the initial topk blocked examples by sent-bert
        """
        EmbModel = SentenceTransformer("stsb-roberta-base")
        left_entry_text = input_data.left_entry_dataset.entitytext
        right_entry_text = input_data.right_entry_dataset.entitytext
        self.embeddingA = EmbModel.encode(left_entry_text, batch_size=512)
        self.embeddingB = EmbModel.encode(right_entry_text, batch_size=512)
        self.embeddingA = l2_norm(self.embeddingA)
        self.embeddingB = l2_norm(self.embeddingB)
        top_sim_data = self.similarity_based_pairing(topk)
        return top_sim_data

    def get_embedding(self, dataset: SingleEntityDataset) -> ndarray:
        """
        Get the embeddings of the input data
        """
        data_iter = DataLoader(
            dataset=dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.pad,
        )
        self.model.eval()
        emb = []
        device = self.model.device
        with torch.no_grad():
            for batch in data_iter:
                x1, x1_mask = batch
                x1, x1_mask = x1.to(device), x1_mask.to(device)
                emb1 = self.model.get_emb(x1)
                emb.extend(emb1.cpu().numpy())
        embedding = np.array(emb)
        embedding = l2_norm(embedding)
        return embedding

    def get_topk(self, input_data: InputData, topk=5, cross_table=True) -> TopkSimData:
        """
        Get the topk blocked examples by the blocker
        """
        self.embeddingA = self.get_embedding(input_data.left_entry_dataset)
        self.embeddingB = self.get_embedding(input_data.right_entry_dataset)
        top_sim_data = self.similarity_based_pairing(topk, cross_table)
        return top_sim_data

    def similarity_based_pairing(self, topk, cross_table=True):
        embeddingA = torch.tensor(self.embeddingA).cuda()
        embeddingB = torch.tensor(self.embeddingB).cuda()
        sim_score = util.pytorch_cos_sim(embeddingA, embeddingB)
        if not cross_table:
            # exclude the diagonal
            sim_score = sim_score - torch.eye(sim_score.size(0)).cuda() * 10
        # lid_topk [sizeA, hp.K]
        lid_sim, lid_topk = torch.topk(sim_score, k=topk, dim=1)
        # rid_topk [sizeB, hp.K]
        rid_sim, rid_topk = torch.topk(sim_score.T, k=topk, dim=1)
        lid_sim = lid_sim.cpu().numpy()
        lid_topk = lid_topk.cpu().numpy()
        rid_sim = rid_sim.cpu().numpy()
        rid_topk = rid_topk.cpu().numpy()
        sim_score = sim_score.cpu().numpy()
        top_sim_data = TopkSimData(lid_topk, lid_sim, rid_topk, rid_sim, sim_score)
        return top_sim_data

    def gen_pseudo(self, top_sim_data: TopkSimData, lids=None, hard=False) -> dict:
        """
        Generate pseudo labels by rules
        """
        return top_sim_data.gen_pseudo(target_lids=lids, hard=hard)

    def gen_pseudo_rids(self, lid_topk, lids=None) -> dict:
        """
        Generate pseudo labels on rids
        """
        if lids is None:
            lids = list(range(len(lid_topk)))
        pseudo_rids = {}
        for i, lid in enumerate(lids):
            rids = lid_topk[lid]
            embs = self.embeddingB[rids]
            sim_scores = np.dot(embs, embs.T)  # [k, k]
            sim_scores = sim_scores - np.eye(sim_scores.shape[0]) * 10
            max_sim = np.max(sim_scores)
            for j1 in range(len(rids)):
                for j2 in range(j1 + 1, len(rids)):
                    pseudo_rids[rids[j1], rids[j2]] = 0
                    # ? HH: why always =0?
                    # if sim_scores[j1, j2] < max_sim:
                    #     pseudo_rids[rids[j1], rids[j2]] = 0
                    # else:
                    #     pseudo_rids[rids[j1], rids[j2]] = 1
        return pseudo_rids

    def train(self, train_set, batch_size, weighted_train=False):
        """
        Train the blocker
        """
        f = lambda x: torch.exp(x / 0.05)
        iterator = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=train_set.pad,
        )
        self.model.train()
        total_loss, total_num = 0.0, 0.0
        w_list = []
        for batch in iterator:
            x1, x2, w = batch
            w_list.extend(w.detach().numpy())
            emb1 = self.model(x1)
            emb2 = self.model(x2)
            scores_pos = torch.mm(emb1, emb2.t())
            scores_pos = f(scores_pos)
            w = w.cuda()
            if weighted_train:
                loss = (
                    (-torch.log(scores_pos.diag() / (scores_pos.sum(1)))) * w
                ).mean()
            else:
                loss = -torch.log(scores_pos.diag() / (scores_pos.sum(1))).mean()
            # if args.fp16:
            with amp.scale_loss(loss, self.opt) as scaled_loss:
                scaled_loss.backward()
            self.opt.step()
            total_loss += loss.item() * len(w)
            total_num += len(w)
            del loss
        return total_loss / total_num

    def evaluate(self, valid_set, input_data: InputData):
        """
        Evaluate the blocker
        """
        data_dict = {(d.lid, d.rid): 1 if d.prob > 0.5 else 0 for d in valid_set}
        top_sim_data = self.get_topk(input_data, topk=20)
        ndcg_list = []
        pos_score, neg_score = [], []
        for lid in input_data.valid_indices:
            rids = top_sim_data.lid_topk[lid]
            true_relevance, scores = [], []
            for rid in rids:
                true_relevance.append(data_dict.get((lid, rid), 0))
                scores.append(top_sim_data.sim_score[lid, rid])
                if true_relevance[-1] > 0.5:
                    pos_score.append(scores[-1])
                else:
                    neg_score.append(scores[-1])
            if np.sum(true_relevance) == 0:
                continue
            ndcg = ndcg_score([true_relevance], [scores])
            ndcg_list.append(ndcg)
        bk_rec = len(pos_score) / np.sum(list(data_dict.values()))
        score = np.mean(ndcg_list) * bk_rec
        return score, np.mean(pos_score) - np.mean(neg_score)

        # iterator = DataLoader(
        #     dataset=valid_set, batch_size=128, collate_fn=valid_set.pad
        # )
        # self.model.eval()
        # scores, y_truth = [], []
        # with torch.no_grad():
        #     for i, batch in enumerate(iterator):
        #         x1, x2, y, lids, rids = batch
        #         x1_emb = self.model.get_emb(x1).detach().cpu().numpy()
        #         x2_emb = self.model.get_emb(x2).detach().cpu().numpy()
        #         batch_scores = np.sum(x1_emb * x2_emb, 1)
        #         scores.extend(list(1 / (1 + np.exp(-batch_scores))))
        #         y_truth.extend(y.detach().cpu().numpy())
        # y_truth = np.array(y_truth)
        # scores = np.array(scores)
        # pos_avg = np.mean(scores[y_truth == 1])
        # neg_avg = np.mean(scores[y_truth == 0])
        # pos_neg_gap = pos_avg - neg_avg
        # bestF1 = 0.0
        # for th in np.arange(0, 1, 0.05):
        #     pred_list = [1 if s > th else 0 for s in scores]
        #     f1 = f1_score(y_truth, pred_list)
        #     if f1 > bestF1:
        #         bestF1 = f1
        # return bestF1, pos_neg_gap

    def predict(self, input_data):
        """
        Predict the blocks
        """
        pass

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
