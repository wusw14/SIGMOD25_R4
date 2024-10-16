import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy import ndarray


class TopkSimData:
    def __init__(
        self,
        lid_topk: ndarray,
        lid_sim: ndarray,
        rid_topk: ndarray,
        rid_sim: ndarray,
        sim_score: ndarray,
    ):
        self.lid_topk = lid_topk
        self.lid_sim = lid_sim
        self.rid_topk = rid_topk
        self.rid_sim = rid_sim
        self.sim_score = sim_score
        self.gap = 0.03

    def gen_pseudo(self, target_lids: Optional[List[int]], hard=False) -> dict:
        entrypair_with_label = {}
        if target_lids is None:
            target_lids = range(len(self.lid_topk))
        for lid in target_lids:
            rids = self.lid_topk[lid]
            sims = self.lid_sim[lid]
            rid_top1 = rids[0]
            sim_top1 = sims[0]
            if self.rid_topk[rid_top1][0] == lid and sim_top1 - sims[-1] >= self.gap:
                entrypair_with_label[lid, rid_top1] = 1
            cnt = 0
            for rid, sim in zip(rids[1:], sims[1:]):
                if sim_top1 - sim >= self.gap:
                    entrypair_with_label[lid, rid] = 0
                    cnt += 1
                    if hard and cnt == 2:
                        break
        return entrypair_with_label


class PairData:
    def __init__(self, lid, rid, prob, w=None):
        # ? HH: why int() is needed here? when the lid or rid is not a int?
        self.lid = int(lid)
        self.rid = int(rid)
        self.prob = prob
        if w is None:
            self.w = max(prob, 1 - prob)
        else:
            self.w = w


class Prediction:
    def __init__(self, mc_pred=None, bk_pred=None, llm_pred=None, iter_num=0):
        self.mc_pred = mc_pred
        self.bk_pred = bk_pred
        self.llm_pred = llm_pred if llm_pred is not None else []
        self.iter_num = iter_num
        self.mc_trans_vlt = 0
        self.llm_trans_vlt = 0
        self.trans_vlt = 0
        self.trans_vlt_rvs = 0
        self.trans_vlt_wt = 0
        self.corrected_pred = None
        self.conf = None
        self.label = None

    def flatten_probs_into_list(self):
        probs = []
        if self.mc_pred is not None:
            probs.append(self.mc_pred)
        if self.bk_pred is not None:
            probs.append(self.bk_pred)
        if len(self.llm_pred) > 0:
            probs.extend(self.llm_pred)
        return probs

    def agg_pred(self, only_llm=False) -> int:
        if only_llm:
            probs = self.llm_pred
        else:
            probs = self.flatten_probs_into_list()
        if len(probs) == 0:
            return -1
        preds = [1 if prob > 0.5 else 0 for prob in probs]
        # max vote
        if len(preds) == 2:
            return int(np.mean(preds) > 0.5)
        return max(set(preds), key=preds.count)

    def agg_prob(self, only_llm=False) -> float:
        if only_llm:
            probs = self.llm_pred
        else:
            probs = self.flatten_probs_into_list()
        if len(probs) == 0:
            return -1
        if (np.mean(probs) > 0.5) == self.agg_pred(only_llm):
            return np.mean(probs)
        else:
            return 0.5

    def consistent(self, probs=None, only_llm=False) -> bool:
        if probs is None:
            if only_llm:
                probs = self.llm_pred
            else:
                probs = self.flatten_probs_into_list()
        cond2 = True
        if len(probs) < 1:
            return False
        elif len(self.llm_pred) == 0:
            if len(probs) == 2:
                p = np.random.rand() / 2 + 0.5
                cond2 = p < min(0.95, max(self.mc_pred, 1 - self.mc_pred))
            else:
                cond2 = False
        elif len(self.llm_pred) == 1:
            if self.llm_pred[0] < 0.1 or self.llm_pred[0] > 0.9:
                return True
        if len(probs) < 2:
            return False
        preds = [1 if prob > 0.5 else 0 for prob in probs]
        cond1 = len(set(preds)) == 1
        return cond1 and cond2

    def query_llm(self, only_llm=False, mc_thr_dict=None) -> bool:
        if only_llm:
            if len(self.llm_pred) < 3 and not self.consistent(self.llm_pred):
                return True
        else:
            if (
                self.mc_pred is not None
                and mc_thr_dict is not None
                and (
                    self.mc_pred > mc_thr_dict["pos"]
                    or self.mc_pred < mc_thr_dict["neg"]
                )
            ):
                return False
            if (
                len(self.llm_pred) < 3
                and not self.consistent()
                and not self.consistent(self.llm_pred)
            ):
                return True
        return False

    def get_prob(self, predictor):
        if self.label is not None:
            return self.label
        if predictor == "corrected":
            if self.corrected_pred is None:
                predictor = "llm"
            else:
                return self.corrected_pred
        if predictor == "llm":
            if len(self.llm_pred) == 0:
                return self.mc_pred
            else:
                return np.mean(self.llm_pred)
        if predictor == "bk":
            return self.bk_pred
        if predictor == "mc":
            return self.mc_pred


class PredCollection:
    def __init__(self, only_llm=False):
        self.pred_dict: Dict[Tuple[int, int], Prediction] = {}
        self.pred_dict_rtb: Dict[Tuple[int, int], Prediction] = {}
        self.only_llm = only_llm

    def add_dict(
        self, predicted_data: dict, predictor="bk", cross_table=True, iter_num=0
    ):
        for (lid, rid), prob in predicted_data.items():
            self.add(lid, rid, prob, predictor, cross_table, iter_num)
            if cross_table == False:
                self.add(rid, lid, prob, predictor, cross_table, iter_num)

    def add(self, lid, rid, prob=None, predictor=None, cross_table=True, iter_num=0):
        pred_dict = self.pred_dict if cross_table else self.pred_dict_rtb
        if (lid, rid) not in pred_dict:
            pred_dict[(lid, rid)] = Prediction()
        if predictor == "bk":
            pred_dict[(lid, rid)].bk_pred = prob
        elif predictor == "mc":
            pred_dict[(lid, rid)].mc_pred = prob
        elif predictor == "llm":
            pred_dict[(lid, rid)].llm_pred.append(prob)
        pred_dict[(lid, rid)].iter_num = iter_num
        # HH: what about predictor is None? is it allowed?

    def update_trans_vlt(self, lid_bk, predictor="mc", reverse=False):
        trans_cnt = 0
        for lid, rids in lid_bk.items():
            for rid1 in rids:
                trans_score_all, trans_wt_all = [], []
                for rid2 in rids:
                    if rid1 == rid2:
                        continue
                    pred1 = self.pred_dict[(lid, rid1)]
                    pred2 = self.pred_dict[(lid, rid2)]
                    pred3 = self.pred_dict_rtb[(rid1, rid2)]
                    trans_score, trans_score_wt = trans_check(
                        pred1, pred2, pred3, reverse, predictor
                    )
                    trans_score_all.append(trans_score)
                    trans_wt_all.append(trans_score_wt)
                trans_score_all = (
                    np.mean(trans_score_all) if len(trans_score_all) > 0 else 0
                )
                trans_wt_all = np.mean(trans_wt_all) if len(trans_wt_all) > 0 else 0
                if predictor == "mc":
                    self.pred_dict[(lid, rid1)].mc_trans_vlt = trans_score_all
                elif predictor == "llm":
                    self.pred_dict[(lid, rid1)].llm_trans_vlt = trans_score_all
                elif predictor == "corrected":
                    if reverse:
                        self.pred_dict[(lid, rid1)].trans_vlt_rvs = trans_score_all
                    else:
                        self.pred_dict[(lid, rid1)].trans_vlt = trans_score_all
                        self.pred_dict[(lid, rid1)].trans_vlt_wt = trans_wt_all
                if trans_score_all > 0:
                    trans_cnt += 1
        if len(lid_bk) > 10:
            print(f"transitivity violation number = {trans_cnt}")

    def update_corrected_pred(self, lid_bk, gt_dict={}):
        revised_num = 0
        tp, fp, tn, fn = 0, 0, 0, 0
        trans_vlt_num = 0
        inconsist_num = 0
        for lid, rids in lid_bk.items():
            rids_to_check = []
            for rid in rids:
                pred = self.pred_dict[(lid, rid)]
                if pred.label is not None:
                    pred.corrected_pred = pred.label
                elif len(pred.llm_pred) > 0:
                    if (np.mean(pred.llm_pred) > 0.5) != (pred.mc_pred > 0.5):
                        rids_to_check.append(rid)
                self.pred_dict[(lid, rid)] = pred
            inconsist_num += len(rids_to_check)
            while len(rids_to_check) > 0:
                self.update_trans_vlt({lid: rids}, predictor="corrected")
                self.update_trans_vlt({lid: rids}, predictor="corrected", reverse=True)
                max_trans_vlt = -100
                for rid in rids_to_check:
                    trans_vlt = self.pred_dict[(lid, rid)].trans_vlt
                    trans_vlt_rvs = self.pred_dict[(lid, rid)].trans_vlt_rvs
                    if trans_vlt - trans_vlt_rvs > max_trans_vlt:
                        rid_to_correct = rid
                        max_trans_vlt = trans_vlt - trans_vlt_rvs
                prob = self.pred_dict[(lid, rid_to_correct)].get_prob(
                    predictor="corrected"
                )
                mc_pred = self.pred_dict[(lid, rid_to_correct)].mc_pred
                trans_vlt_wt = self.pred_dict[(lid, rid_to_correct)].trans_vlt_wt
                trans_vlt = self.pred_dict[(lid, rid_to_correct)].trans_vlt
                if (max_trans_vlt > 0 and trans_vlt_wt > trans_vlt / 2) or (
                    max_trans_vlt > 0.3
                ):
                    self.pred_dict[(lid, rid_to_correct)].corrected_pred = 1 - prob
                    conf = 1.5 - max(prob, 1 - prob)
                    self.pred_dict[(lid, rid_to_correct)].conf = conf
                    revised_num += 1
                    label = gt_dict.get((lid, rid_to_correct), 0)
                    if label == 1:
                        if 1 - prob > 0.5:
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if 1 - prob > 0.5:
                            fp += 1
                        else:
                            tn += 1
                    if label != int(prob > 0.5):
                        ans = "[+1]"
                    else:
                        ans = "[-1]"
                    print(
                        f"{ans}: lid={lid},rid={rid_to_correct},label={label},"
                        f"corrected={1 - prob:.4f},mc={mc_pred:.4f},trans_diff={max_trans_vlt:.4f}"
                    )
                rids_to_check.remove(rid_to_correct)
            self.update_trans_vlt({lid: rids}, predictor="corrected")
            self.update_trans_vlt({lid: rids}, predictor="corrected", reverse=True)
            for rid in rids:
                if self.pred_dict[(lid, rid)].trans_vlt > 0:
                    trans_vlt_num += 1
        print(f"transitivity violation = {trans_vlt_num}")
        print(f"Revise {revised_num} of inconsistent predictions")
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"Inconsistent number = {inconsist_num}")

    def self_correct(self, lid, rid, rids, mc_thr_dict, gt_dict):
        l1 = gt_dict.get((lid, rid), 0)
        for rid1 in rids:
            if rid1 == rid:
                continue
            pred3 = self.pred_dict_rtb.get((rid, rid1), None)
            if pred3 is None:
                continue
            if pred3.mc_pred > mc_thr_dict["pos"]:
                l3 = 1
            elif pred3.mc_pred < mc_thr_dict["neg"]:
                l3 = 0
            else:
                continue
            pred2 = self.pred_dict[(lid, rid1)]
            if pred2.label is not None:
                continue
            prob2 = pred2.get_prob("mc")
            if prob2 > mc_thr_dict["pos"] or prob2 < mc_thr_dict["neg"]:
                continue
            l2 = int(prob2 > 0.5)
            if l1 + l2 + l3 == 2:
                label = 1 - l2
            elif l1 + 1 - l2 + l3 == 2:
                label = l2
            else:
                continue
            self.pred_dict[(lid, rid1)].corrected_pred = label
            llm_prob = np.mean(pred2.llm_pred)
            if label == gt_dict.get((lid, rid1), 0):
                print(f"[+1]: {int(lid)},{rid1},{label},mc={prob2:.4f},llm={llm_prob}")
            else:
                print(f"[-1]: {int(lid)},{rid1},{label},mc={prob2:.4f},llm={llm_prob}")

    def retrieve_data_for_bk(
        self, lid_bk, lids, annotated_data=None, weight_dict={}, predictor="corrected"
    ):
        data_list = []
        for (lid, rid), pred in self.pred_dict.items():
            if (
                lid not in lids
                or (lid, rid) in annotated_data
                or rid not in lid_bk.get(lid, [])
            ):
                continue
            if pred is not None:
                prob = pred.get_prob(predictor=predictor)
                data_list.append(PairData(lid, rid, prob, 1))
        for (lid, rid), label in annotated_data.items():
            if lid in lids:
                data_list.append(PairData(lid, rid, label, 2))
        return data_list

    def retrieve_data_for_mc(
        self, lid_bk, lids, annotated_data=None, weight_dict={}, predictor="corrected"
    ):
        data_list = []
        for (lid, rid), pred in self.pred_dict.items():
            if (
                lid not in lids
                or (lid, rid) in annotated_data
                or rid not in lid_bk.get(lid, [])
            ):
                continue
            if pred is not None:
                prob = pred.get_prob(predictor=predictor)
                w = cal_weight(pred)
                data_list.append(PairData(lid, rid, prob, w))
        for (lid, rid), label in annotated_data.items():
            if lid in lids:
                data_list.append(PairData(lid, rid, label, 2))
        return data_list

    def retrieve_icl_examples(self):
        examples = []
        for (lid, rid), pred in self.pred_dict.items():
            if pred.agg_pred() == 1 and pred.consistent() and pred.agg_prob() > 0.9:
                examples.append((lid, rid, 1))
            elif pred.agg_pred() == 0 and pred.consistent() and pred.agg_prob() < 0.1:
                examples.append((lid, rid, 0))
        print(f"Examples updated!!! {len(examples)}")
        return examples

    def save(self, llm, dataset, version, gt_dict=None, iter_num=0):
        pred_path = os.path.join("results", llm, dataset, str(version))
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        pred_file = os.path.join(pred_path, f"pred_dict{iter_num}.txt")
        output_pred_collection_to_file(pred_file, self.pred_dict, gt_dict)

        # output pred_dictB to the file
        pred_file = os.path.join(pred_path, f"pred_dictB{iter_num}.txt")
        output_pred_collection_to_file(pred_file, self.pred_dict_rtb)


def output_pred_collection_to_file(pred_file, pred_dict, gt_dict=None):
    with open(pred_file, "w") as f:
        for (e1, e2), pred in pred_dict.items():
            bk = pred.bk_pred
            mc = pred.mc_pred
            llm = pred.llm_pred
            iter_num = pred.iter_num
            if gt_dict is not None:
                label = gt_dict.get((e1, e2), 0)
            else:
                label = None
            f.write(
                f"{e1},{e2},{label},iter={iter_num}\n"
                f"BK: {bk}\nMC: {mc}\nLLM: {llm}\n"
                f"Corrected: {pred.corrected_pred}\n"
                f"mc_trans_vlt: {pred.mc_trans_vlt}\n"
                f"trans_vlt: {pred.trans_vlt}\n"
                f"trans_vlt_rvs: {pred.trans_vlt_rvs}\n"
                f"trans_vlt_wt: {pred.trans_vlt_wt}\n\n"
            )


def cal_weight(pred: Prediction):
    if pred.bk_pred is not None and pred.mc_pred is not None and len(pred.llm_pred) > 0:
        if pred.bk_pred == int(pred.mc_pred > 0.5) and pred.bk_pred == int(
            np.mean(pred.llm_pred) > 0.5
        ):
            return 1
    # if len(pred.llm_pred) == 0:
    #     return 0.5
    prob = pred.get_prob(predictor="corrected")
    if pred.conf is not None:
        conf = pred.conf
    else:
        conf = max(prob, 1 - prob)
    trans_vlt_wt = pred.trans_vlt_wt
    w1 = 2 * conf - 1
    w2 = 1 - 5 * trans_vlt_wt
    return max(0, w1 * w2)


def cal_score(p1, p2, p3):
    conf1 = max(p1, 1 - p1)
    conf2 = max(p2, 1 - p2)
    conf3 = max(p3, 1 - p3)
    return (1 - conf1) / max(((1 - conf2) + (1 - conf1) + (1 - conf3)), 1e-6)


def trans_check(pred1, pred2, pred3, reverse=False, predictor="mc"):
    if pred1 is None or pred2 is None or pred3 is None:
        return 0, 0
    prob1 = pred1.get_prob(predictor)
    prob2 = pred2.get_prob(predictor)
    prob3 = pred3.get_prob(predictor)
    edge1, edge2, edge3 = int(prob1 > 0.5), int(prob2 > 0.5), int(prob3 > 0.5)
    if reverse:
        edge1 = 1 - edge1
    if edge1 + edge2 + edge3 != 2:
        return 0, 0

    if predictor != "mc" and len(pred3.llm_pred) == 0:
        prob1_mc = pred1.mc_pred
        prob2_mc = pred2.mc_pred
        prob3_mc = pred3.mc_pred
        score3 = cal_score(prob3_mc, prob1_mc, prob2_mc)
    else:
        score3 = cal_score(prob3, prob1, prob2)
    score = cal_score(prob1, prob2, edge3)
    score = score * (1 - score3)
    return int(score > 0), score
