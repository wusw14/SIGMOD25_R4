import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.predict import PredCollection
from typing import Tuple
from copy import deepcopy


def load_llm_preds(filename: str, cross_table=True):
    pred_dict = {}
    if not os.path.exists(filename):
        # print(f"{filename} not found")
        return pred_dict
    f = open(filename, "r")
    while True:
        line = f.readline()
        if len(line) <= 0:
            break
        while True:
            if "iter=" in line:
                lid, rid = line.strip().split(",")[:2]
            elif "LLM:" in line:
                llm_pred = eval(line[len("LLM: ") :])
                pred_dict[(int(lid), int(rid))] = llm_pred
            line = f.readline()
            if len(line) <= 0 or len(line.strip()) == 0:
                break
        if not cross_table:
            pred_dict[(int(rid), int(lid))] = llm_pred
    # print(f"Loaded {len(pred_dict)} predictions from {filename}")
    return pred_dict


def load_past_predictions(llm_name: str, dataset: str, run_id: int = 1, iter_num=1):
    llm_pred_past = {}
    path_list = ["../wt2/results", "wt3_results", "../KD_v2/results"]
    for i in range(run_id + 1):
        for path in path_list:
            filename = f"{path}/{llm_name}/{dataset}/{i}/pred_dict{iter_num}.txt"
            pred = load_llm_preds(filename)
            for k, v in pred.items():
                llm_pred_past[k] = list(set(llm_pred_past.get(k, []) + v))

    llm_pred_past = {k: v[:3] for k, v in llm_pred_past.items()}
    llm_pred_rids_past = {}
    for i in range(run_id + 1):
        for path in path_list:
            filename = f"{path}/{llm_name}/{dataset}/{i}/pred_dictB{iter_num}.txt"
            pred = load_llm_preds(filename)
            for k, v in pred.items():
                llm_pred_rids_past[k] = list(set(llm_pred_rids_past.get(k, []) + v))
    llm_pred_rids_past = {k: v[:3] for k, v in llm_pred_rids_past.items()}
    print(f"Cross_table={len(llm_pred_past)}, Rtable={len(llm_pred_rids_past)}")
    return llm_pred_past, llm_pred_rids_past


def eval_pseudo(gt_dict, pred_dict, predictor: str, indices=None):
    y_pred, y_truth = [], []
    for (lid, rid), pred in pred_dict.items():
        if indices is not None and lid not in indices:
            continue
        y_pred.append(int(pred > 0.5))
        y_truth.append(gt_dict.get((lid, rid), 0))
    pre = precision_score(y_truth, y_pred)
    rec = recall_score(y_truth, y_pred)
    f1 = f1_score(y_truth, y_pred)
    print(f"({len(y_pred)}) {predictor} pre/rec/f1: {pre:.4f}/{rec:.4f}/{f1:.4f}")
    tp = np.sum([int(y_truth[i] == 1 and y_pred[i] == 1) for i in range(len(y_pred))])
    fp = np.sum([int(y_truth[i] == 0 and y_pred[i] == 1) for i in range(len(y_pred))])
    fn = np.sum([int(y_truth[i] == 1 and y_pred[i] == 0) for i in range(len(y_pred))])
    print(f"tp: {int(tp)}, fp: {int(fp)}, fn: {int(fn)}")


def violation(edge1, edge2, edge3):
    if edge1 is None or edge2 is None or edge3 is None:
        return 0
    if int(edge1 > 0.5) + int(edge2 > 0.5) + int(edge3 > 0.5) != 2:
        return 0
    transform = lambda x: np.exp((max(x, 1 - x) - 0.5) * 5)
    conf1 = transform(edge1)
    conf2 = transform(edge2)
    conf3 = transform(edge3)
    return 0.5 * (1 - conf1 / (conf1 + conf2 + conf3))


def check_violation(
    pred_collection, lid, rid1, rid2, only_llm=False, annotated_data={}
):
    pred1 = pred_collection.pred_dict.get((lid, rid1), None)
    pred2 = pred_collection.pred_dict.get((lid, rid2), None)
    if pred1 is None or pred2 is None:
        return 0
    if len(pred1.llm_pred) > 0 and (
        np.mean(pred1.llm_pred) > 0.9 or np.mean(pred1.llm_pred) < 0.1
    ):
        return 0
    edge1 = pred1.agg_prob(only_llm) if len(pred1.llm_pred) > 0 else pred1.mc_pred
    edge2 = pred2.agg_prob(only_llm) if len(pred2.llm_pred) > 0 else pred2.mc_pred
    edge1 = annotated_data.get((lid, rid1), edge1)
    edge2 = annotated_data.get((lid, rid2), edge2)
    pred = pred_collection.pred_dict_rtb.get((rid1, rid2), None)
    if pred is None:
        pred = pred_collection.pred_dict_rtb.get((rid2, rid1), None)
    if pred is not None:
        edge3 = pred.agg_prob(only_llm) if len(pred.llm_pred) > 0 else pred.mc_pred
    else:
        edge3 = 0
    if edge1 is None or edge2 is None or edge3 is None:
        return 0
    violation_org = violation(edge1, edge2, edge3)
    violation_reverse = violation(1 - edge1, edge2, edge3)
    return violation_org - violation_reverse
    # return violation_org


def cal_acc(label_list, pred_list):
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    for l, y in zip(label_list, pred_list):
        if l == 1:
            if y == 1:
                tp += 1
            else:
                fn += 1
        else:
            if y == 1:
                fp += 1
            else:
                tn += 1
    pos_acc = tp / (tp + fp) if tp + fp > 0 else 0
    neg_acc = tn / (tn + fn) if tn + fn > 0 else 0
    return pos_acc, neg_acc


def update(trans_violation, pred_collection, lid, rids, only_llm, annotated_data):
    norm_factor = 5
    for j1 in range(len(rids)):
        trans_violation[(lid, rids[j1])] = 0
        for j2 in range(len(rids)):
            if j1 == j2:
                continue
            rid1, rid2 = rids[j1], rids[j2]
            violation = check_violation(
                pred_collection,
                lid,
                rid1,
                rid2,
                only_llm=only_llm,
                annotated_data=annotated_data,
            )
            trans_violation[(lid, rid1)] += violation  # / norm_factor
    return trans_violation


def get_annotation(
    input_data,
    lid_bk,
    pred_collection: PredCollection,
    annotated_data={},
    budget=100,
    mc_thr_dict={},
    iter_num=1,
) -> Tuple[dict, PredCollection]:

    pd_dict = {
        "lid": [],
        "rid": [],
        "prob": [],
        "conf": [],
        "mc_prob": [],
        "trans_vlt": [],
        "trans_diff": [],
    }
    # TODO: assign confidence to the llm's predictions and matcher's predictions
    for lid in list(input_data.train_indices) + list(input_data.valid_indices):
        for rid in lid_bk[lid]:
            if (lid, rid) in annotated_data:
                continue
            pred = pred_collection.pred_dict.get((lid, rid), None)
            if (
                pred is None
                or (len(pred.llm_pred) == 0)
                or (
                    pred.mc_pred is not None
                    and (
                        pred.mc_pred < mc_thr_dict["neg"]
                        or pred.mc_pred > mc_thr_dict["pos"]
                    )
                )
            ):
                continue
            prob = pred.get_prob(predictor="corrected")
            llm_pred = np.mean(pred.llm_pred)
            mc_pred = pred.mc_pred if iter_num > 1 else 0.5
            if pred.conf is not None:
                conf = pred.conf
            else:
                conf = max(llm_pred, 1 - llm_pred)
            conf = (conf + max(mc_pred, 1 - mc_pred)) / 2
            pd_dict["lid"].append(lid)
            pd_dict["rid"].append(rid)
            pd_dict["prob"].append(prob)
            pd_dict["conf"].append(conf)
            pd_dict["mc_prob"].append(mc_pred)
            pd_dict["trans_vlt"].append(pred.trans_vlt_wt)
            pd_dict["trans_diff"].append(pred.trans_vlt - pred.trans_vlt_rvs)
    df = pd.DataFrame(pd_dict)
    if iter_num == 1:
        df["mc_prob"] = df["prob"]
    df["trans_neg"] = -df["trans_vlt"]
    df["score"] = (2 * df["conf"] - 1) * (1 - 5 * df["trans_vlt"])
    attr_list = [
        "lid",
        "rid",
        "prob",
        "conf",
        "trans_diff",
        "trans_neg",
        "mc_prob",
        "score",
    ]
    print(
        f"[Trans_Violation]: {len(df[df['trans_diff'] > 0])}/{len(df[df['trans_neg'] < 0])}/{len(df)}"
    )
    # filter out confident predictions
    df_pos = df[df["prob"] > 0.5]
    df_pos = df_pos.sort_values("conf", ascending=True)
    df_pos = df_pos.iloc[: max(100, len(df_pos) // 2)]
    df_neg = df[df["prob"] <= 0.5]
    df_neg = df_neg.sort_values("conf", ascending=True)
    df_neg = df_neg.iloc[: max(100, len(df_neg) // 2)]
    df = pd.concat([df_pos, df_neg])
    index_cond = [input_data.train_indices, input_data.valid_indices]
    budget_limit = [int(budget * 0.75), budget - int(budget * 0.75)]
    for i, (indices, budget) in enumerate(zip(index_cond, budget_limit)):
        label_list = []
        corrected_all = 0
        corrected_neg, corrected_pos = 0, 0
        for t in range(6):
            if t == 0:
                if iter_num == 1:
                    continue
                sort_attrs = ["score"]
                cur_budget = budget
                cond = df["trans_neg"] < 0
            elif t > 0:
                sort_attrs = ["conf"]
                if t <= 2:
                    if iter_num == 1:
                        cur_budget = (budget - len(label_list)) // (5 - t)
                    else:
                        cur_budget = (budget - len(label_list)) // (3 - t)
                elif t == 3:
                    if corrected_neg + corrected_pos == 0:
                        ratio = 0.5
                    else:
                        ratio = corrected_neg / (corrected_neg + corrected_pos)
                        ratio = np.clip(ratio, 0.3, 0.7)
                    cur_budget = int((budget - len(label_list)) * ratio)
                else:
                    cur_budget = budget - len(label_list)
                if t == 1 or (t == 3 and iter_num == 1):
                    cond = df["prob"] <= 0.5  # & (df["mc_prob"] <= 0.5)
                elif t == 2 or (t == 4 and iter_num == 1):
                    cond = df["prob"] >= 0.5  # & (df["mc_prob"] >= 0.5)
                else:
                    cond = df["prob"] > 0
            if cur_budget <= 0:
                continue
            cnt, corrected_cnt = 0, 0
            df_sub = df[cond].sort_values(sort_attrs, ascending=True)
            print(f"sub_candidates: {len(df_sub)}")
            while True:
                df_sub = df[cond].sort_values(sort_attrs, ascending=True)
                flag = False
                for _, row in df_sub[attr_list].iterrows():
                    if cnt >= cur_budget:
                        break
                    lid, rid, prob, conf, trans_diff, _, mc_prob, score = row
                    if (lid, rid) in annotated_data or lid not in indices:
                        continue
                    trans_vlt = pred_collection.pred_dict[(lid, rid)].trans_vlt_wt
                    if t == 0 and trans_vlt == 0:
                        continue
                    flag = True
                    label = input_data.gt_dict.get((lid, rid), 0)
                    annotated_data[(lid, rid)] = label
                    label_list.append(label)
                    pred_collection.pred_dict[(lid, rid)].label = label
                    pred = pred_collection.pred_dict[(lid, rid)]
                    print(
                        f"annot{len(label_list)}: lid={int(lid)}, rid={int(rid)}, label={label}, "
                        f"prob={prob:.4f}, mc_prob={mc_prob:.4f}, conf={conf:.4f}, "
                        f"trans_vlt={trans_vlt:.4f}, trans_diff={trans_diff:.4f}, score={score:.4f}"
                    )
                    cnt += 1
                    if label != int(prob > 0.5):
                        corrected_cnt += 1
                        if t == 1:
                            corrected_neg += 1
                        elif t == 2:
                            corrected_pos += 1
                    if iter_num > 1:
                        pred_collection.update_trans_vlt(
                            {lid: lid_bk[lid]}, predictor="corrected"
                        )
                        pred_collection.self_correct(
                            lid, rid, lid_bk[lid], mc_thr_dict, input_data.gt_dict
                        )
                        # update df for the next iteration
                        for rid1 in lid_bk[lid]:
                            pred = pred_collection.pred_dict[(lid, rid1)]
                            loc_cond = (df["lid"] == lid) & (df["rid"] == rid1)
                            df.loc[loc_cond]["score"] = (
                                2 * df.loc[loc_cond]["conf"] - 1
                            ) * (1 - 5 * pred.trans_vlt_wt)
                            df.loc[loc_cond]["trans_vlt"] = pred.trans_vlt_wt
                            df.loc[loc_cond]["trans_diff"] = (
                                pred.trans_vlt - pred.trans_vlt_rvs
                            )
                        break
                if cnt >= cur_budget or flag is False:
                    break
            print(f"t={t}, newly annotated: {cnt}, corrected: {corrected_cnt}\n")
            corrected_all += corrected_cnt
        print(f"newly annotated: {len(label_list)}, corrected: {corrected_all}\n")
    return annotated_data, pred_collection


def weighted_f1_score(y_true=[], y_pred=[], weights=None):
    if weights is None:
        return (
            f1_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
        )
    tp, fp, fn = 0, 0, 0
    for i, y in enumerate(y_true):
        if y == 1:
            if y_pred[i] == 1:
                tp += weights[i]
            else:
                fn += weights[i]
        else:
            if y_pred[i] == 1:
                fp += weights[i]
    if tp == 0:
        return 0, 0, 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def get_weight_dict(pred_collection: PredCollection, gt_dict={}, annotated_data={}):
    weight_dict = {}
    w1_cnt, w0_cnt = 0, 0
    w1_pred, w0_pred = [], []
    w1_label, w0_label = [], []
    for (lid, rid), pred in pred_collection.pred_dict.items():
        if (lid, rid) in annotated_data:
            weight_dict[(lid, rid)] = 2
        bk_pred = pred.bk_pred
        mc_prob = pred.mc_pred
        corrected = pred.get_prob(predictor="corrected")
        llm_prob = np.mean(pred.llm_pred) if len(pred.llm_pred) > 0 else None
        if bk_pred is None or mc_prob is None or llm_prob is None:
            continue
        mc_pred = int(mc_prob > 0.5)
        corrected_pred = int(corrected > 0.5)
        if bk_pred == mc_pred and bk_pred == int(llm_prob > 0.5):
            weight_dict[(lid, rid)] = 1 - 1e-4
            w1_cnt += 1
            w1_pred.append(bk_pred)
            w1_label.append(gt_dict.get((lid, rid), 0))
        elif bk_pred == mc_pred and mc_pred != corrected_pred:
            weight_dict[(lid, rid)] = 1e-2
            w0_cnt += 1
            w0_pred.append(corrected_pred)
            w0_label.append(gt_dict.get((lid, rid), 0))
    print(f"#samples w=1: {w1_cnt}, w=0: {w0_cnt}")
    print(f"w=1 f1: {f1_score(w1_label, w1_pred):.4f}")
    print(f"w=0 f1: {f1_score(w0_label, w0_pred):.4f}")
    return weight_dict


def retrive_uncertain_lids(
    lids, pseudo_by_mc, lid_bk, lid_sim, sim_lower_bound, sim_upper_bound
):
    lids_uncertain = []
    certain_cnt, uncertain_cnt = 0, 0
    for lid in lids:
        rids = lid_bk.get(lid, [])
        probs = []
        for rid in rids:
            probs.append(pseudo_by_mc.get((lid, rid), 0))
        probs = np.array(probs)
        cos_sim = lid_sim[lid, len(rids) - 1]
        if (
            np.sum(probs[-5:] < 0.5) != 5
            or (np.sum(probs > 0.5) == 0 and cos_sim > sim_lower_bound)
            or (np.sum(probs > 0.5) > 0 and cos_sim > sim_upper_bound)
        ):
            lids_uncertain.append(lid)

    return lids_uncertain


def retrive_uncertain_lids_train(lids, pseudo_by_mc, lid_bk):
    lids_uncertain = []
    certain_cnt, uncertain_cnt = 0, 0
    for lid in lids:
        rids = lid_bk.get(lid, [])
        probs = []
        for rid in rids:
            probs.append(pseudo_by_mc.get((lid, rid), 0))
        probs = np.array(probs)
        if np.sum(probs[-5:] < 0.5) != 5 or (
            np.sum(probs > 0.5) == 0 and max(probs) > 0.1
        ):
            lids_uncertain.append(lid)

    return lids_uncertain
