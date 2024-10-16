import argparse
import os
import random
import time
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from blocker import Blocker
from llm import LLM
from matcher import Matcher
from sklearn.cluster import KMeans
from utils.dataset import (
    gen_augmented_data,
    prepare_bk_train_data,
    prepare_bk_valid_data,
    prepare_mc_predict_data,
    prepare_mc_train_data,
    prepare_mc_valid_data,
)
from utils.io import InputData, get_dataset_fullname
from utils.predict import PredCollection, PairData, Prediction
from utils.prompt import Prompt
from utils.utils import (
    eval_pseudo,
    get_annotation,
    get_weight_dict,
    load_past_predictions,
    retrive_uncertain_lids_train,
)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="../KD_v2/data/")
parser.add_argument("--dataset", type=str, default="wdc/shoes")
parser.add_argument("--run_id", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_len", type=int, default=256)
parser.add_argument("--add_token", type=bool, default=True)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--finetuning", dest="finetuning", action="store_true")
parser.add_argument("--save_model", dest="save_model", action="store_true")
parser.add_argument("--lm", type=str, default="roberta")
parser.add_argument("--llm", type=str, default="llama3-8b")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--fp16", dest="fp16", action="store_true")
parser.add_argument("--add_trans_loss", dest="add_trans_loss", action="store_true")
parser.add_argument("--only_llm", dest="only_llm", action="store_true")
parser.add_argument("--weighted_train", dest="weighted_train", action="store_true")
parser.add_argument("--topk", type=int, default=5)
parser.add_argument("--entry_type", type=str, default="Product")
parser.add_argument("--aug_type", type=str, default="random")
parser.add_argument("--budget", type=int, default=100)

args = parser.parse_args()

if args.dataset in ["DA", "DS"]:
    args.entry_type = "Paper"
elif args.dataset == "FZ":
    args.entry_type = "Resturant"
elif args.dataset == "IA":
    args.entry_type = "Song"
else:
    args.entry_type = "Product"

dataset = get_dataset_fullname(args.dataset)

if "camera" in dataset or "monitor" in dataset:
    args.path = "../KD_v2/data/Alaska"
elif "Abt" in dataset:
    args.path = "../KD_v2/data/ER-Magellan/Textual"
else:
    args.path = "../KD_v2/data/ER-Magellan/Structured"
args.dataset = dataset

# set seeds
seed = args.run_id
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print(args)


def sample_indices(indices, num, lid_sim):
    if len(indices) <= num:
        return indices
    sim_scores = lid_sim[indices][:, 0]
    df = pd.DataFrame({"lid": indices, "sim": sim_scores})
    df["group"] = pd.cut(df["sim"], bins=10, labels=False)
    # sample from each group
    sampled_indices = []
    for i in range(10):
        group_indices = df[df["group"] == i]["lid"].tolist()
        if len(group_indices) > 0:
            sampled_indices.extend(
                random.sample(group_indices, min(num // 10, len(group_indices)))
            )
    if len(sampled_indices) < num:
        sampled_indices.extend(
            random.sample(
                list(set(indices) - set(sampled_indices)), num - len(sampled_indices)
            )
        )
    return sampled_indices


def train():
    start_time = time.time()
    """
    Initialization
    """
    # TODO: just for testing, remove later
    # load past predictions for running efficiency

    # llm_pred_past, llm_pred_rids_past = {}, {}

    # [1] load_data from files
    input_data = InputData(
        path=args.path, dataset=args.dataset, entry_type=args.entry_type
    )

    # [2] init model
    model_path = os.path.join("checkpoints", args.llm, args.dataset, str(args.run_id))
    if os.path.exists(model_path) == False:
        os.makedirs(model_path)
    blocker = Blocker(lm="sent-bert", model_path=model_path, lr=args.lr)
    matcher = Matcher(lm=args.lm, model_path=model_path, lr=args.lr)
    llm = LLM(args.llm)

    # [3] initialize in-context examples
    # get topk similar examples for each left entry based on Sent-BERT
    top_sim_data = blocker.get_init_topk(input_data, topk=3)
    pseudo_by_bk = blocker.gen_pseudo(
        top_sim_data,
        lids=list(input_data.train_indices) + list(input_data.valid_indices),
    )
    icl_examples = input_data.icl_examples

    """
    Iterative Training until blocker converges
    """
    pred_collection = PredCollection(only_llm=args.only_llm)
    pred_collection.add_dict(pseudo_by_bk, predictor="bk", cross_table=True)
    lid_bk = None
    prompt_builder = Prompt(input_data)
    iter_num = 0
    annotated_data = {(lid, rid): label for lid, rid, label, _ in icl_examples}
    for (lid, rid), label in annotated_data.items():
        pred = Prediction()
        pred.label = label
        pred_collection.pred_dict[(lid, rid)] = pred
    mc_thr_dict = {"pos": 1, "neg": 0}
    for _ in range(4):
        iter_num += 1
        print(f"----------------\nITERATION {iter_num}:\n")
        llm_pred_past, llm_pred_rids_past = load_past_predictions(
            args.llm, args.dataset, args.run_id, iter_num
        )

        if lid_bk is None:
            lid_bk = defaultdict(list)
            llm_pred_dict = {}
            for (lid, rid), llm_probs in llm_pred_past.items():
                if len(llm_probs) > 0 and (lid, rid) not in annotated_data:
                    pred_collection.pred_dict[(lid, rid)] = Prediction(
                        llm_pred=llm_probs
                    )
                    lid_bk[lid].append(rid)

        # [1.1] query LLM for pseudo labels
        llm_query_cnt = 0
        for t in range(min(3, (iter_num - 1) * 3)):
            first_query = True
            llm_pred_dict = {}
            for lid in lid_bk:
                for rid in lid_bk[lid]:
                    pred = pred_collection.pred_dict.get((lid, rid), None)
                    if (
                        pred is None
                        or pred.query_llm(args.only_llm, mc_thr_dict)
                        or (pred.mc_trans_vlt > 0 and len(pred.llm_pred) == 0)
                    ):
                        llm_query_cnt += 1
                        try:
                            prob = llm_pred_past[(lid, rid)][t]
                        except:
                            prob = None
                        if prob is None:  # or iter_num > 1:
                            llm_prompt = prompt_builder.construct_prompt(
                                icl_examples,
                                query=(lid, rid),
                                same_table=False,
                                trans_examples=True,
                            )
                            prob = llm.inference(llm_prompt=llm_prompt)
                        llm_pred_dict[(lid, rid)] = prob
            pred_collection.add_dict(llm_pred_dict, predictor="llm", iter_num=iter_num)
            eval_pseudo(input_data.gt_dict, llm_pred_dict, predictor="llm")
        print(f"llm query cnt: {llm_query_cnt}")
        print(f"Time: {(time.time() - start_time)/3600:.4f}H")

        # [1.2] query llm for pseudo labels for rids
        llm_query_cnt = 0
        for t in range(min(3, (iter_num - 1) * 3)):
            llm_pred_dict = {}
            for lid, rids in lid_bk.items():
                for j1 in range(len(rids)):
                    for j2 in range(j1 + 1, len(rids)):
                        rid1, rid2 = rids[j1], rids[j2]
                        y1 = annotated_data.get((lid, rid1), -1)
                        y2 = annotated_data.get((lid, rid2), -1)
                        if y1 == -1:
                            pred1 = pred_collection.pred_dict.get((lid, rid1), None)
                            y1 = (
                                pred1.get_prob(predictor="corrected")
                                if pred1 is not None
                                else 0
                            )
                        if y2 == -1:
                            pred2 = pred_collection.pred_dict.get((lid, rid2), None)
                            y2 = (
                                pred2.get_prob(predictor="corrected")
                                if pred2 is not None
                                else 0
                            )
                        if y1 < 0.5 and y2 < 0.5:
                            continue
                        pred = pred_collection.pred_dict_rtb.get((rid1, rid2), None)
                        if pred is not None:
                            y3 = pred.get_prob(predictor="corrected")
                            if int(y1 > 0.5) + int(y2 > 0.5) + int(y3 > 0.5) != 2:
                                continue
                        if pred is None or pred.query_llm(args.only_llm, mc_thr_dict):
                            llm_query_cnt += 1
                            try:
                                prob = llm_pred_rids_past[(rid1, rid2)][t]
                            except:
                                prob = None
                            if prob is None:  # or iter_num > 1:
                                llm_prompt = prompt_builder.construct_prompt(
                                    icl_examples,
                                    query=(rid1, rid2),
                                    same_table=True,
                                    trans_examples=True,
                                )
                                prob = llm.inference(llm_prompt=llm_prompt)
                            llm_pred_dict[(rid1, rid2)] = prob
                            llm_pred_dict[(rid2, rid1)] = prob
            pred_collection.add_dict(
                llm_pred_dict, predictor="llm", cross_table=False, iter_num=iter_num
            )
        print(f"llm query cnt on the same table: {llm_query_cnt}")
        print(f"Time: {(time.time() - start_time)/3600:.4f}H")
        pred_collection.save(
            args.llm, dataset, args.run_id, input_data.gt_dict, iter_num
        )

        # [2.1] update the corrected
        if iter_num > 1:
            # pred_collection.update_trans_vlt(lid_bk, predictor="llm")
            pred_collection.update_corrected_pred(lid_bk, input_data.gt_dict)
            for (lid, rid), label in annotated_data.items():
                pred_collection.self_correct(
                    lid, rid, lid_bk[lid], mc_thr_dict, input_data.gt_dict
                )

        # [2.2] correct llm's labels
        annotated_data, pred_collection = get_annotation(
            input_data,
            lid_bk,
            pred_collection,
            annotated_data=annotated_data,
            budget=args.budget,
            mc_thr_dict=mc_thr_dict,
            iter_num=iter_num,
        )

        # weight_dict = get_weight_dict(
        #     pred_collection, input_data.gt_dict, annotated_data
        # )
        weight_dict = {}

        # [3.1] retrieve data for training and validation for blocker
        bk_train_data = pred_collection.retrieve_data_for_bk(
            lid_bk=lid_bk,
            lids=input_data.train_indices,
            annotated_data=annotated_data,
            weight_dict=weight_dict,
        )
        bk_valid_data = pred_collection.retrieve_data_for_bk(
            lid_bk=lid_bk,
            lids=input_data.valid_indices,
            annotated_data=annotated_data,
            weight_dict=weight_dict,
        )
        ensemble_dict = {(d.lid, d.rid): d.prob for d in bk_train_data}
        eval_pseudo(input_data.gt_dict, ensemble_dict, predictor="Train")
        ensemble_dict = {(d.lid, d.rid): d.prob for d in bk_valid_data}
        eval_pseudo(input_data.gt_dict, ensemble_dict, predictor="Valid")
        bk_train_data = prepare_bk_train_data(bk_train_data, input_data)
        # bk_valid_data = prepare_bk_valid_data(bk_valid_data, input_data)

        # [3.2] update blocker
        blocker = Blocker(lm="sent-bert", model_path=model_path, lr=args.lr)
        bk_f1_best, bk_best_gap = blocker.evaluate(bk_valid_data, input_data)
        bk_early_stop = 0
        bk_epoch_min = 50000 / (
            len(input_data.left_entries) + len(input_data.right_entries)
        )
        bk_epoch_min = min(max(5, bk_epoch_min), 10)
        print(f"[bk_epoch_min]: {bk_epoch_min}")
        for epoch in range(1, args.n_epochs + 1):
            train_loss = blocker.train(
                bk_train_data, args.batch_size, args.weighted_train
            )
            torch.cuda.empty_cache()
            bk_f1, bk_gap = blocker.evaluate(bk_valid_data, input_data)
            print(
                f"[Blocker]: epoch {epoch}, train_loss {train_loss:4f} "
                f"valid_score {bk_f1:.4f}, valid_best {bk_f1_best:.4f}"
            )
            if bk_f1 > bk_f1_best or bk_f1 == bk_f1_best and bk_gap > bk_best_gap:
                bk_f1_best = bk_f1
                bk_best_gap = bk_gap
                blocker.save(iter_num=iter_num)
                bk_early_stop = 0
            else:
                bk_early_stop += 1
                if bk_early_stop >= bk_epoch_min:
                    break

        # [4.1] retrieve data for training and validation for matcher
        mc_train_data = pred_collection.retrieve_data_for_mc(
            lid_bk=lid_bk,
            lids=input_data.train_indices,
            annotated_data=annotated_data,
            weight_dict=weight_dict,
        )
        # pos weights vs neg weights
        pos_weight, neg_weight = 0, 0
        pos_num, neg_num = 0, 0
        for d in mc_train_data:
            prob, w = d.prob, d.w
            if prob > 0.5:
                pos_weight += w
                pos_num += 1
            else:
                neg_weight += w
                neg_num += 1
        print(f"[POS/NEG] {pos_num} / {neg_num} [Weight]{pos_weight / neg_weight:.4f}")

        mc_valid_data = pred_collection.retrieve_data_for_mc(
            lid_bk=lid_bk,
            lids=input_data.valid_indices,
            annotated_data=annotated_data,
            weight_dict=weight_dict,
        )
        print(f"[Train data]: {len(mc_train_data)}, [Valid data]: {len(mc_valid_data)}")

        # [4.2] update matcher
        if args.add_trans_loss:
            add_trans_loss = iter_num > 1
        else:
            add_trans_loss = False
        matcher = Matcher(lm=args.lm, model_path=model_path, lr=args.lr)
        mc_score_best, mc_f1_best = 0, 0
        mc_early_stop = 0
        mc_train_data = prepare_mc_train_data(mc_train_data, input_data)
        mc_valid_data = prepare_mc_train_data(mc_valid_data, input_data)
        for epoch in range(1, args.n_epochs + 1):
            train_loss = matcher.train(
                mc_train_data,
                args.batch_size,
                add_trans_loss,
                args.weighted_train,
            )
            torch.cuda.empty_cache()
            mc_score, mc_f1, pre, rec = matcher.evaluate(
                mc_valid_data, add_trans_loss, args.weighted_train
            )
            print(
                f"[Matcher]: epoch {epoch}, train_loss {train_loss:4f} "
                f"valid_score {mc_score:.4f}, valid_best {mc_score_best:.4f} "
                f"valid_f1 {mc_f1:.4f}, pre {pre:.4f}, rec {rec:.4f} "
                f"valid_best_f1 {mc_f1_best:.4f}"
            )
            if mc_score > mc_score_best or (
                mc_score == mc_score_best
                and (mc_f1 > mc_f1_best or mc_f1 == mc_f1_best and epoch < 10)
            ):
                mc_score_best = mc_score
                mc_f1_best = mc_f1
                matcher.save(iter_num=iter_num)
                mc_early_stop = 0
            elif mc_f1 > 0:
                mc_early_stop += 1
                if mc_early_stop >= 10:
                    break

        # [5] update predictions
        # [5.1] update the predictions of retrieved pairs by the blocker
        blocker.load()
        top_sim_data = blocker.get_topk(input_data, topk=args.topk, cross_table=True)
        lids = list(input_data.train_indices) + list(input_data.valid_indices)
        pseudo_by_bk = blocker.gen_pseudo(top_sim_data, lids=lids)

        # [5.2] update the predictions of retrieved pairs by the matcher
        matcher.load()
        pseudo_by_mc = {}
        lid_bk = defaultdict(list)
        for k in range(5, args.topk + 1):
            pairs_cross_table = prepare_mc_predict_data(
                input_data, top_sim_data.lid_topk[:, :k], lid_bk, lids=lids
            )
            print(f"pairs_cross_table: {len(pairs_cross_table)}")
            pseudo_by_mc.update(matcher.predict(pairs_cross_table))
            if k == 5:
                for lid in lids:
                    lid_bk[lid].extend(top_sim_data.lid_topk[lid][:k])
            else:
                for lid in lids:
                    lid_bk[lid].append(top_sim_data.lid_topk[lid][k - 1])
            # update lids
            lids = retrive_uncertain_lids_train(lids, pseudo_by_mc, lid_bk)
            if len(lids) == 0:
                break

        # [5.2] save the updated predictions to the pred_collection
        pseudo_by_bk_filtered = {}
        for (lid, rid), prob in pseudo_by_bk.items():
            if rid in lid_bk.get(lid, []):
                pseudo_by_bk_filtered[(lid, rid)] = prob
        eval_pseudo(input_data.gt_dict, pseudo_by_bk_filtered, predictor="bk")
        pred_collection.add_dict(
            pseudo_by_bk_filtered, predictor="bk", cross_table=True, iter_num=iter_num
        )
        eval_pseudo(input_data.gt_dict, pseudo_by_mc, predictor="mc")
        lids = list(input_data.train_indices) + list(input_data.valid_indices)
        pairs_same_table = prepare_mc_predict_data(
            input_data, lid_bk=lid_bk, lids=lids, cross_table=False
        )
        pseudo_rids_by_mc = matcher.predict(pairs_same_table)
        pred_collection.add_dict(
            pseudo_by_mc, predictor="mc", cross_table=True, iter_num=iter_num
        )
        pred_collection.add_dict(
            pseudo_rids_by_mc, predictor="mc", cross_table=False, iter_num=iter_num
        )
        eval_pseudo(
            input_data.gt_dict,
            pseudo_by_mc,
            predictor="mc_valid",
            indices=input_data.valid_indices,
        )

        # [5.3] update transitivity violation based on matcher's predictions
        pred_collection.update_trans_vlt(lid_bk, predictor="mc")

        # [5.3] update in-context examples
        for (lid, rid), label in annotated_data.items():
            contained = False
            for d in icl_examples:
                if d[0] == lid and d[1] == rid:
                    contained = True
                    break
            if not contained:
                pred = pred_collection.pred_dict.get((lid, rid), None)
                if pred is not None:
                    prob = pred.agg_prob(only_llm=True)
                    conf = max(prob, 1 - prob)
                    if int(prob > 0.5) == label:
                        icl_examples.append([lid, rid, label, conf])
                    else:
                        icl_examples.append([lid, rid, label, -conf])
                else:
                    icl_examples.append([lid, rid, label, 1])
        print(f"ICL examples: {len(icl_examples)}")

        # [5.4] update matcher threshold dict
        pairs = []
        for (lid, rid), label in annotated_data.items():
            if lid not in input_data.valid_indices:
                continue
            pairs.append(PairData(lid, rid, 0.5))
        pairs = prepare_mc_valid_data(pairs, input_data)
        pairs_probs = matcher.predict(pairs)
        prob_list, label_list = [], []
        for (lid, rid), prob in pairs_probs.items():
            prob_list.append(prob)
            label_list.append(annotated_data[(lid, rid)])
        probs, labels = np.array(prob_list), np.array(label_list)
        probs, labels = zip(*sorted(zip(probs, labels), key=lambda x: x[0]))
        probs, labels = np.array(probs), np.array(labels)
        print(f"##### probs({len(probs)}): {list(probs)}\nlabels: {list(labels)}")

        pos_thr, neg_thr = 1, 0
        for i in range(len(probs) - 5):
            if probs[i] > 0.5:
                acc = np.mean(labels[i:])
                if acc == 1:
                    pos_thr = probs[i]
                    break
        for i in range(len(probs) - 1, 3, -1):
            if probs[i] < 0.5:
                acc = 1 - np.mean(labels[: i + 1])
                if acc == 1:
                    neg_thr = probs[i]
                    break
        pos_thr = max(pos_thr, np.median(probs[probs > 0.5]))
        neg_thr = min(neg_thr, np.median(probs[probs < 0.5]))
        mc_thr_dict = {"pos": pos_thr, "neg": neg_thr}
        print(f"##### mc_thr_dict: {mc_thr_dict}")

        # [5.5] evaluate pseduo by matcher based on thresholds
        pseudo_by_mc = {}
        for (lid, rid), pred in pred_collection.pred_dict.items():
            if (
                pred is None
                or pred.mc_pred is None
                or lid not in input_data.valid_indices
            ):
                continue
            if pred.mc_pred > pos_thr or pred.mc_pred < neg_thr:
                pseudo_by_mc[(lid, rid)] = pred.mc_pred
        print(f"##### [Valid by MC thr]: {len(pseudo_by_mc)}")
        eval_pseudo(input_data.gt_dict, pseudo_by_mc, predictor="mc_thr")
        print(f"Time: {(time.time() - start_time)/3600:.4f}H")

        # write the ICL examples to file
        df_icl = pd.DataFrame(icl_examples, columns=["lid", "rid", "label", "conf"])
        path = os.path.join("results", args.llm, dataset, str(args.run_id))
        df_icl.to_csv(f"{path}/icl_examples{iter_num}.csv", index=False)


if __name__ == "__main__":
    train()
