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
from utils.predict import PredCollection, PairData
from utils.prompt import Prompt
from utils.utils import (
    eval_pseudo,
    get_annotation,
    get_weight_dict,
    load_llm_preds,
    retrive_uncertain_lids,
)
from sklearn.metrics import precision_score, recall_score

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


def train():
    start_time = time.time()
    """
    Initialization
    """
    # [1] load_data from files
    input_data = InputData(
        path=args.path, dataset=args.dataset, entry_type=args.entry_type
    )
    prompt_builder = Prompt(input_data)

    # [2] init model
    model_path = os.path.join("checkpoints", args.llm, args.dataset, str(args.run_id))
    if os.path.exists(model_path) == False:
        os.makedirs(model_path)
    blocker = Blocker(lm="sent-bert", model_path=model_path, lr=args.lr)
    matcher = Matcher(lm=args.lm, model_path=model_path, lr=args.lr)
    # llm = LLM(args.llm)

    result_path = os.path.join("results", args.llm, args.dataset, str(args.run_id))

    for iter_num in range(2, 5):
        start_time = time.time()
        blocker.load(iter_num=iter_num)
        matcher.load(iter_num=iter_num)

        # get valid data from annotations
        annot_data = pd.read_csv(f"{result_path}/icl_examples{iter_num}.csv")
        input_data.icl_examples = annot_data[["lid", "rid", "label", "conf"]].values
        icl_examples = annot_data[["lid", "rid", "label", "conf"]].values
        annot_data = annot_data[["lid", "rid", "label"]].values.astype(int)
        valid_data = []
        for lid, rid, label in annot_data:
            if lid in input_data.valid_indices:
                valid_data.append(PairData(lid, rid, label))

        # mc predictions on valid data
        pairs_mc_valid = prepare_mc_valid_data(valid_data, input_data)
        mc_preds_valid = matcher.predict(pairs_mc_valid)

        # load llm predictions
        llm_preds = load_llm_preds(f"{result_path}/pred_dict{iter_num}.txt")

        # preds on valid data
        valid_preds = []
        result_summ = defaultdict(int)
        for d in valid_data:
            lid, rid, label = d.lid, d.rid, d.prob
            mc_pred = mc_preds_valid.get((lid, rid), 0.5)
            llm_pred = llm_preds.get((lid, rid), None)
            if llm_pred is not None and len(llm_pred) > 0:
                llm_pred = np.mean(llm_pred)
                valid_preds.append([mc_pred, llm_pred, label])
                result_summ[(int(mc_pred > 0.5), int(llm_pred > 0.5), label)] += 1

        valid_preds = sorted(valid_preds, key=lambda x: x[0])
        valid_preds = np.array(valid_preds)
        print(f"valid_preds: {valid_preds.shape}, pos: {np.sum(valid_preds[:, 2])}")

        top_sim_data = blocker.get_topk(input_data, topk=args.topk, cross_table=True)
        lids = list(input_data.test_indices)
        pseudo_by_bk = blocker.gen_pseudo(top_sim_data, lids=lids)
        lid_sim = top_sim_data.lid_sim
        print(lid_sim.shape)
        valid_sim = []
        for d in valid_data:
            lid, rid, label = d.lid, d.rid, d.prob
            if label == 1:
                try:
                    index = np.where(top_sim_data.lid_topk[lid] == rid)[0][0]
                except:
                    continue
                valid_sim.append(lid_sim[lid][index])
        print(valid_sim)
        sim_upper_bound = np.mean(valid_sim)
        sim_lower_bound = np.mean(valid_sim) - 2.33 * np.std(valid_sim)
        print(f"sim_upper_bound: {sim_upper_bound}, sim_lower_bound: {sim_lower_bound}")

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
            lids = retrive_uncertain_lids(
                lids, pseudo_by_mc, lid_bk, lid_sim, sim_lower_bound, sim_upper_bound
            )
            if len(lids) == 0:
                break

        print(f"Budget = {(iter_num + 1) * 100}")
        # eval_pseudo(input_data.gt_dict, pseudo_by_mc, predictor="mc")
        bk_rec = []
        for (lid, rid), label in input_data.gt_dict.items():
            if lid not in input_data.test_indices or label != 1:
                continue
            if lid in lid_bk and rid in lid_bk[lid]:
                bk_rec.append(1)
            else:
                bk_rec.append(0)
        bk_rec = np.mean(bk_rec)
        pred_list, label_list = [], []
        for (lid, rid), prob in pseudo_by_mc.items():
            pred_list.append(int(prob > 0.5))
            label_list.append(input_data.gt_dict.get((lid, rid), 0))
        pre = precision_score(label_list, pred_list)
        rec = recall_score(label_list, pred_list)
        rec = rec * bk_rec
        f1 = 2 * pre * rec / (pre + rec)
        print(f"bk rec:{bk_rec:.4f}")
        print(f"mc pre/rec/f1:{pre:.4f}/{rec:.4f}/{f1:.4f}")
        print(f"Time: {time.time() - start_time:.4f}s")


if __name__ == "__main__":
    train()
