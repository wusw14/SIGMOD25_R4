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
parser.add_argument("--path", type=str, default="data/")
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
    args.path = "data/Alaska"
elif "Abt" in dataset:
    args.path = "data/ER-Magellan/Textual"
else:
    args.path = "data/ER-Magellan/Structured"
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
    # matcher = Matcher(lm=args.lm, model_path=model_path, lr=args.lr)

    # [3] initialize in-context examples
    # get topk similar examples for each left entry based on Sent-BERT
    top_sim_data = blocker.get_init_topk(input_data, topk=3)
    pseudo_by_bk = blocker.gen_pseudo(
        top_sim_data,
        lids=list(input_data.train_indices) + list(input_data.valid_indices),
    )
    del blocker
    icl_examples = input_data.icl_examples

    llm = LLM(args.llm)

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
    for _ in range(1):
        iter_num += 1
        print(f"----------------\nITERATION {iter_num}:\n")
        # llm_pred_past, llm_pred_rids_past = load_past_predictions(
        #     args.llm, args.dataset, args.run_id, iter_num
        # )
        llm_pred_past = {}
        train_indices = deepcopy(input_data.train_indices)
        valid_indices = deepcopy(input_data.valid_indices)
        if iter_num == 1:
            train_indices = sample_indices(train_indices, 600, top_sim_data.lid_sim)
            valid_indices = sample_indices(valid_indices, 200, top_sim_data.lid_sim)

        if lid_bk is None:
            lid_bk = defaultdict(list)
            for lid in list(train_indices) + list(valid_indices):
                for rid in top_sim_data.lid_topk[lid]:
                    lid_bk[lid].append(rid)

        # [1.1] query LLM for pseudo labels
        llm_query_cnt = 0
        for t in range(3):
            first_query = True
            llm_pred_dict = {}
            for lid in lid_bk:
                for rid in lid_bk[lid]:
                    if (lid, rid) in annotated_data:
                        continue
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
                            if first_query:
                                print(llm_prompt)
                                first_query = False

                        llm_pred_dict[(lid, rid)] = prob
                        if len(llm_pred_dict) % 200 == 0:
                            print(f"llm query cnt: {llm_query_cnt}")
            pred_collection.add_dict(llm_pred_dict, predictor="llm", iter_num=iter_num)
            eval_pseudo(input_data.gt_dict, llm_pred_dict, predictor="llm")
        print(f"llm query cnt: {llm_query_cnt}")
        print(f"Time: {(time.time() - start_time)/3600:.4f}H")
        pred_collection.save(
            args.llm, dataset, args.run_id, input_data.gt_dict, iter_num
        )

        # eval pseudo
        ensemble_dict = {}
        for (lid, rid), pred in pred_collection.pred_dict.items():
            if len(pred.llm_pred) > 0:
                llm_pred = pred.agg_prob(only_llm=True)
                ensemble_dict[(lid, rid)] = int(llm_pred > 0.5)
        eval_pseudo(input_data.gt_dict, ensemble_dict, predictor="ensemble")


if __name__ == "__main__":
    train()
