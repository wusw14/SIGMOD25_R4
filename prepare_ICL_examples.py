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
from utils.predict import PredCollection
from utils.prompt import Prompt
from utils.utils import (
    eval_pseudo,
    get_annotation,
    get_weight_dict,
    load_past_predictions,
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


def get_seed_icl_examples(
    llm: LLM, pred_collection: PredCollection, input_data: InputData
):
    icl_examples = gen_augmented_data(
        input_data.attrs, input_data.left_entries, input_data.right_entries, 50
    )
    prompt_builder = Prompt(input_data)
    for _ in range(3):
        first_query = True
        llm_pred_dict = {}
        for (lid, rid), pred in pred_collection.pred_dict.items():
            if pred.query_llm(only_llm=True):
                llm_prompt = prompt_builder.construct_prompt(
                    icl_examples,
                    query=(lid, rid),
                    same_table=False,
                    trans_examples=False,
                )
                prob = llm.inference(llm_prompt=llm_prompt)
                llm_pred_dict[(lid, rid)] = prob
                if first_query:
                    print(llm_prompt)
                    first_query = False
        # update pred_collection
        pred_collection.add_dict(llm_pred_dict, predictor="llm")

    pd_dict = {"lid": [], "rid": [], "pred": []}
    for (lid, rid), pred in pred_collection.pred_dict.items():
        llm_pred = pred.agg_prob(only_llm=True)
        pd_dict["lid"].append(lid)
        pd_dict["rid"].append(rid)
        pd_dict["pred"].append(llm_pred)
    llm_pred_df = pd.DataFrame(pd_dict)
    # sample 10 examples from each group
    icl_examples = []
    budget = 100
    cnt = 0
    for i in range(5):
        for j in range(2):
            if j == 0:
                cond1 = llm_pred_df["pred"] > 0.5 - (i + 1) * 0.1
                cond2 = llm_pred_df["pred"] <= 0.5 - i * 0.1
            else:
                cond1 = llm_pred_df["pred"] > 0.5 + i * 0.1
                cond2 = llm_pred_df["pred"] <= 0.5 + (i + 1) * 0.1
            df_sub = llm_pred_df[cond1 & cond2]
            cur_budget = (budget - len(icl_examples)) // (10 - cnt)
            if len(df_sub) > cur_budget:
                df_sub = df_sub.sample(cur_budget)
            cnt += 1
            for _, row in df_sub.iterrows():
                lid, rid, pred = row["lid"], row["rid"], row["pred"]
                label = input_data.gt_dict.get((lid, rid), 0)
                conf = max(pred, 1 - pred)
                if int(pred > 0.5) == label:
                    icl_examples.append([lid, rid, label, conf])
                else:
                    icl_examples.append([lid, rid, label, -conf])
    return icl_examples


def main():
    start_time = time.time()
    """
    Initialization
    """
    # [1] load_data from files
    input_data = InputData(
        path=args.path, dataset=args.dataset, entry_type=args.entry_type
    )

    # [2] init model
    model_path = os.path.join("checkpoints", args.llm, args.dataset, str(args.run_id))
    if os.path.exists(model_path) == False:
        os.makedirs(model_path)
    blocker = Blocker(lm="sent-bert", model_path=model_path, lr=args.lr)
    llm = LLM(args.llm)

    # [3] initialize in-context examples
    # get topk similar examples for each left entry based on Sent-BERT
    top_sim_data = blocker.get_init_topk(input_data, topk=args.topk)
    pred_collection = PredCollection(only_llm=True)
    lids = list(input_data.train_indices) + list(input_data.valid_indices)
    if len(lids) > 500:
        lids = random.sample(lids, 500)
    for lid in lids:
        for rid in top_sim_data.lid_topk[lid][:2]:
            pred_collection.add(lid, rid)
    icl_examples = get_seed_icl_examples(llm, pred_collection, input_data)

    icl_examples = pd.DataFrame(icl_examples, columns=["lid", "rid", "label", "conf"])
    filename = f"{args.path}/{args.dataset}/icl_examples_{args.llm}.csv"
    icl_examples.to_csv(filename, header=True)

    print(f"TIME: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
