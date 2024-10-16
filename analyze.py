import os
import sys

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.io import InputData, get_dataset_fullname
from utils.utils import load_llm_preds

dataset = sys.argv[1]
dataset = get_dataset_fullname(dataset)
path = f"data/ER-Magellan"
if dataset == "monitor":
    path = f"data/Alaska"
elif "Abt" in dataset:
    path = os.path.join(path, "Textual")
else:
    path = os.path.join(path, "Structured")
filename = f"results/llama3-70b/{dataset}/38/pred_dict3.txt"
pred_dict = load_llm_preds(filename)
input_data = InputData(path, dataset)

y_pred_list, y_truth_list = [], []
for (lid, rid), llm_probs in pred_dict.items():
    y_truth = input_data.gt_dict.get((lid, rid), 0)
    y_pred = int(np.mean([1 if p > 0.5 else 0 for p in llm_probs]) > 0.5)
    y_pred_list.append(y_pred)
    y_truth_list.append(y_truth)
    if y_truth != y_pred:
        print(f"({lid}, {rid}): {y_truth} -> {y_pred}: {llm_probs}")
        for v1, v2 in zip(input_data.left_entries[lid], input_data.right_entries[rid]):
            print(f"{v1} || {v2}")
        print("------------\n")

pre = precision_score(y_truth_list, y_pred_list)
rec = recall_score(y_truth_list, y_pred_list)
f1 = f1_score(y_truth_list, y_pred_list)
print(f"pre/rec/f1: {pre:.4f}/{rec:.4f}/{f1:.4f}")
