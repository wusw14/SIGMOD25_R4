import os

import numpy as np
import pandas as pd
from utils.io import InputData

Q_FORMAT = {
    "q1": "Do {item} A and {item} B refer to the same entity?",
    "q2": "Are {item} A and {item} B the same?",
    "q3": "Do these two {item}s refer to the same entity?",
    "q4": "Are these two {item}s the same?",
}

TASK_DESP = {
    "t1": "The task is to identify whether the two {item}s refer to the same entity based on the attribute values.\n\n",
    "t2": "The task is to identify whether the two {item}s refer to the same entity.\n\n",
    "t3": "This is an entity matching task.\n\n",
    # "t3": "This is an entity matching task. The values of the attributes might be shuffled or be missing.\n\n",
}

INSTRUCT = {
    "i1": "Determine whether {item} A and {item} B refer to the same entity. ",
    "i2": "Determine whether the two {item}s refer to the same entity. ",
    "i3": "Determine whether the two {item}s are the same. ",
}

OUT_FORMAT = {
    "o1": "Give your answer as either yes or no.\n\n",
    "o2": "First give your answer as either yes or no, then briefly explain your thoughts.\n\n",
}


def get_prompt_parts(entry_type):
    return {
        "question": Q_FORMAT["q2"].format(item=entry_type),
        "desc": TASK_DESP["t3"].format(item=entry_type),
        "out": OUT_FORMAT["o1"],
    }


def val_transform(val, valType="tabular"):
    if type(val) == str and len(val.split()) > 100:
        val = val.split(".")[0]
    elif (
        ((type(val) == float or type(val) == int) and val == 0)
        or (len(str(str)) == 0)
        or (str(val) == "nan")
    ):
        if valType == "tabular":
            val = "NULL"
        else:
            val = "missing"
    return str(val)


class Prompt:
    def __init__(self, input_data: InputData):
        self.attrs = input_data.attrs
        self.left_entries = input_data.left_entries
        self.right_entries = input_data.right_entries
        self.entry_type = input_data.entry_type

    def serialization(self, valsA, valsB):
        valsA = "\t".join([f"{self.entry_type} A"] + [val_transform(v) for v in valsA])
        valsB = "\t".join([f"{self.entry_type} B"] + [val_transform(v) for v in valsB])
        cols = "\t".join(["Entry"] + self.attrs)
        text = f"{cols}\n{valsA}\n{valsB}"
        return text

    def construct_prompt(self, examples, query, same_table=False, trans_examples=False):
        prompt_parts = get_prompt_parts(self.entry_type)
        question = prompt_parts.get("question", "")
        desc = prompt_parts.get("desc", "")

        def sample(indices, num):
            if len(indices) <= num:
                return indices
            return np.random.choice(indices, num, replace=False)

        k = min(10, len(examples)) // 2
        # [1] sample k/2 matches and k/2 non-matches as icl examples
        # random selection
        if trans_examples == False:
            pos_indices, neg_indices = [], []
            for i in range(len(examples)):
                if examples[i][2] == 1:
                    pos_indices.append(i)
                else:
                    neg_indices.append(i)
            pos = np.random.choice(pos_indices, min(k, len(pos_indices)), replace=False)
            neg = np.random.choice(neg_indices, min(k, len(neg_indices)), replace=False)
            indices = np.concatenate([pos, neg])
        else:
            df = pd.DataFrame(
                examples, columns=["ltable_id", "rtable_id", "label", "conf"]
            )
            df["index"] = list(range(len(df)))
            indices = []
            for cond in [1, 0]:
                df_sub = df[df["label"] == cond]
                true_indices = df_sub[df_sub["conf"] > 0]["index"].values
                false_indices = df_sub[df_sub["conf"] < 0]["index"].values
                if len(true_indices) > len(false_indices):
                    indices.extend(sample(false_indices, 2))
                    indices.extend(sample(true_indices, k - min(2, len(false_indices))))
                else:
                    indices.extend(sample(true_indices, 2))
                    indices.extend(sample(false_indices, k - min(2, len(true_indices))))
        np.random.shuffle(indices)

        # [2] construct demonstrations
        demonstrations = ""
        for idx in indices:
            e1, e2, y = examples[idx][0], examples[idx][1], examples[idx][2]
            if trans_examples:
                e1 = self.left_entries[int(e1)]
                e2 = self.right_entries[int(e2)]
            if y == 1:
                answer = "Yes"
            else:
                answer = "No"
            input_text = self.serialization(e1, e2)
            demonstrations += f"{input_text}\n{question} {answer}\n\n"

        lid, rid = int(query[0]), int(query[1])
        query_text = self.serialization(
            self.right_entries[lid] if same_table else self.left_entries[lid],
            self.right_entries[rid],
        )
        query = f"{query_text}\n{question}"
        llm_prompt = f"{desc}{demonstrations}{query}"
        return llm_prompt
