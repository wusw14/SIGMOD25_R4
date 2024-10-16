import os

import pandas as pd
from utils.dataset import SingleEntityDataset


def get_dataset_fullname(dataset: str) -> str:
    dataset_dict = {
        "AG": "Amazon-Google",
        "BR": "BeerAdvo-RateBeer",
        "DA": "DBLP-ACM",
        "DS": "DBLP-Scholar",
        "FZ": "Fodors-Zagats",
        "IA": "iTunes-Amazon",
        "WA": "Walmart-Amazon",
        "AB": "Abt-Buy",
        "M": "monitor",
    }
    return dataset_dict.get(dataset, dataset)


class InputData:
    def __init__(self, path: str, dataset: str, entry_type: str = "Product"):
        self.path = path
        self.dataset = dataset
        self.entry_type = entry_type
        self.train_indices = self.load_indices("train")
        self.valid_indices = self.load_indices("valid")
        self.test_indices = self.load_indices("test")
        self.gt_dict = self.load_matches()
        self.attrs, self.left_entries, self.right_entries = self.load_entries()
        self.left_entry_dataset = self.transform_entries_into_dataset(self.left_entries)
        self.right_entry_dataset = self.transform_entries_into_dataset(
            self.right_entries
        )
        self.icl_examples = self.load_icl_examples()

    def load_csv(self, file: str, sep: str = ",") -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(self.path, self.dataset, file), sep=sep, index_col=0
        )

    def load_matches(self):
        matches = self.load_csv("matches.csv")
        gt_dict = {}
        for lid, rid, y in matches[["ltable_id", "rtable_id", "label"]].values:
            gt_dict[(int(lid), int(rid))] = y
        return gt_dict

    def load_icl_examples(self):
        filepath = os.path.join(self.path, self.dataset, "icl_examples_llama3-70b.csv")
        if not os.path.exists(filepath):
            return None
        icl_examples = pd.read_csv(filepath)
        icl_examples = icl_examples[["lid", "rid", "label", "conf"]].values
        print(f"ICL examples: {sum(icl_examples[:,2])}/{len(icl_examples)}")
        icl_examples = [
            (int(lid), int(rid), int(y), conf) for lid, rid, y, conf in icl_examples
        ]
        return icl_examples

    def load_indices(self, split):
        idxs = self.load_csv(f"{split}_idxs.csv")
        return idxs["ltable_id"].values

    def load_entries(self):
        # HH: no need to compute and pass the `path`
        path = os.path.join(self.path, self.dataset)
        attrs, left_entries = self.read_entity(path, table="tableA")
        if "wdc" in path:  # ? does this equivlent to "wdc" in self.dataset?
            _, right_entries = self.read_entity(path, table="tableA")
        else:
            _, right_entries = self.read_entity(path, table="tableB")
        return attrs, left_entries, right_entries

    def read_entity(self, dataset, table=None):
        def shorten(x):
            x = str(x)
            x = x.split(" ")
            x = x[:40]
            x = " ".join(x)
            return x

        # TODO: replace pd.read_csv with self.load_csv
        if table is None:
            df = pd.read_csv(os.path.join(dataset, "tableA.csv"), sep=",", index_col=0)
            if "wdc" not in dataset:
                df = df.append(
                    pd.read_csv(
                        os.path.join(dataset, "tableB.csv"), sep=",", index_col=0
                    )
                )
        else:
            df = pd.read_csv(
                os.path.join(dataset, table + ".csv"), sep=",", index_col=0
            )
        if table is not None and "table" in table:
            if "Amazon-Google" in dataset:
                df = df[["title", "manufacturer", "price"]]
            elif "Walmart" in dataset:
                df = df[["title", "modelno"]]

        entity_list = df.values
        for i in range(len(entity_list)):
            entity_list[i] = [shorten(x) for x in entity_list[i]]
        return list(df.columns), list(entity_list)

    def transform_entries_into_dataset(self, entries):
        return SingleEntityDataset(
            entries, self.attrs, lm="sent-bert", max_len=128, add_token=True
        )

    def get_data_pair_value(self, data_indices):
        data_pair_value = []
        for lid, rid, y in data_indices:
            lid_values = self.left_entries[lid]
            rid_values = self.right_entries[rid]
            data_pair_value.append((lid_values, rid_values, y))
        return data_pair_value
