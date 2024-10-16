import os
import sys

import numpy as np

llm = int(sys.argv[1])

dataset_list = ["head", "AG", "DA", "DS", "FZ", "WA", "AB", "M"]
# dataset_list = ["head", "WA", "DA", "AB"]
for budget in range(2, 6):
    print(f"Budget={budget * 100}")
    for dataset in dataset_list:
        output = []
        f1_list = []
        query_num_list = []
        for version in range(int(sys.argv[2]), int(sys.argv[3])):
            if dataset == "head":
                output.append(f"R{version}-{llm}B [Test]:pre/rec/f1 #query_llm")
            else:
                filename = f"logs/test/llama3-{llm}b/{dataset}/{version}.log"
                llm_label, train_label, matcher_result = None, None, None
                if os.path.exists(filename):
                    with open(filename, "r") as f:
                        lines = f.readlines()
                        flag = False
                        query_num = 0
                        for line in lines:
                            if f"Budget = {budget * 100}" in line:
                                flag = True
                            if flag and "bk rec:" in line:
                                bk_rec = float(line.strip().split(":")[-1].strip())
                            if flag and "mc pre/rec/f1" in line:
                                matcher_result = line.strip().split(":")[-1].strip()
                                f1 = float(matcher_result.split("/")[2])
                                f1_list.append(f1)
                                break
                            if flag and "query_llm =" in line:
                                query_num = int(line.strip().split(" ")[-1].strip())
                                query_num_list.append(query_num)
                                break
                    output.append(
                        f"{dataset} {bk_rec:.4f}|{matcher_result} {query_num}"
                    )
                else:
                    output.append(f"{dataset} N/A N/A")
        if dataset == "head":
            output.append("AVG_F1")
            output.append("AVG_QUERY")
        elif len(f1_list) > 0:
            output.append(f"{np.mean(f1_list):.4f}")
            output.append(f"{np.mean(query_num_list):.0f}")
        else:
            output.append("N/A")
            output.append("N/A")
        print(" || ".join(output))
    print()
