import os
import sys
import time
import warnings

import numpy as np
import pynvml

dataset_list = sys.argv[1]
gpus = sys.argv[2]

pynvml.nvmlInit()
gpuDeriveInfo = pynvml.nvmlSystemGetDriverVersion()
handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpus.split(",")[0]))
memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

print(memoryInfo.used / memoryInfo.total)
cnt, iter_num = 0, 0
# while True:
#     for gpu in gpus.split(","):
#         handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
#         memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         memory = memoryInfo.used / memoryInfo.total
#         if memory > 0.2:
#             cnt = 0
#             time.sleep(300)
#             iter_num += 1
#         else:
#             cnt += 1
#             print(f"cnt: {cnt}, memory: {memory}")
#             time.sleep(np.random.randint(10, 20))
#     if iter_num % 12 == 0:
#         print(f"{iter_num//12} hours passed")
#     if cnt > 20:
#         break

K = 20
add_trans_loss = False
weighted_train = False
only_llm = True

cnt = 0
for only_llm in [True, False][1:]:
    for add_trans_loss in [False, True][:1]:
        for weighted_train in [False, True][1:]:
            cnt += 1
            for run_id in range(10, 15):
                for dataset in dataset_list.split(","):
                    if only_llm:
                        continue
                    log_dir = f"logs/llama3-{sys.argv[3]}b/{dataset}"
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                    cmd = (
                        f"CUDA_VISIBLE_DEVICES={sys.argv[2]} python -u train.py --lr 1e-5"
                        f" --dataset {dataset}"
                        f" --run_id {run_id}"
                        f" --batch_size 64 --save_model --topk 20 --n_epochs 50"
                        f" --llm llama3-{sys.argv[3]}b"
                    )
                    if add_trans_loss:
                        cmd += " --add_trans_loss"
                    if weighted_train:
                        cmd += " --weighted_train"
                    if only_llm:
                        cmd += " --only_llm"
                    cmd += f" >> {log_dir}/{run_id}.log"
                    if os.path.exists(f"{log_dir}/{run_id}.log"):
                        continue
                    print(cmd)
                    os.system(cmd)

                    log_dir = f"logs/test/llama3-{sys.argv[3]}b/{dataset}"
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                    cmd = (
                        f"CUDA_VISIBLE_DEVICES={sys.argv[2]} python -u test.py --dataset {dataset}"
                        f" --topk 20 --run_id {run_id} --llm llama3-{sys.argv[3]}b"
                        f" > {log_dir}/{run_id}.log"
                    )
                    if os.path.exists(f"{log_dir}/{run_id}.log"):
                        continue
                    print(cmd)
                    os.system(cmd)


# for dataset in dataset_list.split(","):
#     cmd = (
#         f"CUDA_VISIBLE_DEVICES={sys.argv[2]} python -u prepare_ICL_examples.py --lr 1e-5"
#         f" --dataset {dataset}"
#         f" --run_id 1234"
#         f" --batch_size 64 --save_model"
#         f" --llm llama3-{sys.argv[3]}b"
#     )
#     print(cmd)
#     os.system(cmd)
