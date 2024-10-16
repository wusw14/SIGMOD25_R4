# Large Language Models as Selective Annotators in Low-Resource Entity Resolution

This is the source code of SAiLER, which selectively leverages the LLM as an annotator, thus alleviating the reliance on the human annotations.

Our paper is submitted to SIGMOD 2025. 

## Requirements
### Create conda environment and install packages
The implementation requires python 3.8.  

All the packages except apex could be installed via "pip intall <package_name>".  
```  
torch   
pandas   
scikit-learn   
packaging   
urllib3==1.26.6   
importlib-metadata   
sentence_transformers 
transformers  
apex
```

### Hardware environment
Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz  
NVIDIA A100 80GB  
Note: the experiments do not require the same hardware environment.

## Datasets
We conduct experiments on seven widely adopted public datasets from various domains for ER tasks. 
These datasets are obtained from the Magellan data repository and the Alaska benchmark. 
    
| Dataset  | \# entries $D,D'$ | \# matches | (%) matches 
| :----: | :----: | :----: | :----: |
| Amazon-Google (AG) | 1363, 3226 | 1300 | 0.0296 
| DBLP-ACM (DA) | 2616, 2294 | 2224 | 0.0371 
| DBLP-Scholar (DS) | 2616, 64263 | 5347 | 0.0032 
| Fodors-Zagats (FZ) | 533, 331 | 112 | 0.0635 
| Walmart-Amazon (WA) | 2554, 22074 | 1154 | 0.0020
| Abt-Buy (AB) | 1081, 1092 | 1098 | 0.0930 
| Monitor (M) | 603, 4323 | 343 | 0.0132 

The datasets used in this work can be downloaded from this.  
The downloaded datasets should be stored in the path "data/"

## Preparation
Annotate 100 samples as the candidte pool of in-context examples
```
CUDA_VISIBLE_DEVICES=0,1 python -u prepare_ICL_examples.py --lr 1e-5
    --dataset AG
    --llm llama3-70b
```


Initially query the LLM to generate labels for a subset of samples that are retrieved by the initial blocker, SBERT.
```
CUDA_VISIBLE_DEVICES=0,1 python -u preparation.py --lr 1e-5
    --dataset AG
    --llm llama3-70b
```

## Training
Train the blocker and the matcher by obtaining the labels from the blocker, matcher, LLM and human annotator. 

In our implementation, we set the number of iterations to 4, corresponding to the annotation budget B=500. The 1st iteration is to warm up the blocker and the matcher. The 2nd iteration corresponds to our experimental setting when B=300, and so on.
```
CUDA_VISIBLE_DEVICES=0,1,2 python -u train.py --lr 1e-5
    --dataset AG
    --run_id 10
    --batch_size 64 --save_model --topk 20 --n_epochs 50
    --llm llama3-70b
    --weighted_train
```

## Evaluation
To evaluate the performance of the trained models under the annotation budgets B=[300, 400, 500], run the test.py file as follows:
```
CUDA_VISIBLE_DEVICES=0 python -u test.py --dataset AG
     --topk 20 --run_id 10 --llm llama3-70b
```