import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


def get_model_name(name) -> str:
    if "llama3-" in name:
        if "bf" in name:
            version = name.split("-")[-1][:-2]
            return f"../../llama_models/Llama-3-{version}B-chat-hf"
        else:
            version = name.split("-")[-1][:-1]
            return f"../../llama_models/Llama-3-{version}B-hf"
    elif "llama2-" in name:
        if "bf" in name:
            version = name.split("-")[-1][:-2]
            return f"../../llama_models/llama-2-{version}b-chat-hf"
        else:
            version = name.split("-")[-1][:-1]
            return f"../../llama_models/llama-2-{version}b-hf"
    elif "llama" in name:  # llama7, llama13, llama30, llama65
        size = int(name[5:])
        if size == 65:
            return "huggyllama/llama-65b"
        else:
            return f"decapoda-research/llama-{size}b-hf"
    else:
        raise ValueError(f"Unknown model name: {name}")


class LLM:
    def __init__(self, llm_name: str) -> None:
        self.llm_name = get_model_name(llm_name)
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        if llm_name.startswith("llama2-") or llm_name.startswith("llama3-"):
            self.max_len = 4096
        else:
            self.max_len = 2048

    def load_tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(self.llm_name, device_map="auto")

    def load_model(self) -> AutoModelForCausalLM:
        return AutoModelForCausalLM.from_pretrained(
            self.llm_name, torch_dtype=torch.half, device_map="auto"
        )

    def inference(self, llm_prompt: str) -> float:
        inputs = self.tokenizer(
            [llm_prompt],
            # max_length=self.max_len,
            # truncation=True,
            return_tensors="pt",
        )
        outputs = self.model.generate(
            inputs.input_ids.cuda(),
            max_new_tokens=1,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        scores = outputs["scores"][0].detach().cpu().numpy()
        prob = self.score_to_prob(scores[0])
        return prob

    def score_to_prob(self, score):
        idxs, score = zip(
            *sorted(
                zip(list(range(len(score))), score),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        pos_score, neg_score = None, None
        for idx, s in zip(idxs, score):
            if pos_score is None and "yes" in self.tokenizer.decode(idx).lower():
                pos_score = s
            elif neg_score is None and "no" in self.tokenizer.decode(idx).lower():
                neg_score = s
            if pos_score is not None and neg_score is not None:
                break
        pos_score = np.exp(pos_score) / (np.exp(pos_score) + np.exp(neg_score))
        return pos_score
