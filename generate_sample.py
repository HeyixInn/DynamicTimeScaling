from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import json

from typing import Tuple

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4"
os.environ["HUGGINGFACE_API_KEY"] = "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"

def get_prompt(question, model_type='Qwen'):
    SYSTEM_PROMPT = {
        'Qwen': "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n",
        'Llama': "<|im_start|>system\nYou are a helpful assistant. Whenever you give a final answer, wrap it using LaTeX boxed syntax like \\boxed{answer}.<|im_end|>\n"
    }    
    def get_user_prompt(q, model_type):
        if model_type=='deepseek':
            return "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n"
        elif model_type=='Qwen' or model_type=='Llama':
            return SYSTEM_PROMPT[model_type]+"<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n"

    return get_user_prompt(question, model_type)
def load_my_dataset(data_name):
    ds  = load_dataset("simplescaling/s1K-1.1")['train']
    if data_name=='aime':
        aime = []
        for d in ds:
            if 'qq8933/AIME_1983_2024' in d['source_type']:
                aime.append(d)
        return aime
    elif data_name=='omni':
        aime = []
        for d in ds:
            if 'KbsdJames/Omni-MATH' in d['source_type']:
                aime.append(d)
        return aime
    
ID_2_MODELS = {
    # authors'
    0: "simplescaling/s1-32B",
    # deepseek models
    1: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    2: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    3: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # Qwen
    4: "Qwen/Qwen2.5-7B-Instruct",
    5: "Qwen/Qwen2.5-32B-Instruct",
    # Llama
    6: "meta-llama/Meta-Llama-3-8B-Instruct"
}

def get_model_type(model_name):
    if 'deepseek' in model_name: 
        return 'deepseek'
    elif 'Qwen' in model_name: 
        return 'Qwen'
    elif 'Llama' in model_name: 
        return 'Llama'
    return 'Qwen'
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='aime')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default = 30000)
    parser.add_argument('--overwrite', type=bool, default = False)
    parser.add_argument('--nsamples', type=int, default = 5)
    
    args = parser.parse_args()
    
    model_type = get_model_type(ID_2_MODELS[args.model_id])

    save_path = f"./results_sample/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    # source_path = f"./results/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    save_file = f"{args.data_name}.json"
    
    
    saved_result=[]
    # if os.path.exists(source_path+save_file) and not args.overwrite:
    #     with open(source_path+save_file, 'r') as jf:
    #         saved_result = json.load(jf)
    #         print(f"***************already saved {len(saved_result)} results.***************")
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    
    model = LLM(
        ID_2_MODELS[args.model_id],
        tensor_parallel_size=4,
        enforce_eager=True, 
        gpu_memory_utilization=0.6,
    )
    tok = AutoTokenizer.from_pretrained(
        ID_2_MODELS[args.model_id]
    )

    stop_token_ids = tok("<|im_end|>")["input_ids"]
    # sampling_params = SamplingParams(
    #     max_tokens=args.max_tokens,
    #     min_tokens=0,
    #     # dtype="float32",
    #     stop_token_ids=stop_token_ids,
    #     skip_special_tokens=False,
    #     temperature=args.temperature,
    #     n=args.nsamples
    # )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=0,
        stop_token_ids=tok("<|im_end|>")["input_ids"],
        skip_special_tokens=False,
        temperature=args.temperature,
        logprobs=1,
        n=args.ncandidates,
    )

    dataset = load_my_dataset(args.data_name)

    questions = [d['question'] for d in dataset]
    prompts_no_budget = [get_prompt(q, model_type) for q in questions]

    outputs_no_budget = model.generate(prompts_no_budget, sampling_params=sampling_params)
    results = []
    for d, o in zip(dataset, outputs_no_budget):
        sample_logprobs = []
        sample_texts = []
        for i in range(args.nsamples):
            temp = []
            for lps in o.outputs[i].logprobs:
                temp.append(Tuple([list(lps.keys())[0], list(lps.values())[0].logprob]))
            sample_logprobs.append(temp)
            sample_texts.append(o.outputs[i].text)
        results.append({
            'question': d['question'],
            'solution': d['solution'],
            'sample_texts': sample_texts,
            'sample_logprobs': sample_logprobs,
        })
    with open(save_path + save_file, 'w') as file:
        json.dump(results, file)
        file.flush()
    