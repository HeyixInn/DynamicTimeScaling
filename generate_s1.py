from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import json
from utils import *

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["HUGGINGFACE_API_KEY"] = "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"

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='aime')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default = 32000)
    parser.add_argument('--overwrite', type=bool, default = False)
    
    args = parser.parse_args()
    
    model_type = get_model_type(ID_2_MODELS[args.model_id])

    save_path = f"./results_s1/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    source_path = f"./results_greedy/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    save_file = f"{args.data_name}.json"

    saved_result=[]
    if os.path.exists(source_path+save_file) and not args.overwrite:
        with open(source_path+save_file, 'r') as jf:
            saved_result = json.load(jf)
            print(f"***************already saved {len(saved_result)} results.***************")
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    
    model = LLM(
        ID_2_MODELS[args.model_id],
        tensor_parallel_size=1,
        enforce_eager=True, 
        gpu_memory_utilization=0.95,
    )
    tok = AutoTokenizer.from_pretrained(
        ID_2_MODELS[args.model_id]
    )

    stop_token_ids = tok("<|im_end|>")["input_ids"]
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=args.temperature
    )

    dataset = load_my_dataset(args.data_name)

    questions = [d['question'] for d in dataset]
    prompts_no_budget = [get_prompt(q, model_type) for q in questions]
    no_budget_texts = [d['model_output'] for d in saved_result]

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=0,
        stop_token_ids=tok("<|im_end|>")["input_ids"],
        skip_special_tokens=False,
        temperature=args.temperature,
    )

    # outputs_no_budget = model.generate(prompts_no_budget, sampling_params=sampling_params)
    # no_budget_texts = [o.outputs[0].text for o in outputs_no_budget]
    
    
    thinking_texts = []
    ignore_token = "Wait"
    prompts_thinking = [
        p + "<|im_start|>think" + ans + ignore_token
        for p, ans in zip(prompts_no_budget, no_budget_texts)
    ]
    budget = args.max_tokens
    
    while budget>0:
        sampling_params_thinking = SamplingParams(
            max_tokens=budget,
            min_tokens=1,
            stop_token_ids=tok("<|im_start|><|im_end|>")["input_ids"],
            skip_special_tokens=False,
            temperature=0.0,
        )
        outputs_thinking = model.generate(prompts_thinking, sampling_params=sampling_params_thinking)
        thinking_texts = [o.outputs[0].text for o in outputs_thinking]
        
        budget_cuts = [len(o.outputs[0].token_ids) for o in outputs_thinking]
        budget-=max(budget_cuts)

        prompts_thinking = [
            p + ignore_token
            for p in prompts_thinking
        ]
    prompts_final = [
        p + t
        for p, t in zip(prompts_thinking, thinking_texts)
    ]
    sampling_params_final = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=0,
        stop_token_ids=tok("<|im_end|>")["input_ids"],
        skip_special_tokens=False,
        temperature=args.temperature,
    )
    outputs_final = model.generate(prompts_final, sampling_params=sampling_params_final)
    budget_texts = [p + o.outputs[0].text for p, o in zip(prompts_final, outputs_final)]

    # Step 6: 汇总结果并保存
    results = []
    for i, d in enumerate(dataset):
        results.append({
            'question': d['question'],
            'solution': d['solution'],
            'model_output': no_budget_texts[i]
        })

    with open(save_path + save_file, 'w') as file:
        json.dump(results, file)
        file.flush()