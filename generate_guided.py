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
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default = 30000)
    parser.add_argument('--overwrite', type=bool, default = False)
    
    args = parser.parse_args()
    
    model_type = get_model_type(ID_2_MODELS[args.model_id])

    save_path = f"./results_guided**/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    source_path = f"./results_greedy**/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
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
        max_seq_len_to_capture=args.max_tokens,
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
    prompts_no_budget = [get_prompt(q, model_type, tokenizer=tok, enable_thinking=True) for q in questions]
    no_budget_texts = [d['model_output'] for d in saved_result]
    # no_budget_texts = [d['sample_texts'][0] for d in saved_result]
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=0,
        stop_token_ids=tok("<|im_end|>")["input_ids"],
        skip_special_tokens=False,
        temperature=args.temperature,
    )

    # outputs_no_budget = model.generate(prompts_no_budget, sampling_params=sampling_params)
    # no_budget_texts = [o.outputs[0].text for o in outputs_no_budget]
    
    
    
    
    with open(f"./uncover_guides**/{ID_2_MODELS[args.model_id].split('/')[-1]}/{args.data_name}_graph.json", 'r') as file:
        guide_graph_list = json.load(file)
        guide_graph = {}
        for d in guide_graph_list:
            guide_graph.update({d['guide']: d['children']})
    with open(f"./uncover_guides**/{ID_2_MODELS[args.model_id].split('/')[-1]}/{args.data_name}.json", 'r') as file:
        guide_pool = json.load(file)
        guide_pool = list(guide_pool[-1].values())[-1]
    
    sorted_guide_pool = sorted(guide_pool.items(), key=lambda item: item[1], reverse=True)
    ignore_token = sorted_guide_pool[0][0]
    
    thinking_texts = []
    if model_type=="Qwen3":
        prompts_thinking = [
            # p + "<|im_start|>think" + ans + ignore_token
            get_prompt(p, model_type, tokenizer=tok, enable_thinking=True) + ans + ignore_token
            for p, ans in zip(prompts_no_budget, no_budget_texts)
        ]
    else:
        prompts_thinking = [
            p + "<|im_start|>think" + ans + ignore_token
            # get_prompt(p, model_type, tokenizer=tok, enable_thinking=True) + ans + ignore_token
            for p, ans in zip(prompts_no_budget, no_budget_texts)
        ]
    budget = args.max_tokens
    budget_cuts = [len(tok(t)['input_ids']) for t in no_budget_texts]
    budget-=max(budget_cuts)
    
    used_guide = []
    max_iter = 10
    while budget>0 and max_iter>0:
        max_iter -=1
        used_guide.append(ignore_token)
        sampling_params_thinking = SamplingParams(
            max_tokens=budget,
            min_tokens=1,
            stop_token_ids=tok("<|im_start|><|im_end|>")["input_ids"],
            skip_special_tokens=False,
            temperature=args.temperature,
        )
        outputs_thinking = model.generate(prompts_thinking, sampling_params=sampling_params_thinking)
        thinking_texts = [o.outputs[0].text for o in outputs_thinking]
        
        budget_cuts = [len(o.outputs[0].token_ids) for o in outputs_thinking]
        budget-=max(budget_cuts)

        # prompts_thinking = [
        #     p + ignore_token
        #     for p in thinking_texts
        # ]
        prompts_thinking = [
            p + t + ignore_token
            for p, t in zip(prompts_thinking, thinking_texts)
        ]
        
        guide_pool = {}
        for g in guide_graph[ignore_token]:
            guide_pool.update(g)
        sorted_guide_pool = sorted(guide_pool.items(), key=lambda item: item[1], reverse=True)
        g=0
        while g<len(sorted_guide_pool) and sorted_guide_pool[g][0] in used_guide:
            g+=1
        if g>=len(sorted_guide_pool) or sorted_guide_pool[g][1]<0.2:
            break
        ignore_token = sorted_guide_pool[g][0]
        # used_guide.append(ignore_token)
    
    prompts_final = prompts_thinking
    
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
            'model_output': budget_texts[i],
            'used_guide': used_guide,
            'num_guide': len(used_guide)
        })

    with open(save_path + save_file, 'w') as file:
        json.dump(results, file)
        file.flush()