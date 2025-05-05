from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import json, pickle

from utils import *

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HUGGINGFACE_API_KEY"] = "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"



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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='aime')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_iter', type=int, default = 20)
    parser.add_argument('--num_gpus', type=int, default = 2)
    parser.add_argument('--max_tokens', type=int, default = 32000)
    parser.add_argument('--overwrite', type=bool, default = False)
    
    parser.add_argument('--train_ratio', type=float, default = 0.1)
    
    args = parser.parse_args()
    
    model_type = get_model_type(ID_2_MODELS[args.model_id])

    save_path = f"./results/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    source_path = f"./results_greedy/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    save_file = f"{args.data_name}.json"    
    
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

    train_shuffle = random.Random(42)
    dataset = load_my_dataset(args.data_name)
    num_train = round(len(dataset)*args.train_ratio)
    # train_shuffle.shuffle(dataset)
    
    save_path = f"./uncover_guides/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    saved_result=[]
    if os.path.exists(source_path+save_file) and not args.overwrite:
        with open(source_path+save_file, 'r') as jf:
            saved_result = json.load(jf)
            print(f"***************loading saved {len(saved_result)} non-budget results.***************")
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
        
    combined = list(zip(dataset, saved_result))
    train_shuffle.shuffle(combined)
    dataset, saved_result = zip(*combined)

    dataset = list(dataset)
    saved_result = list(saved_result)
    
    
    dataset = dataset[:num_train]
    questions = [d['question'] for d in dataset]
    solutions = [d['solution'] for d in dataset]
    prompts_no_budget = [get_prompt(q, model_type, tok, enable_thinking=False) for q in questions]
    no_budget_texts = [d['model_output'] for d in saved_result[:num_train]]
        
# ---------------------------------------------------------------------------
    
    
    prompts_thinking = []
    for seed in SEEDs:
        prompts_thinking += [
            # p + "<|im_start|>think" + ans + seed
            get_prompt(p, model_type, tok, enable_thinking=True) + ans + seed
            for p, ans in zip(prompts_no_budget, no_budget_texts)
        ]
    sampling_params_thinking = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=1,
        stop_token_ids=tok("<|im_start|><|im_end|>")["input_ids"],
        skip_special_tokens=False,
        temperature=args.temperature,
    )
    outputs_thinking = model.generate(prompts_thinking, sampling_params=sampling_params_thinking)
    thinking_texts = [o.outputs[0].text for o in outputs_thinking]
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
    outputs = [o.outputs[0].text for o in outputs_final]

    gp = GuidePool()
    for i in range(0, len(prompts_thinking), len(dataset)):
        gp.guide_pool.append(
            GuideNode(
                guide_text=SEEDs[i//len(dataset)],
                eval_results=evaluate(outputs[i:i+len(dataset)], solutions*len(SEEDs))
            )
        )
    test_range = 5
    all_results = []
    for iter in range(args.max_iter):
        # select guidance pair
        pairs = gp.get_guidance_pairs(len(gp.guide_pool)*2)
        
        # round1, prompt thinking with guide pair[0]
        prompts_thinking1 = []
        sampled_idx = []
        for p in pairs:
            for _ in range(test_range):
                idx = random.choice(range(len(no_budget_texts)))
                sampled_idx.append(idx)
                prompts_thinking1.append(prompts_no_budget[idx] + "<|im_start|>think" + no_budget_texts[idx] + p[0].guide)
        
        outputs_thinking = model.generate(prompts_thinking1, sampling_params=sampling_params_thinking)
        thinking_texts = [o.outputs[0].text for o in outputs_thinking]
        
        # round2, prompt thinking with guide pair[1]
        prompts_thinking2 = []
        c=0
        for p in pairs:
            for _ in range(test_range):
                prompts_thinking2.append(prompts_thinking1[c] + thinking_texts[c] + p[1].guide)
                c+=1
        outputs_thinking = model.generate(prompts_thinking2, sampling_params=sampling_params_thinking)
        thinking_texts = [o.outputs[0].text for o in outputs_thinking]
        costs = [len(o.outputs[0].token_ids) for o in outputs_thinking]

        # get final output texts
        prompts_final = [
            p + t
            for p, t in zip(prompts_thinking2, thinking_texts)
        ]
        outputs_final = model.generate(prompts_final, sampling_params=sampling_params_final)
        budget_texts = [p + o.outputs[0].text for p, o in zip(prompts_final, outputs_final)]
        outputs = [o.outputs[0].text for o in outputs_final]
        
        # evaluate outputs
        eval_outputs = evaluate(outputs, [solutions[idx] for idx in sampled_idx])
        # calcualte acc for each pair
        avg_acc = []
        avg_cost = []
        for i in range(0, len(eval_outputs), test_range):
            avg_acc.append(sum(eval_outputs[i:i+test_range])/test_range)
            avg_cost.append(sum(costs[i:i+test_range])/test_range)
        assert len(avg_acc)==len(pairs)
        assert len(avg_cost)==len(pairs)
        
        # construct edges
        for i, p in enumerate(pairs):
            if p[0].get_base_score(sampled_idx[i*test_range:(i+1)*test_range])<=avg_acc[i]:
                p[0].update_child(p[1], avg_acc[i], avg_cost[i])
        
        # uncover new guide from failure case
        fail_indices = [i for i, val in enumerate(eval_outputs) if val == 0]
        fail_idx = random.choice(fail_indices)
        dataset_id = sampled_idx[fail_idx]
        txt = budget_texts[fail_idx]
        new_guide_text = uncover_new_guide(model, tok, dataset[dataset_id]['question'], dataset[dataset_id]['solution'], txt)
        while new_guide_text is None:
            fail_indices = [i for i, val in enumerate(eval_outputs) if val == 0]
            fail_idx = random.choice(fail_indices)
            dataset_id = sampled_idx[fail_idx]
            txt = budget_texts[fail_idx]
            new_guide_text = uncover_new_guide(model, tok, dataset[dataset_id]['question'], dataset[dataset_id]['solution'], txt)
        prompts_thinking = [
            p + "<|im_start|>think" + ans + new_guide_text
            for p, ans in zip(prompts_no_budget, no_budget_texts)
        ]
        outputs_thinking = model.generate(prompts_thinking, sampling_params=sampling_params_thinking)
        thinking_texts = [o.outputs[0].text for o in outputs_thinking]
        
        prompts_final = [
            p + t
            for p, t in zip(prompts_thinking2, thinking_texts)
        ]
        outputs_final = model.generate(prompts_final, sampling_params=sampling_params_final)
        outputs = [o.outputs[0].text for o in outputs_final]
        eval_new_guide = evaluate(outputs, solutions)
        
        for gn in gp.guide_pool:
            gn.update_reward()
        gp.guide_pool.append(GuideNode(new_guide_text, eval_new_guide))
        gp.dieout()
            
        results = {}
        for g in gp.guide_pool:
            results.update({
                g.guide: g.reward
            })
        all_results.append({
            iter: results
        })
        print(all_results)
        with open(save_path + save_file, 'w') as file:
            json.dump(all_results, file, indent=4)
            file.flush()
        graph = []
        for node in gp.guide_pool:
            graph.append({"guide": node.guide, "children": [{c.guide: c.reward} for c in node.children]})
        with open(save_path + save_file.replace(".json", "_graph.json"), 'w') as file:
            json.dump(graph, file, indent=4)
            file.flush()
        