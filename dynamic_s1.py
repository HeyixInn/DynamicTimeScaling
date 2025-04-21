from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import json
import re
import random
from copy import deepcopy

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
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
                aime.append(d['question'])
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


# SEED = [
#     "What if",
#     "Let us reconsider",
#     "Does it follow that",
#     "Conversely",
#     "Is the converse true",
#     "Can we generalize this",
#     "Under what conditions would this fail?",
#     "Suppose the opposite",
#     "This suggests that",
#     "To verify this, let's consider",
# ]
SEED = [
    'Wait', 'So', 'But', 'Though', 
]
def extract_answer(text):
    # 使用正则表达式匹配\boxed{<answer>}的格式
    match = re.search(r'\\boxed\{(.*?)\}', text)
    if match:
        return match.groups()[-1]  # 提取并返回匹配的内容
    return None

def think_and_answer(seed, model, prompts_no_budget, no_budget_texts):
    prompts_thinking = [
        p + "<|im_start|>think" + ans + seed
        for p, ans in zip(prompts_no_budget, no_budget_texts)
    ]
    sampled_idx = random.choices(range(50), k=20)
    prompts_thinking = [
        prompts_no_budget[i] + "<|im_start|>think" + no_budget_texts[i] + seed
        for i in sampled_idx
    ]

    sampling_params_thinking = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=1,
        stop_token_ids=tok("<|im_start|><|im_end|>")["input_ids"],
        skip_special_tokens=False,
        temperature=0.5,
    )
    outputs_thinking = model.generate(prompts_thinking, sampling_params=sampling_params_thinking)
    thinking_texts = [o.outputs[0].text for o in outputs_thinking]

    # Step 5: 构造最终 prompt，加入 budget forcing 后再生成最终答案
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
    return budget_texts

def evaluate_seeds(answers, new_seeds, model, prompts_no_budget, no_budget_texts):
    results = {}
    for seed in new_seeds:
        budget_texts = think_and_answer(seed, model, prompts_no_budget, no_budget_texts)
        eval_results = []
        for o, a in zip(budget_texts, answers):
            final_ans = extract_answer(o)
            eval_results.append(final_ans is not None and final_ans==a)
        results[seed] = sum(eval_results)
    return results

def sampling_mutation(selected_seed, model, prompts_no_budget):
    prompts_thinking = [
        p + "<|im_start|>think" + ans + selected_seed
        for p, ans in zip(prompts_no_budget, no_budget_texts)
    ]
    prompt = random.choice(prompts_thinking)
    # print(prompt)
    num_mutation = 5
    sampling_params_thinking = SamplingParams(
        max_tokens=3,
        min_tokens=1,
        stop_token_ids=tok("<|im_start|><|im_end|>")["input_ids"],
        skip_special_tokens=False,
        temperature=1.0,
        n=num_mutation,
        top_p=0.8,
        top_k=30
    )
    output = model.generate(prompt, sampling_params=sampling_params_thinking)
    # sampling_texts = [selected_seed+o.outputs[0].text for o in outputs_thinking]
    sampling_texts = [selected_seed+output[0].outputs[i].text for i in range(num_mutation)]
    return sampling_texts    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='aime')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_iter', type=int, default = 5)
    parser.add_argument('--num_gpus', type=int, default = 2)
    parser.add_argument('--max_tokens', type=int, default = 32000)
    parser.add_argument('--overwrite', type=bool, default = False)
    
    args = parser.parse_args()
    
    model_type = get_model_type(ID_2_MODELS[args.model_id])

    save_path = f"./results/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    save_file = f"{args.data_name}.json"
    
    saved_result=[]
    if os.path.exists(save_path+save_file) and not args.overwrite:
        with open(save_path+save_file, 'r') as jf:
            saved_result = json.load(jf)
            print(f"***************already saved {len(saved_result)} results.***************")
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    
    model = LLM(
        ID_2_MODELS[args.model_id],
        tensor_parallel_size=4,
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

    ds = load_dataset("simplescaling/s1K-1.1")['train']
    aime = [d for d in ds if 'qq8933/AIME_1983_2024' in d['source_type']]
    dataset = aime[:50]  # 只处理未完成的部分

    questions = [d['question'] for d in dataset]
    prompts_no_budget = [get_prompt(q, model_type) for q in questions]
    no_budget_texts = [d['model_output']['no_budget_ans'] for d in saved_result[:50]]

    # sampling_params = SamplingParams(
    #     max_tokens=args.max_tokens,
    #     min_tokens=0,
    #     stop_token_ids=tok("<|im_end|>")["input_ids"],
    #     skip_special_tokens=False,
    #     temperature=args.temperature,
    # )

    # outputs_no_budget = model.generate(prompts_no_budget, sampling_params=sampling_params)
    # no_budget_texts = [o.outputs[0].text for o in outputs_no_budget]
    
    answers = [d['solution'] for d in aime]
    # print(prompts_no_budget)
    # eval_seeds = evaluate_seeds(answers, SEED, model, prompts_no_budget, no_budget_texts)
    # with open('./initial_seeds.json', 'r') as file:
    #     initial_seeds = json.load(file)
    initial_seeds = evaluate_seeds(answers, SEED, model, prompts_no_budget, no_budget_texts)
    # collected_seeds = {}
    collected_eval = {}
    eval_seeds = initial_seeds
    for i in range(10):
        # selected_seed = max(eval_seeds, key=eval_seeds.get)
        selected_seed = random.choice(list(eval_seeds.keys()))
        new_seeds = sampling_mutation(selected_seed, model, prompts_no_budget)
        # print(new_seeds)
        # print(len(new_seeds))
        eval_seeds.update(evaluate_seeds(answers, new_seeds, model, prompts_no_budget, no_budget_texts))
        # for _ in range(len(eval_seeds)-len(SEED)):
        #     del eval_seeds[min(eval_seeds, key=eval_seeds.get)]
        eval_seeds.update(initial_seeds)
        # print(eval_seeds)
        # collected_seeds[i]=new_seeds
        collected_eval[i]=deepcopy(eval_seeds)
        # with open('./new_seeds.json', 'w') as file:
        #     json.dump(collected_seeds, file, indent=4)
        #     file.flush()
        with open('./eval_seeds.json', 'w') as file:
            json.dump(collected_eval, file, indent=4)
            file.flush()
    
    # Step 6: 汇总结果并保存
    # results = []
    # for i, d in enumerate(dataset):
    #     results.append({
    #         'question': d['question'],
    #         'solution': d['solution'],
    #         'model_output': {
    #             'no_budget_ans': no_budget_texts[i],
    #             'budget_ans': budget_texts[i]
    #         }
    #     })

    # with open(save_path + save_file, 'w') as file:
    #     json.dump(results, file)
    #     file.flush()