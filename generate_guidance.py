from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
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

    save_path = f"./results_ground/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    source_path = f"./results/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    save_file = f"{args.data_name}.json"
    
    ignore_tokens=[
        "Wait, let’s double-check what the question is really asking.",
        "Wait, let’s slow down and make sure each step logically follows from the last.",
        "Wait, let’s check the math — are all signs, parentheses, and numbers in the right place?",
        "Wait, let’s revisit the rule or formula — is this really the right one for this situation?"
    ]
    
    saved_result=[]
    if os.path.exists(source_path+save_file) and not args.overwrite:
        with open(source_path+save_file, 'r') as jf:
            saved_result = json.load(jf)
            print(f"***************already saved {len(saved_result)} results.***************")
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    
    model = LLM(
        ID_2_MODELS[args.model_id],
        tensor_parallel_size=2,
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

    # For the exact raspberry sample in the paper, change
    # model to `qfq/1k_qr_bt_dm_po_steps` (an earlier version of s1)
    # prompt to `How many r in raspberry?`
    # prompts = [
    #     "Can you list the most common 4 types of mistakes one could probably make while reasoning a math problem, and then generate 4 prompts in the format of \"Wait, let's ...\" corresponding to the 4 mistakes, to guide someone to examine the reasoning process? Put your final answer in  <ans></ans> tags.\n\
    #     For example, <ans1>Wait, let’s double-check what the question is really asking.</ans1>",
    # ]
    prompt_template = """You are an expert at analyzing model mistakes. You will be given a question, the correct gold solution, and a model’s incorrect output. Your task is to:

    1. Compare the gold solution and model output to identify key differences.
    2. Analyze the root cause of the model's mistake.
    3. Suggest a general guide, starting with “Wait, let’s ...”, to help the model avoid this kind of mistake in the future.

    Format your output strictly in the following structure:

    [DIFFERENCE]
    <your analysis of the difference>

    [REASON]
    <your explanation of the likely cause of the mistake>

    [GUIDE]
    Wait, let's <your guide here>.

    ---

    QUESTION:
    {{question}}

    GOLD SOLUTION:
    {{gold_solution}}

    MODEL OUTPUT:
    {{model_output}}
    """
    d = json.load(open("./_tmp.json", 'r'))
    filled_prompt = prompt_template.format(
        question=d['question'],
        gold_solution=d['solution'],
        model_output=d['model_output']
    )
    
    # Step 1: 过滤数据
    # ds = load_dataset("simplescaling/s1K-1.1")['train']
    # aime = [d for d in ds if 'qq8933/AIME_1983_2024' in d['source_type']]
    # dataset = aime  # 只处理未完成的部分

    # # Step 2: 构造第一轮 prompt（no_budget）
    # questions = [d['question'] for d in dataset]
    # prompts_no_budget = [get_prompt(q, model_type) for q in questions]
    # no_budget_texts = [d['model_output']['no_budget_ans'] for d in saved_result]

    # # Step 3: 一次性调用 model.generate
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=0,
        stop_token_ids=tok("<|im_end|>")["input_ids"],
        skip_special_tokens=False,
        temperature=0.7,
    )

    output = model.generate([filled_prompt], sampling_params=sampling_params)
    # no_budget_texts = [o.outputs[0].text for o in outputs_no_budget]
    # print(no_budget_texts)
    import re
    match = re.search(r"\[GUIDE\]\n(Wait, let[’']s[\s\S]+?)(?=\n\[|$)", output[0].outputs[0].text)
    guide = match.group(1) if match else "Guide not found"
    print(guide)
    with open("./output.txt", 'w') as file:
        file.writelines(output[0].outputs[0].text)
    # Step 4: 构造第二轮 prompt（with budget forcing）
    # thinking_texts = []
    # for ignore_token in ignore_tokens:
    #     prompts_thinking = [
    #         p + "<|im_start|>think" + ans + "Wait"
    #         for p, ans in zip(prompts_no_budget, no_budget_texts)
    #     ]

    #     sampling_params_thinking = SamplingParams(
    #         max_tokens=args.max_tokens//len(ignore_tokens),
    #         min_tokens=1,
    #         stop_token_ids=tok("<|im_start|><|im_end|>")["input_ids"],
    #         skip_special_tokens=False,
    #         temperature=0.0,
    #     )
    #     outputs_thinking = model.generate(prompts_thinking, sampling_params=sampling_params_thinking)
    #     if len(thinking_texts)==0:
    #         thinking_texts = [o.outputs[0].text for o in outputs_thinking]
    #     else:
    #         thinking_texts = [old_text+'\n'+o.outputs[0].text for o, old_text in zip(outputs_thinking, thinking_texts)]

    # # Step 5: 构造最终 prompt，加入 budget forcing 后再生成最终答案
    # prompts_final = [
    #     p + t
    #     for p, t in zip(prompts_thinking, thinking_texts)
    # ]
    # sampling_params_final = SamplingParams(
    #     max_tokens=args.max_tokens,
    #     min_tokens=0,
    #     stop_token_ids=tok("<|im_end|>")["input_ids"],
    #     skip_special_tokens=False,
    #     temperature=args.temperature,
    # )
    # outputs_final = model.generate(prompts_final, sampling_params=sampling_params_final)
    # budget_texts = [p + o.outputs[0].text for p, o in zip(prompts_final, outputs_final)]

    # # Step 6: 汇总结果并保存
    # results = []
    # for i, d in enumerate(dataset):
    #     # print("No budget forcing:\n", prompts_no_budget[i] + no_budget_texts[i])
    #     # print("-" * 80)
    #     # print("With budget forcing:\n", budget_texts[i])
    #     # print("=" * 80)

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