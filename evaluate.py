from utils import *
import json
import argparse
import os

prompt_template = """You are a knowledgeable and strict teaching assistant. Given a question, a model-generated answer (model_output), and the correct reference answer (solution), determine whether the model's answer is correct.

### Question:
{question}

### Model Output:
{model_output}

### Reference Solution:
{solution}

Please respond with only one word: “Correct” or “Wrong”. Do not include any other text.

Let's think step by step
"""

def llm_eval(outputs, solutions, questions):
    # llm_output:   list of dict {text, logits}
    # ori_tasks:    list of dict {question, solution}
    # llm_judge:    AbstLiteLLM
    
    prompts = [
        prompt_template.format(
            question=outputs[i],
            model_output=solutions[i],
            solution=questions[i]
        )
        for i in range(len(outputs))
    ]
    outputs = []
    for p in prompts:
        while True:
            try:
                response = litellm.completion(
                    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                    messages=[
                        {"role": "user", "content": p}
                    ],
                    temperature=0,
                    top_p=1.0,
                )
                outputs.append(response.choices[0].message["content"])
                break  
            except HTTPStatusError as e:
                if e.response.status_code == 503:
                    print("503 error, retrying ...")
                    time.sleep(3)  
                else:
                    raise  
            except Exception as e:
                print(f"Unknown error: {e}")
                raise
    eval_results = []
    for o in outputs:
        if "correct" in o.lower():
            eval_results.append(True)
        else:
            eval_results.append(False)
    return [
        {"llm_eval_output": o, "eval": e}
        for o, e in zip(outputs, eval_results)
    ]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="greedy")
    parser.add_argument("--model_id", type=int, default=2)
    parser.add_argument("--data_name", type=str, default="aime")
    
    args = parser.parse_args()
    
    # source_path = f"./results_greedy/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    # save_file = f"{args.data_name}.json"
    # with open(source_path+save_file, 'r') as file:   
    #     saved_results = json.load(file)
    
    if args.mode=="greedy":
        source_path = f"./results_greedy/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
        save_file = f"{args.data_name}.json"
        with open(source_path+save_file, 'r') as file:   
            saved_results = json.load(file)
        output_texts = [r['model_output'] for r in saved_results]
        solutions = [r['solution'] for r in saved_results]
        questions = [r['question'] for r in saved_results]
        if args.data_name=="aime":
            eval_results = evaluate(output_texts, solutions)
            for r, e in zip(saved_results, eval_results):
                r.update({'eval': e})
        else:
            eval_results = llm_eval(outputs=output_texts, solutions=solutions, questions=questions)
            for r, e in zip(saved_results, eval_results):
                r.update({'llm_eval_output': e['llm_eval_output'], 'eval': e['eval']})
    
    elif args.mode=="sampling":
        source_path = f"./results_sample/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
        save_file = f"{args.data_name}.json"
        with open(source_path+save_file, 'r') as file:   
            saved_results = json.load(file)
        sample_num = len(saved_results[0]['sample_texts'])
        output_texts = [s  for r in saved_results for s in r['sample_texts']]
        solutions = [r['solution'] for r in saved_results for _ in range(sample_num)]
        questions = [r['question'] for r in saved_results for _ in range(sample_num)]
        
        if args.data_name=="aime":
            eval_results = evaluate(output_texts, solutions)
            sample_eval = [eval_results[i:i+sample_num] for i in range(0, len(eval_results), sample_num)]
            for r, e in zip(saved_results, sample_eval):
                r.update({'eval': e})
        else:
            eval_results = llm_eval(outputs=output_texts, solutions=solutions, questions=questions)
            sample_llm_eval_output = [[e['llm_eval_output'] for e in eval_results[i:i+sample_num]] for i in range(0, len(eval_results), sample_num)]
            sample_eval = [[e['eval'] for e in eval_results[i:i+sample_num]] for i in range(0, len(eval_results), sample_num)]
            for r, e in zip(saved_results, sample_llm_eval_output):
                r.update({'llm_eval_output': e})
            for r, e in zip(saved_results, sample_eval):
                r.update({'eval': e})

    if not os.path.exists(source_path+"eval/"):
        os.makedirs(source_path+"eval/")
    with open(source_path+"eval/"+save_file, 'w') as file:
        json.dump(saved_results, file, indent=4)
        