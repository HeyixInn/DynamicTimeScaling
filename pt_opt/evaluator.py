import re
import numpy as np
import time
from httpx import HTTPStatusError
from litellm.exceptions import ServiceUnavailableError

import litellm
litellm.bedrock_region_name = "us-east-1"

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

# TODO
def aime_evaluator(llm_outputs, ori_tasks, llm_judge):
    # llm_output:   list of dict {text, logits}
    # ori_tasks:    list of dict {question, solution}
    # llm_judge:    AbstLiteLLM
    def extract_answer(text):
        # 使用正则表达式匹配\boxed{<answer>}的格式
        match = re.search(r'\\boxed\{(.*?)\}', text)
        if match:
            return match.groups()[-1]  # 提取并返回匹配的内容
        else:
            return None
    pred_list, eval_results = [], []
    for llm_output, ori_task in zip(llm_outputs, ori_tasks):
        sol= ori_task['solution']
        answer = extract_answer(llm_output['text'])
        pred_list.append(answer)
        if answer is not None and answer == sol:
            eval_results.append(True)
        else:
            eval_results.append(False)
    res = {
        "score": np.mean(eval_results),
        "pred_list": pred_list,
        "is_correct": eval_results,
    }
    return res

def llm_evaluator(llm_outputs, ori_tasks, llm_judge):
    # llm_output:   list of dict {text, logits}
    # ori_tasks:    list of dict {question, solution}
    # llm_judge:    AbstLiteLLM
    
    prompts = [
        prompt_template.format(
            question=task["question"],
            model_output=output["text"],
            solution=task["solution"]
        )
        for output, task in zip(llm_outputs, ori_tasks)
    ]
    outputs = []
    # for p in prompts:
    #     response = litellm.completion(
    #         model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    #         messages=[
    #             {"role": "user", "content": p}
    #         ],
    #         temperature=0,
    #         top_p=1.0,
    #     )
    #     outputs.append(response.choices[0].message["content"])
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
                
            except ServiceUnavailableError as e:
                time.sleep(60) 
            except Exception as e:
                print(f"Unknown error: {e}")
                raise

    pred_list, eval_results = [], []
    for llm_output, o in zip(llm_outputs, outputs):
        # sol= ori_task['solution']
        answer = llm_output['text']
        pred_list.append(answer)
        # if answer is not None and answer == sol:
        #     eval_results.append(True)
        # else:
        #     eval_results.append(False)
        if "correct" in o.lower():
            eval_results.append(True)
        else:
            eval_results.append(False)
        
    res = {
        "score": np.mean(eval_results),
        "pred_list": pred_list,
        "is_correct": eval_results,
    }
    return res