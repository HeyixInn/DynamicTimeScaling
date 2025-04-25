import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from generate import load_my_dataset
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HUGGINGFACE_API_KEY"] = "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"

dataset = "omni"

import litellm

# 设置 Bedrock 区域（推荐）
litellm.bedrock_region_name = "us-east-1"


with open(f"results_dy*/{model}/{dataset}.json", "r") as f:
    model_outputs = json.load(f)

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

results = []

# for i, r in enumerate(model_outputs):
#     question = r["question"]
#     solution = r["solution"]
#     no_budget_ans = r["model_output"]["no_budget_ans"]
#     budget_ans = r["model_output"]["budget_ans"]


prompts = [
    prompt_template.format(
        question=r["question"],
        model_output=r["model_output"],
        solution=r["solution"]
    )
    for r in model_outputs
]

outputs = []
for p in prompts:
    
    response = litellm.completion(
        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[
            {"role": "user", "content": p}
        ],
        temperature=0,
        top_p=1.0,
    )

    outputs.append(response.choices[0].message["content"])
r1 = [("Correct" in o.strip()) for o in outputs]

correct_count1 = sum(r1)


prompts = [
    prompt_template.format(
        question=r["question"],
        model_output=r["model_output"]["budget_ans"],
        solution=r["solution"]
    )
    for r in model_outputs
]

outputs = []
for p in prompts:
    
    response = litellm.completion(
        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[
            {"role": "user", "content": p}
        ],
        temperature=0,
        top_p=1.0,
    )

    outputs.append(response.choices[0].message["content"])
r2 = [("Correct" in o.strip()) for o in outputs]

correct_count2 = sum(r2)
print(len(r1), "\nno_budget_acc: ", correct_count1/len(r1), "\nbudget_acc: ", correct_count2/len(r2))

for i, o in enumerate(model_outputs):
    o.update({
        "eval":{
            'no_budget_ans': r1[i],
            'budget_and': r2[i],
        }
    })
import os
if not os.path.exists(f"eval/{model}"):
    os.makedirs(f"eval/{model}")
with open(f"eval/{model}/{dataset}.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
