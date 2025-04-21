import re
import random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from math import log10

SEEDs = [        
    "Wait, let\'s make sure we\'re using the correct formula for the problem.",
    "Wait, did we convert all units to the same measurement before starting?",
    "Wait, let\'s double-check the calculations to avoid any arithmetic mistakes.",
    "Wait, let\'s verify that we\'ve correctly applied the signs throughout the problem."
]

prompt_template_for_new_guide = """You are an expert at analyzing model mistakes. You will be given a question, the correct gold solution, and a model’s incorrect output. Your task is to:

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

def evaluate(outputs, solutions):
    eval_results = []
    def extract_answer(text):
        # 使用正则表达式匹配\boxed{<answer>}的格式
        match = re.search(r'\\boxed\{(.*?)\}', text)
        if match:
            return match.groups()[-1]  # 提取并返回匹配的内容
    for o, sol in zip(outputs, solutions):
        answer = extract_answer(o)
        if answer is not None and answer==sol:
            eval_results.append(True)
        else:
            eval_results.append(False)
    return eval_results

def uncover_new_guide(model, tok, question, solution, model_output):

    prompt = prompt_template_for_new_guide.format(
        question=question,
        gold_solution=solution,
        model_output=model_output
    )
    sampling_params = SamplingParams(
        max_tokens=30000,
        min_tokens=0,
        stop_token_ids=tok("<|im_end|>")["input_ids"],
        skip_special_tokens=False,
        temperature=0.7,
    )
    output = model.generate([prompt], sampling_params=sampling_params)
    
    match = re.search(r"\[GUIDE\]\n(Wait, let[’']s[\s\S]+?)(?=\n\[|$)", output[0].outputs[0].text)
    guide = match.group(1) if match else None
    return guide

class GuidPool():
    def __init__(self, max_size=10):
        self.guide_pool = []     
        self.max_size = max_size   
        
    def get_guidance_pairs(self, num_pairs=10):
        probs = [guide.reward for guide in self.guide_pool]
        pairs = [random.choices(self.guide_pool, weights=probs, k=2) for _ in range(num_pairs)]
        flip_pairs = [[p[1], p[0]] for p in pairs]
        return pairs + flip_pairs
    def dieout(self):
        if len(self.guide_pool)<self.max_size:
            return
        while len(self.guid_pool)>=self.max_size:
            min_guide = min(self.guid_pool, key=lambda x: x.reward)
            self.guid_pool.remove(min_guide)

class GuideNode():
    def __init__(self, guide_text, eval_results, max_token=32000):
        self.guide = guide_text
        self.eval_results = eval_results
        self.children = {}
        self.reward = max(0.01, sum(eval_results)/len(eval_results))
        self.max_token=max_token
        
        # coefficients
        self.alpha = 1
        self.beta = 0.1
        
        self.lamda = 0.8
    
    def get_base_score(self, sampled_idx):
        print(self.eval_results)
        print(sampled_idx)
        eval_samples = [self.eval_results[idx] for idx in sampled_idx]
        return sum(eval_samples)/len(eval_samples)
        
    def update_child(self, child, acc, cost):
        if child in self.children.keys():
            self.children[child] *= self.lamda
            self.children[child] += ((1-self.lamda)*(acc*self.alpha+log10(self.max_token/cost)*self.beta))
        else:
            self.children[child] = (acc*self.alpha+log10(self.max_token/cost)*self.beta)
        print(self.children[child], acc, cost)
    
    def update_reward(self):
        self.reward = max(self.reward, sum([v for v in self.children.values()])/len(self.children))