import re
import random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import litellm
from math import log10
from datasets import load_dataset, Dataset
from litellm_abst import AbstLiteLLM

SEEDs = [        
    "Wait, let\'s make sure we\'re using the correct formula for the problem.",
    "Wait, did we convert all units to the same measurement before starting?",
    "Wait, let\'s double-check the calculations to avoid any arithmetic mistakes.",
    "Wait, let\'s verify that we\'ve correctly applied the signs throughout the problem."
]

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

NAME_2_DATASET = {
    'aime': "AI-MO/aimo-validation-aime",
    'gpqa': "Idavidrein/gpqa",
    'openaimath': "simplescaling/openaimath",
    
}
DATASET_2_CONFIG = {
    "AI-MO/aimo-validation-aime": {"split": "train", "question_field": "problem"},
    "Idavidrein/gpqa": {"split": "train", "question_field": "Question"},
    "simplescaling/openaimath": {"split": "test", "question_field": "problem"},
    # "livecodebench/code_generation_lite": {"split": "test", "question_field": "question_content", "version_tag": "release_v4"},
}
DS_COLUMNS = {"question", "solution", "cot_type", "source_type", "metadata"}
def load_generic(name, split, question_field="question", solution_field="solution", cot_type="math", version_tag=None):
    conf = "gpqa_diamond" if name == "Idavidrein/gpqa" else None
    ds = load_dataset(name, conf, version_tag=version_tag, trust_remote_code=True)[split]
    # Make metadata a string that can be loaded via literal_eval to avoid TypeError: Couldn't cast array of type list<item: string> to null 
    ds = ds.map(lambda x: {"question": x.pop(question_field), "solution": x.pop(solution_field, None), "cot_type": cot_type, "source_type": name, "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds
def load_my_dataset(data_name):
    name = NAME_2_DATASET[data_name]
    config = DATASET_2_CONFIG[name]

    ds = load_generic(name, **config)
    # ds  = load_dataset("simplescaling/s1K-1.1")['train']
    return [d for d in ds]
    # if data_name=='aime':
    #     aime = load_dataset("qq8933/AIME_1983_2024")['train']
    #     aime_dataset = []
    #     for example in aime:
    #         result = process_example(example)
    #         if result is not None:
    #             aime_dataset.append(result)
    #     # aime_dataset = Dataset.from_list(aime_dataset)
    #     return aime_dataset
    # elif data_name=='omni':
    #     aime = []
    #     for d in ds:
    #         if 'KbsdJames/Omni-MATH' in d['source_type']:
    #             aime.append(d)
    #     return aime
    


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
    return guide.split('\n')[0]

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
        if len(self.children)==0:
            return
        self.reward = max(self.reward, sum([v for v in self.children.values()])/len(self.children))


def parse_vllm_output(model_pred):

    def extract_token_prob(d_list):
        res = []
        for d in d_list:
            tmp = list(d.values())[0]
            res.append([tmp.decoded_token, tmp.logprob])
        return res

    text = model_pred.outputs[0].text
    logits = extract_token_prob(model_pred.outputs[0].logprobs)
    return {"text": text, "logits": logits}


def parse_litellm_output(model_pred):
    def extract_token_prob(prob_list):
        res = []
        for d in prob_list:
            res.append([d.token, d.logprob])
        return res

    num_of_gen = len(model_pred.choices)
    assert num_of_gen == 1

    text = model_pred.choices[0].message.content
    if hasattr(model_pred.choices[0], 'logprobs'):
        logits = extract_token_prob(model_pred.choices[0].logprobs.content)
    else:
        logits = None
    return {"text": text, "logits": logits}


def load_agent():
    model_name = "anthropic.claude-3-5-haiku-20241022-v1:0"
    # model_name = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    provider = "bedrock"

    agent = AbstLiteLLM(provider, model_name)
    return agent