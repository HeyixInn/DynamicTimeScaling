from datasets import load_dataset

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
                aime.append(d)
        return aime
    elif data_name=='omni':
        aime = []
        for d in ds:
            if 'KbsdJames/Omni-MATH' in d['source_type']:
                aime.append(d)
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