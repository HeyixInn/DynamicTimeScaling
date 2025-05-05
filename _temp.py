from vllm import LLM, SamplingParams

# 模型名称
model_name = "Qwen/Qwen3-8B"

# 创建 vLLM 模型实例
llm = LLM(
    model=model_name,
    dtype="auto",  # 自动推断数据类型
    trust_remote_code=True  # 对 HuggingFace 上的自定义代码进行信任
)

# Prompt 内容
prompt = "Give me a short introduction to large language model."

# 构造 Chat 模板（与 transformers 中的 apply_chat_template 类似）
messages = [
    {"role": "user", "content": prompt + "\no_think"}
]
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # 仅 Qwen 模型支持
)

# 设置采样参数
sampling_params = SamplingParams(
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9
)

# 进行推理
outputs = llm.generate([text], sampling_params)

# 输出生成结果
for output in outputs:
    generated_text = output.outputs[0].text
    print("Generated output:")
    print(generated_text)
