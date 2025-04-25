import re
import numpy as np

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
        sol= extract_answer(ori_task['solution'])
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

