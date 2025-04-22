from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import pandas as pd
import numpy as np
from typing import List, Dict

import os
import argparse
from tqdm import tqdm

from utils import *
from reward_model import load_reward_model
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def evaluate_batch_responses(data, sampling_params, n_samples, 
                             tokenizer, llm: LLM, compute_similarity, reward_model):
    results = []

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": d['question']}], add_generation_prompt=True
        ) for d in data
    ]


    # ===== 1. vLLM 批量生成，取 logprob =====
    outputs = llm.generate(prompts, sampling_params)
    outputs = [[o.outputs[i] for i in range(n_samples)] for o in outputs]
    responses = [[o.outputs[i].text for i in range(n_samples)] for o in outputs]
    max_tokens_list = [[len(o.outputs[i].token_ids) for i in range(n_samples)] for o in outputs]

    for i, output in enumerate(outputs):
        logprobs = [o.logprobs[:max_tokens_list[i][j]] for j, o in enumerate(output)]
        logprob_sum = [sum(lp) for lp in logprobs]

        # ===== 2. similarity matrix =====
        similarity_matrix = compute_similarity.compute_score_matrix(responses[i], data[i]["question"])
        # ===== 3. reward score =====
        reward_score = reward_model.get_rewards(data[i]["solution"], responses[i])  # 一批 response
        

        results.append({
            "logprob": logprob_sum,
            "similarity_matrix": similarity_matrix,
            "reward_score": reward_score
        })

    return results


def compute_decoding_scores_from_batch_eval(
    eval_results: List[List[Dict]],
    proxy_key: str,
    gold_key: str,
    alphas: List[float],
    model_name: str,
    save_path: str = None
):
    """
    计算 BoN、MBR、KL-BoN、WD-BoN、Oracle、Random 等 decoding 策略的分数。

    :param eval_results: List，每个元素是一条样本的 List[Dict]，每个 Dict 包含单个 candidate 的：
                         {"logprob", "reward_score", "similarity_matrix"}（from evaluate_batch_responses）
    :param proxy_key: 用于排序的代理 reward 名称，对应 Dict 中的 "reward_score"
    :param gold_key: 用作 gold reward 的字段
    :param alphas: 正则化超参（用于 KL-BoN / WD-BoN）
    :param model_name: logprob 字段名称（用于标识）
    :param save_path: 保存 CSV 的路径（可选）
    :return: DataFrame
    """

    rows = []

    for instance in eval_results:
        ncandidates = len(instance)
        rewards = np.array([x["reward_score"] for x in instance])
        logprobs = np.array([x["logprob"] for x in instance])
        sims = np.array([
            x["similarity_matrix"].mean(axis=1).mean() if isinstance(x["similarity_matrix"], np.ndarray)
            else x["similarity_matrix"]
            for x in instance
        ])  # 简化：取 similarity matrix 的均值

        # ========== BoN ==========
        bon_idx = rewards.argmax()
        bon_score = rewards[bon_idx]

        # ========== MBR ==========
        mbr_idx = sims.argmax()
        mbr_score = rewards[mbr_idx]

        # ========== WD-BoN ==========
        wd_bon_scores = []
        for alpha in alphas:
            score = (rewards + alpha * sims).argmax()
            wd_bon_scores.append(rewards[score])

        # ========== KL-BoN ==========
        kl_bon_scores = []
        for alpha in alphas:
            score = (rewards + alpha * logprobs).argmax()
            kl_bon_scores.append(rewards[score])

        # ========== Oracle & Random ==========
        oracle_score = rewards.max()
        random_score = rewards.mean()

        row = [bon_score, mbr_score] + wd_bon_scores + kl_bon_scores + [oracle_score, random_score]
        rows.append(row)

    df = pd.DataFrame(rows, columns=(
        ["BoN", "MBR"] +
        [f"WD-BoN-{alpha}" for alpha in alphas] +
        [f"KL-BoN-{alpha}" for alpha in alphas] +
        ["Oracle", "Random"]
    ))

    if save_path:
        df.to_csv(save_path, index=False)

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aime")
    parser.add_argument("--model_id", type=int, default="2")
    parser.add_argument("--ncandidates", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0])
    args = parser.parse_args()

    print("Loading tokenizer and model...")
    llm = LLM(
        ID_2_MODELS[args.model_id],
        tensor_parallel_size=1,
        enforce_eager=True, 
        gpu_memory_utilization=0.95,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        ID_2_MODELS[args.model_id]
    )

    stop_token_ids = tokenizer("<|im_end|>")["input_ids"]
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=args.temperature
    )
    compute_similarity = load_similarity("comet")
    reward_model = load_reward_model("llm-blender/PairRM")

    print("Loading dataset...")
    data = load_my_dataset(args.dataset)
    data=data[:10]

    eval_results = []

    print("Evaluating responses with logprob, similarity, reward...")
    for i in tqdm(range(min(len(data), args.max_instances))):
        results = evaluate_batch_responses(
            data, 
            sampling_params, 
            args.ncandidates, 
            tokenizer, 
            llm, 
            compute_similarity, 
            reward_model
        )
        eval_results.append(results)

    print("Computing decoding scores (BoN, MBR, etc)...")
    save_name = f"{args.dataset}_{os.path.basename(args.model_path)}_{os.path.basename(args.reward_model_path)}.csv"
    save_path = os.path.join(args.save_dir, save_name)
    os.makedirs(args.save_dir, exist_ok=True)

    df = compute_decoding_scores_from_batch_eval(
        eval_results=eval_results,
        proxy_key="reward_score",
        gold_key="reward_score",
        alphas=args.alphas,
        model_name=os.path.basename(args.model_path),
        save_path=save_path
    )

    print(f"Done! Scores saved to: {save_path}")


if __name__ == "__main__":
    main()
    