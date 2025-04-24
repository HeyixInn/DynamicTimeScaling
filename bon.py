import argparse
import numpy as np
import json

def select_by_logprob(results, score_type):
    if score_type=='avg':
        scores = [[np.mean(seq) for seq in r['sample_logprobs']] for r in results]
    elif score_type=='sum':
        scores = [[np.sum(seq) for seq in r['sample_logprobs']] for r in results]
    return scores

def select_by_cons(results, score_type):
    scores = []
    for r in results:
        sampled_text_list = r['sample_texts']
        sampled_logits_list = r['sample_logprobs']

        vocab = {token for logits in sampled_logits_list for (token, _) in logits}
        vocab_list = list(vocab)
        vocab_index = {token: idx for idx, token in enumerate(vocab_list)}

        matrix = np.zeros((len(sampled_logits_list), len(vocab)), dtype=np.float32)
        if score_type=="base":
            for i, logits in enumerate(sampled_logits_list):
                for token, _ in logits:
                    matrix[i, vocab_index[token]] = 1.0
        elif score_type=="weight":
            for i, logits in enumerate(sampled_logits_list):
                v = np.zeros(len(vocab), dtype=np.float32)
                count = np.zeros(len(vocab), dtype=np.float32)
                for token, logit in logits:
                    idx = vocab_index[token]
                    count[idx] += 1
                    v[idx] += np.exp(logit)  # softmax前的指数变换
                matrix[i] = v / (count + 1e-9)  # 避免除0
        elif score_type=="consensus":
            for i, logits in enumerate(sampled_logits_list):
                v = np.zeros(len(vocab), dtype=np.float32)
                count = np.zeros(len(vocab), dtype=np.float32)
                for token, logit in logits:
                    idx = vocab_index[token]
                    count[idx] += 1
                    v[idx] += np.exp(logit)
                matrix[i] = v / (count + 1e-9)
        sim = (matrix @ matrix.T) / len(vocab)
        avg_score = sim.mean(axis=1)
        if score_type=="consensus":
            for i, logits in enumerate(sampled_logits_list):
                confidence_score = np.exp(np.mean([logit for _, logit in logits]))
                avg_score[i] *= confidence_score

        scores.append([float(avg_score[i]) for i in range(len(sampled_text_list))])
    print(scores)
    return scores

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

def select(results, select_type="logprob_avg"):
    select_type, score_type = select_type.split('_')
    
    if select_type=='logprob':
        scores = select_by_logprob(results, score_type)
    elif select_type=='cons':
        scores = select_by_cons(results, score_type)
    return scores
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=2)
    parser.add_argument('--data_name', type=str, default='aime')
    parser.add_argument('--overwrite', type=bool, default = False)
    parser.add_argument('--nsamples', type=int, default = 5)
    parser.add_argument('--select_type', type=str, default = "logprob_avg")

    args = parser.parse_args()

    source_path = f"./results_sample/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    save_file = f"{args.data_name}_seleted_by_{args.select_type}.json"
    with open(source_path+f"{args.data_name}.json") as file:
        data = json.load(file)
    
    
    scores = select(data, args.select_type)
    for d, score in zip(data, scores):
        d.update({
            "scores": score
        })
    print(source_path+save_file)
    with open(source_path+save_file, 'w') as file:
        json.dump(data, file)