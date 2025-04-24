import argparse
import numpy as np

def select_by_logprob(results, score_type):
    if score_type=='avg':
        scores = [[np.mean(seq) for seq in sample] for sample in results]
    elif score_type=='sum':
        scores = [[np.sum(seq) for seq in sample] for sample in results]

def select_by_cons(results, score_type):
    scores = []
    for r in results:
        sampled_text_list = r['sample_texts']
        sampled_logits_list = r['sample_logprobs']

        vocab = {token for logits in sampled_logits_list for (token, _) in logits}
        vocab_list = list(vocab)
        vocab_index = {token: idx for idx, token in enumerate(vocab_list)}

        matrix = np.zeros((len(sampled_logits_list), len(vocab)), dtype=np.float32)
        for i, logits in enumerate(sampled_logits_list):
            for token, _ in logits:
                matrix[i, vocab_index[token]] = 1.0

        sim = (matrix @ matrix.T) / len(vocab)
        avg_score = sim.mean(axis=1)

        scores.append([avg_score[i] for i in range(len(sampled_text_list))])

def select(results, select_type="logprob_avg"):
    select_type, score_type = select_type.split('_')
    
    if select_type=='logprob':
        select_by_logprob(results, score_type)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=2)
    parser.add_argument('--data_name', type=str, default='aime')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default = 30000)
    parser.add_argument('--overwrite', type=bool, default = False)
    parser.add_argument('--nsamples', type=int, default = 5)
    parser.add_argument('--select_type', type=int, default = "logprob_avg")

    args = parser.parse_args()

    source_path = f"./results_sample/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    save_file = f"{args.data_name}_seleted_by_{args.select_type}.json"