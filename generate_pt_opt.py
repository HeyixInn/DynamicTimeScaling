import argparse
import json
from utils import *

from pt_opt.grips import GripsOptimizer
from pt_opt.llm_opt import LLMOptimizer
from pt_opt import AIME_PT_LIST
from pt_opt.evaluator import aime_evaluator, llm_evaluator

from utils import parse_litellm_output, parse_vllm_output
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["HUGGINGFACE_API_KEY"] = "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"

PT_OPT_LIST = [
    GripsOptimizer,
    LLMOptimizer,
]


def evaluate_pt(pt, llm, dataset, evaluator, agent):
    messages = [pt.task2msg(task) for task in dataset]

    llm_output_list = llm.chat(messages, sampling_params=llm.sampling_params)
    llm_outputs = [llm.parse_func(d) for d in llm_output_list]

    return_dict = evaluator(llm_outputs, dataset, agent)

    return return_dict



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=2)
    parser.add_argument('--data_name', type=str, default='aime')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=30000) # TODO
    parser.add_argument('--overwrite', type=bool, default=False)
    parser.add_argument('--opt', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default = 0.1)

    args = parser.parse_args()
    
    save_path = f"./results_pt_opt/{ID_2_MODELS[args.model_id].split('/')[-1]}/opt_{args.opt}/"
    # source_path = f"./results/{ID_2_MODELS[args.model_id].split('/')[-1]}/"
    save_file = f"{args.data_name}.json"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_type = get_model_type(ID_2_MODELS[args.model_id])

    # save_path = f"./pt_opt_{args.opt}/{ID_2_MODELS[args.model_id].split('/')[-1]}__{args.data_name}/"
    # os.makedirs(save_path, exist_ok=True)

    train_shuffle = random.Random(42)
    dataset = load_my_dataset(args.data_name)
    train_shuffle.shuffle(dataset)
    train_num = round(len(dataset)*args.train_ratio)   # TODO # int(len(dataset) * 0.1)  #TOOD use fordebug
    # train_num=3
    # test_num= 20
    train_data, valid_data, test_data = dataset[:train_num], dataset[:train_num], dataset[train_num:]
    # test_data = test_data[:test_num]

    agent = load_agent()
    model = LLM(
        ID_2_MODELS[args.model_id],
        trust_remote_code = True,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_seq_len_to_capture=args.max_tokens,
        gpu_memory_utilization=0.95,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        skip_special_tokens=False,
        temperature=0.0,
        logprobs=1,
    )
    model.sampling_params= sampling_params
    model.parse_func = parse_vllm_output

    if args.data_name == 'aime':
        seed_pts = AIME_PT_LIST
        evaluator = aime_evaluator
    elif args.data_name == 'gpqa' or args.data_name == 'openaimath':
        seed_pts = AIME_PT_LIST
        evaluator = llm_evaluator
    else:
        raise NotImplementedError

    pt_class = PT_OPT_LIST[args.opt]
    pt_optimizer = pt_class(
        seed_pts, model, agent, evaluator,
        train_data, valid_data, test_data,{"num_steps": 3}
    )
    opt_res = pt_optimizer.optimize_pt()

    # print(evaluate_re)


    # outputs_no_budget = model.generate(prompts_no_budget, sampling_params=sampling_params)
    # no_budget_texts = [o.outputs[0].text for o in outputs_no_budget]
    #
    # results = []
    # for i, d in enumerate(dataset):
    #     results.append({
    #         'question': d['question'],
    #         'solution': d['solution'],
    #         'model_output': no_budget_texts[i]
    #     })
    #
    results = [
        {
            "best_pt": opt_res["best_pt"].instruction,
            "best_pt_list": [pt.instruction for pt in opt_res["best_pt_list"]],
            # "valid_scores_list": opt_res["valid_scores_list"],
            "train_scores_list": opt_res["train_scores_list"]
        }
    ]
    with open(save_path + save_file, 'w') as file:
        json.dump(results, file)
        file.flush()
    evaluate_re =  evaluate_pt(opt_res["best_pt"], model, dataset, evaluator, agent)
    with open(save_path + save_file.replace(".json","_eval.json"), 'w') as file:
        json.dump(evaluate_re, file)
        file.flush()