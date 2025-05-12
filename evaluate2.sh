# python evaluate.py --source_dir=results_guided** --model_id=2 --mode=greedy --data_name=aime
# python evaluate.py --source_dir=results_guided** --model_id=2 --mode=greedy --data_name=gpqa
# python evaluate.py --source_dir=results_guided** --model_id=2 --mode=greedy --data_name=openaimath

# python evaluate.py --source_dir=results_guided** --model_id=3 --mode=greedy --data_name=aime
# python evaluate.py --source_dir=results_guided** --model_id=3 --mode=greedy --data_name=gpqa
# python evaluate.py --source_dir=results_guided** --model_id=3 --mode=greedy --data_name=openaimath
# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=2 --opt=0 --data_name=aime
# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=2 --opt=1 --data_name=aime
# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=3 --opt=0 --data_name=aime
# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=3 --opt=1 --data_name=aime
# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=4 --opt=0 --data_name=aime
# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=4 --opt=1 --data_name=aime
# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=6 --opt=0 --data_name=aime
# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=6 --opt=1 --data_name=aime
# CUDA_VISIBLE_DEVICES=0 python evaluate_pt_opt.py --model_id=4 --opt=0 --data_name=aime &
# CUDA_VISIBLE_DEVICES=2 python evaluate_pt_opt.py --model_id=4 --opt=1 --data_name=aime &
# CUDA_VISIBLE_DEVICES=3 python evaluate_pt_opt.py --model_id=6 --opt=0 --data_name=aime &
# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=6 --opt=1 --data_name=aime &
CUDA_VISIBLE_DEVICES=5 python evaluate_pt_opt.py --model_id=4 --opt=0 --data_name=openaimath &
CUDA_VISIBLE_DEVICES=6 python evaluate_pt_opt.py --model_id=4 --opt=1 --data_name=openaimath &
# CUDA_VISIBLE_DEVICES=7 python evaluate_pt_opt.py --model_id=6 --opt=0 --data_name=gpqa &
wait
# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=6 --opt=1 --data_name=gpqa &
# CUDA_VISIBLE_DEVICES=0 python evaluate_pt_opt.py --model_id=4 --opt=0 --data_name=openaimath &
# CUDA_VISIBLE_DEVICES=2 python evaluate_pt_opt.py --model_id=4 --opt=1 --data_name=openaimath &
# CUDA_VISIBLE_DEVICES=3 python evaluate_pt_opt.py --model_id=6 --opt=0 --data_name=openaimath &
# CUDA_VISIBLE_DEVICES=5 python evaluate_pt_opt.py --model_id=6 --opt=1 --data_name=openaimath &
# wait
