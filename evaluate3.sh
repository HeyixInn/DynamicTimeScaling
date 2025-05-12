# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=2 --opt=0 --data_name=aime

# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=3 --opt=0 --data_name=aime

# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=4 --opt=0 --data_name=aime

# CUDA_VISIBLE_DEVICES=4 python evaluate_pt_opt.py --model_id=6 --opt=0 --data_name=aime

CUDA_VISIBLE_DEVICES=0 python evaluate_pt_opt.py --model_id=4 --opt=0 --data_name=openaimath
CUDA_VISIBLE_DEVICES=0 python evaluate_pt_opt.py --model_id=4 --opt=1 --data_name=openaimath
CUDA_VISIBLE_DEVICES=0 python evaluate_pt_opt.py --model_id=6 --opt=0 --data_name=openaimath
CUDA_VISIBLE_DEVICES=0 python evaluate_pt_opt.py --model_id=6 --opt=1 --data_name=openaimath