# CUDA_VISIBLE_DEVICES=0 python generate_pt_opt.py --data_name=aime --opt=0
# CUDA_VISIBLE_DEVICES=4 python generate_pt_opt.py --data_name=gpqa --opt=0
# CUDA_VISIBLE_DEVICES=0 python generate_pt_opt.py --data_name=openaimath --opt=0 --model_id=2

# CUDA_VISIBLE_DEVICES=0 python generate_pt_opt.py --data_name=aime --opt=0 --model_id=3
# CUDA_VISIBLE_DEVICES=5 python generate_pt_opt.py --data_name=gpqa --opt=0 --model_id=3
# CUDA_VISIBLE_DEVICES=4 python generate_pt_opt.py --data_name=openaimath --opt=0 --model_id=3

# CUDA_VISIBLE_DEVICES=6 python generate_pt_opt.py --data_name=aime --opt=0 --model_id=4
# CUDA_VISIBLE_DEVICES=6 python generate_pt_opt.py --data_name=gpqa --opt=0 --model_id=4
# CUDA_VISIBLE_DEVICES=6 python generate_pt_opt.py --data_name=openaimath --opt=0 --model_id=4


CUDA_VISIBLE_DEVICES=1 python generate_pt_opt.py --data_name=aime --opt=0 --model_id=6
CUDA_VISIBLE_DEVICES=1 python generate_pt_opt.py --data_name=gpqa --opt=0 --model_id=6
CUDA_VISIBLE_DEVICES=1 python generate_pt_opt.py --data_name=openaimath --opt=0 --model_id=6

CUDA_VISIBLE_DEVICES=1 python generate_pt_opt.py --data_name=aime --opt=1 --model_id=6
CUDA_VISIBLE_DEVICES=1 python generate_pt_opt.py --data_name=gpqa --opt=1 --model_id=6
CUDA_VISIBLE_DEVICES=1 python generate_pt_opt.py --data_name=openaimath --opt=1 --model_id=6
