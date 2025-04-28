# CUDA_VISIBLE_DEVICES=4 python generate_pt_opt.py --data_name=aime --opt=0
# CUDA_VISIBLE_DEVICES=4 python generate_pt_opt.py --data_name=gpqa --opt=0
CUDA_VISIBLE_DEVICES=4 python generate_pt_opt.py --data_name=openaimath --opt=0

# CUDA_VISIBLE_DEVICES=4 python generate_pt_opt.py --data_name=aime --opt=0 --model_id=3
CUDA_VISIBLE_DEVICES=4 python generate_pt_opt.py --data_name=gpqa --opt=0 --model_id=3
# CUDA_VISIBLE_DEVICES=4 python generate_pt_opt.py --data_name=openaimath --opt=0 --model_id=3