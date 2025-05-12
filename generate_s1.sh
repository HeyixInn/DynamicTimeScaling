# CUDA_VISIBLE_DEVICES=0 python generate_s1.py --model_id=2 --data_name=aime --temperature=0.7
# CUDA_VISIBLE_DEVICES=0 python generate_s1.py --model_id=3 --data_name=aime --temperature=0.7

# CUDA_VISIBLE_DEVICES=0 python generate_s1.py --model_id=2 --data_name=gpqa --temperature=0.7
# CUDA_VISIBLE_DEVICES=0 python generate_s1.py --model_id=2 --data_name=openaimath --temperature=0.7

# CUDA_VISIBLE_DEVICES=0 python generate_s1.py --model_id=3 --data_name=gpqa --temperature=0.7
# CUDA_VISIBLE_DEVICES=0 python generate_s1.py --model_id=3 --data_name=openaimath --temperature=0.7
# CUDA_VISIBLE_DEVICES=3 python generate_s1.py --model_id=3 --data_name=aime --temperature=0
# CUDA_VISIBLE_DEVICES=3 python generate_guided.py --model_id=3 --data_name=aime --temperature=0

CUDA_VISIBLE_DEVICES=6 python generate_s1.py --model_id=4 --temperature=0 --data_name=aime
CUDA_VISIBLE_DEVICES=6 python generate_s1.py --model_id=4 --temperature=0 --data_name=gpqa
CUDA_VISIBLE_DEVICES=6 python generate_s1.py --model_id=4 --temperature=0 --data_name=openaimath

# CUDA_VISIBLE_DEVICES=1 python generate_s1.py --model_id=6 --temperature=0 --data_name=aime
# CUDA_VISIBLE_DEVICES=1 python generate_s1.py --model_id=6 --temperature=0 --data_name=gpqa
# CUDA_VISIBLE_DEVICES=1 python generate_s1.py --model_id=6 --temperature=0 --data_name=openaimath