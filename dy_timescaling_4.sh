# CUDA_VISIBLE_DEVICES=0,1,3,4 python generate.py --model_id=5 --temperature=0 --data_name=aime --max_tokens=30000
# CUDA_VISIBLE_DEVICES=0,1,3,4 python generate.py --model_id=5 --temperature=0 --data_name=gpqa --max_tokens=30000
# CUDA_VISIBLE_DEVICES=0,1,3,4 python generate.py --model_id=5 --temperature=0 --data_name=openaimath --max_tokens=30000

CUDA_VISIBLE_DEVICES=2,3,4,7 python dy_timescaling.py --model_id=5 --temperature=0 --data_name=aime --max_tokens=30000
CUDA_VISIBLE_DEVICES=2,3,4,7 python dy_timescaling.py --model_id=5 --temperature=0 --data_name=gpqa --max_tokens=30000
CUDA_VISIBLE_DEVICES=2,3,4,7 python dy_timescaling.py --model_id=5 --temperature=0 --data_name=openaimath --max_tokens=30000

CUDA_VISIBLE_DEVICES=2,3,4,7 python generate_guided.py --model_id=5 --temperature=0 --data_name=aime --max_tokens=30000
CUDA_VISIBLE_DEVICES=2,3,4,7 python generate_guided.py --model_id=5 --temperature=0 --data_name=gpqa --max_tokens=30000
CUDA_VISIBLE_DEVICES=2,3,4,7 python generate_guided.py --model_id=5 --temperature=0 --data_name=openaimath --max_tokens=30000
