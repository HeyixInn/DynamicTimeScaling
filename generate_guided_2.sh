CUDA_VISIBLE_DEVICES=5 python generate.py --model_id=4 --temperature=0.7 --data_name=openaimath
CUDA_VISIBLE_DEVICES=5 python generate.py --model_id=6 --temperature=0.7 --data_name=openaimath

CUDA_VISIBLE_DEVICES=5 python generate_guided.py --model_id=4 --data_name=openaimath
CUDA_VISIBLE_DEVICES=5 python generate_guided.py --model_id=6 --data_name=openaimath