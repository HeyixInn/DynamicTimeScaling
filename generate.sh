CUDA_VISIBLE_DEVICES=3 python generate.py --model_id=4 --temperature=0.7 --data_name=aime
CUDA_VISIBLE_DEVICES=3 python generate_s1.py --model_id=4 --temperature=0.7 --data_name=aime

CUDA_VISIBLE_DEVICES=3 python generate.py --model_id=4 --temperature=0.7 --data_name=gpqa
CUDA_VISIBLE_DEVICES=3 python generate.py --model_id=4 --temperature=0.7 --data_name=openaimath


CUDA_VISIBLE_DEVICES=3 python generate_s1.py --model_id=4 --temperature=0.7 --data_name=gpqa
CUDA_VISIBLE_DEVICES=3 python generate_s1.py --model_id=4 --temperature=0.7 --data_name=openaimath