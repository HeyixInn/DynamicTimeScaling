# CUDA_VISIBLE_DEVICES=1 python generate.py --model_id=4 --temperature=0 --data_name=aime&
# CUDA_VISIBLE_DEVICES=3 python generate.py --model_id=4 --temperature=0 --data_name=gpqa&
# CUDA_VISIBLE_DEVICES=3 python generate.py --model_id=4 --temperature=0 --data_name=gpqa&
CUDA_VISIBLE_DEVICES=4 python generate.py --model_id=6 --temperature=0 --data_name=openaimath&
CUDA_VISIBLE_DEVICES=2 python generate.py --model_id=4 --temperature=0 --data_name=aime&
wait

# CUDA_VISIBLE_DEVICES=4 python generate.py --model_id=6 --temperature=0 --data_name=openaimath&
# CUDA_VISIBLE_DEVICES=1 python generate_s1.py --model_id=4 --temperature=0 --data_name=aime&
# CUDA_VISIBLE_DEVICES=7 python generate_s1.py --model_id=4 --temperature=0 --data_name=gpqa&
# wait
# CUDA_VISIBLE_DEVICES=3 python generate_s1.py --model_id=4 --temperature=0 --data_name=openaimath&
# CUDA_VISIBLE_DEVICES=4 python generate_s1.py --model_id=6 --temperature=0 --data_name=aime&
# CUDA_VISIBLE_DEVICES=1 python generate_s1.py --model_id=6 --temperature=0 --data_name=gpqa&
# CUDA_VISIBLE_DEVICES=7 python generate_s1.py --model_id=6 --temperature=0 --data_name=openaimath&
# wait
# CUDA_VISIBLE_DEVICES=3 python generate_guided.py --model_id=4 --temperature=0 --data_name=aime&
# CUDA_VISIBLE_DEVICES=4 python generate_guided.py --model_id=4 --temperature=0 --data_name=gpqa&
# CUDA_VISIBLE_DEVICES=1 python generate_guided.py --model_id=4 --temperature=0 --data_name=openaimath&
# CUDA_VISIBLE_DEVICES=7 python generate_guided.py --model_id=6 --temperature=0 --data_name=aime&
# wait
# CUDA_VISIBLE_DEVICES=3 python generate_guided.py --model_id=6 --temperature=0 --data_name=gpqa&
# CUDA_VISIBLE_DEVICES=4 python generate_guided.py --model_id=6 --temperature=0 --data_name=openaimath&
# wait