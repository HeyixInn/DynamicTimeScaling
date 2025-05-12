# CUDA_VISIBLE_DEVICES=2 python dy_timescaling.py --model_id=4 --data_name=aime&
CUDA_VISIBLE_DEVICES=1 python dy_timescaling.py --model_id=4 --data_name=openaimath&
CUDA_VISIBLE_DEVICES=0 python dy_timescaling.py --model_id=4 --data_name=gpqa&
CUDA_VISIBLE_DEVICES=3 python dy_timescaling.py --model_id=6 --data_name=aime&
# CUDA_VISIBLE_DEVICES=4 python dy_timescaling.py --model_id=6 --data_name=openaimath&
CUDA_VISIBLE_DEVICES=5 python dy_timescaling.py --model_id=6 --data_name=gpqa&
wait

