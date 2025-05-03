# python evaluate.py  --model_id=2 --mode=greedy --data_name=gpqa
# python evaluate.py  --model_id=3 --mode=greedy --data_name=gpqa
# python evaluate.py  --model_id=2 --mode=sampling --data_name=gpqa
# python evaluate.py  --model_id=3 --mode=sampling --data_name=gpqa

# python evaluate.py  --model_id=2 --mode=greedy --data_name=openaimath
# python evaluate.py  --model_id=3 --mode=greedy --data_name=openaimath

# python evaluate.py  --model_id=3 --mode=sampling --data_name=aime
# python evaluate.py  --model_id=2 --mode=sampling --data_name=openaimath
# python evaluate.py  --model_id=3 --mode=sampling --data_name=openaimath

python evaluate.py --source_dir=results_s1 --model_id=2 --mode=greedy --data_name=aime
python evaluate.py --source_dir=results_s1 --model_id=2 --mode=greedy --data_name=gpqa
python evaluate.py --source_dir=results_s1 --model_id=2 --mode=greedy --data_name=openaimath

python evaluate.py --source_dir=results_s1 --model_id=3 --mode=greedy --data_name=aime
python evaluate.py --source_dir=results_s1 --model_id=3 --mode=greedy --data_name=gpqa
python evaluate.py --source_dir=results_s1 --model_id=3 --mode=greedy --data_name=openaimath

# python evaluate.py --source_dir=results_s1 --model_id=2 --mode=greedy --data_name=gpqa
