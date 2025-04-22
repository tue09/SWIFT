export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 generate_vllm.py --model model_hub/gpt2_120M --input_dir data/Ultrachat200k --frac_len 5000 --data_frac 0 --world_size 1 --output_dir data/Ultrachat200k/ite0  --split train  
python3 generate_vllm.py --model model_hub/gpt2_120M --input_dir data/Ultrachat200k --frac_len 5000 --data_frac 1 --world_size 1 --output_dir data/Ultrachat200k/ite0  --split train
python3 generate_vllm.py --model model_hub/gpt2_120M --input_dir data/Ultrachat200k --frac_len 5000 --data_frac 2 --world_size 1 --output_dir data/Ultrachat200k/ite0  --split train
python3 generate_vllm.py --model model_hub/gpt2_120M --input_dir data/Ultrachat200k --frac_len 5000 --data_frac 3 --world_size 1 --output_dir data/Ultrachat200k/ite0  --split train
python3 generate_vllm.py --model model_hub/gpt2_120M --input_dir data/Ultrachat200k --frac_len 5000 --data_frac 4 --world_size 1 --output_dir data/Ultrachat200k/ite0  --split train
python3 generate_vllm.py --model model_hub/gpt2_120M --input_dir data/Ultrachat200k --frac_len 5000 --data_frac 5 --world_size 1 --output_dir data/Ultrachat200k/ite0  --split train
python3 generate_vllm.py --model model_hub/gpt2_120M --input_dir data/Ultrachat200k --frac_len 5000 --data_frac 6 --world_size 1 --output_dir data/Ultrachat200k/ite0  --split train
python3 generate_vllm.py --model model_hub/gpt2_120M --input_dir data/Ultrachat200k --frac_len 5000 --data_frac 7 --world_size 1 --output_dir data/Ultrachat200k/ite0  --split train
python3 generate_vllm.py --model model_hub/gpt2_120M --input_dir data/Ultrachat200k --frac_len 5000 --data_frac 8 --world_size 1 --output_dir data/Ultrachat200k/ite0  --split train
python3 generate_vllm.py --model model_hub/gpt2_120M --input_dir data/Ultrachat200k --frac_len 5000 --data_frac 9 --world_size 1 --output_dir data/Ultrachat200k/ite0  --split train
python3 generate_vllm.py --model model_hub/gpt2_120M --input_dir data/Ultrachat200k --frac_len 5000 --data_frac 10 --world_size 1 --output_dir data/Ultrachat200k/ite0 --split train

# Generate for the test split as well
python3 generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --frac_len 5000 --data_frac 0 --world_size 8 --output_dir generated/iter0 --split test