forward:
	mkdir -p ./runfile/medium_gpt_attention_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./medium_gpt_attention.csv -i gemm -p ./runfile/medium_gpt_attention_normal/	
	mkdir -p ./runfile/medium_gpt_mlp_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./medium_gpt_mlp.csv -i gemm -p ./runfile/medium_gpt_mlp_normal/
	mkdir -p ./runfile/moe_1point3_256_attention_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./moe_1point3_256_attention.csv -i gemm -p ./runfile/moe_1point3_256_attention_normal/	
	mkdir -p ./runfile/moe_1point3_256_mlp_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./moe_1point3_256_mlp.csv -i gemm -p ./runfile/moe_1point3_256_mlp_normal/
	mkdir -p ./runfile/moe_1point3_256_moe_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./moe_1point3_256_moe.csv -i gemm -p ./runfile/moe_1point3_256_moe_normal/


backward: 
	mkdir -p ./runfile/medium_gpt_attention_input_backward_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./medium_gpt_attention_input_backward.csv -i gemm -p ./runfile/medium_gpt_attention_input_backward/
	mkdir -p ./runfile/medium_gpt_attention_weight_backward_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./medium_gpt_attention_weight_backward.csv -i gemm -p ./runfile/medium_gpt_attention_weight_backward/
	mkdir -p ./runfile/medium_gpt_mlp_input_backward_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./medium_gpt_mlp_input_backward.csv -i gemm -p ./runfile/medium_gpt_mlp_input_backward/
	mkdir -p ./runfile/medium_gpt_mlp_weight_backward_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./medium_gpt_mlp_weight_backward.csv -i gemm -p ./runfile/medium_gpt_mlp_weight_backward/
	mkdir -p ./runfile/moe_1point3_256_attention_input_backward_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./moe_1point3_256_attention_input_backward.csv -i gemm -p ./runfile/moe_1point3_256_attention_input_backward/
	mkdir -p ./runfile/moe_1point3_256_attention_weight_backward_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./moe_1point3_256_attention_weight_backward.csv -i gemm -p ./runfile/moe_1point3_256_attention_weight_backward/
	mkdir -p ./runfile/moe_1point3_256_mlp_input_backward_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./moe_1point3_256_mlp_input_backward.csv -i gemm -p ./runfile/moe_1point3_256_mlp_input_backward/
	mkdir -p ./runfile/moe_1point3_256_mlp_weight_backward_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./moe_1point3_256_mlp_weight_backward.csv -i gemm -p ./runfile/moe_1point3_256_mlp_weight_backward/
	mkdir -p ./runfile/moe_1point3_256_moe_input_backward_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./moe_1point3_256_moe_input_backward.csv -i gemm -p ./runfile/moe_1point3_256_moe_input_backward/
	mkdir -p ./runfile/moe_1point3_256_moe_weight_backward_normal
	python ./scale-sim-v2/scalesim/scale.py -c ./scale-sim-v2/configs/normal.cfg -t ./moe_1point3_256_moe_weight_backward.csv -i gemm -p ./runfile/moe_1point3_256_moe_weight_backward/

