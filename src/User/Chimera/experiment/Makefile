num_threads=32

# Experiment 1: Effective Bandwidth with the Scaling of Message Size (Figure 10: 672 experiments)
run_synthetic:
	./run_script.sh ./ISCA25_Synthetic/py/ $(num_threads)


# Experiment 2: End-to-End Speedups for Forward Pass on Common Models (Figure 12: 64+16 experiments)
generate_forward: # just check whether the scripts are working
	python tp_alpa_gpt.py
	python tp_fusion_gpt.py
	python ts_base_gpt.py
	python ts_fusion_gpt.py
	python se_base_moe.py
	python se_fusion_moe.py
	python te_base_moe.py
	python te_fusion_moe.py
	python tp_alpa_gpt_3D.py
	python tp_fusion_gpt_3D.py
	python ts_base_gpt_3D.py
	python ts_fusion_gpt_3D.py
	python se_base_moe_3D.py
	python se_fusion_moe_3D.py
	python te_base_moe_3D.py
	python te_fusion_moe_3D.py 
# ./test_script.sh ./ISCA25_Forward/py
run_forward:
	./run_script.sh ./ISCA25_Forward/py $(num_threads)


# Experiment 3: End-to-End Speedups for Backward Pass with WG-comp & IG-comm overlapping (Figure 13(a): 64+16 experiments)
generate_backward: # just check whether the scripts are working
	python tp_alpa_gpt_backward.py
	python tp_fusion_gpt_backward.py
	python ts_base_gpt_backward.py
	python ts_fusion_gpt_backward.py
	python se_base_moe_backward.py
	python se_fusion_moe_backward.py
	python te_base_moe_backward.py
	python te_fusion_moe_backward.py 
	python tp_alpa_gpt_backward_3D.py
	python tp_fusion_gpt_backward_3D.py
	python ts_base_gpt_backward_3D.py
	python ts_fusion_gpt_backward_3D.py
	python se_base_moe_backward_3D.py
	python se_fusion_moe_backward_3D.py
	python te_base_moe_backward_3D.py
	python te_fusion_moe_backward_3D.py 
# ./test_script.sh ./ISCA25_Backward/py
run_backward:
	./run_script.sh ./ISCA25_Backward/py $(num_threads)


# Experiment 4: End-to-End Speedups for Backward Pass with extra chunk pipeline overlapping (Figure 13(b): 24+4 experiments)
generate_backward_pipeline: # just check whether the scripts are working
	python tp_alpa_gpt_backward_pipeline.py
	python tp_fusion_gpt_backward_pipeline.py
	python ts_base_gpt_backward_pipeline.py
	python ts_fusion_gpt_backward_pipeline.py
	python se_base_moe_backward_pipeline.py
	python se_fusion_moe_backward_pipeline.py
	python te_base_moe_backward_pipeline.py
	python te_fusion_moe_backward_pipeline.py 
# ./test_script.sh ./ISCA25_Backward-Pipeline/py
run_backward_pipeline:
	./run_script.sh ./ISCA25_Backward-Pipeline/py $(num_threads)
