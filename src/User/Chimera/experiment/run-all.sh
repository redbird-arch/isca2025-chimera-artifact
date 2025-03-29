#!/bin/bash

# ==============================================
# Synthetic Experiments (Figure 10)
# ==============================================
echo "Running synthetic experiments for Figure 10..."
make run_synthetic
echo "Synthetic experiments completed."

# ==============================================
# Run scale-sim-v2 first for next experiments
# ==============================================
echo "Running scale-sim-v2 for subsequent experiments..."
cd ../../../computation/backend/
make forward
make backward
echo "scale-sim-v2 setup completed."

# ==============================================
# Forward Pass Experiments (Figure 12)
# ==============================================
echo "Running forward pass experiments for Figure 12..."
cd ../../User/Chimera/experiment/
make generate_forward 
make run_forward
echo "Forward pass experiments completed."

# ==============================================
# Backward Pass Experiments (Figure 13(a))
# Overlapped weight-gradient computation with input-gradient communication
# ==============================================
echo "Running backward pass experiments for Figure 13(a)..."
make generate_backward 
make run_backward
echo "Backward pass experiments completed."

# ==============================================
# Backward Pass with Extra Pipelining (Figure 13(b))
# Chunk-pipelining overlapping experiments
# ==============================================
echo "Running backward pass with extra pipelining for Figure 13(b)..."
make generate_backward_pipeline 
make run_backward_pipeline
echo "Backward pass pipelining experiments completed."

echo "All experiments completed successfully!"