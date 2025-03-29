#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate Chimera
# generate Chimera_Synthetic_All_Patterns figure (together with Chimera_Synthetic_Legend.pdf to get Figure 10 in the paper)
python Chimera_Synthetic.py && echo "Finish Figure 10!"
# generate Chimera_EndtoEnd_forward figure (Figure 12(a) in the paper)
python Chimera_EndtoEnd_forward.py && echo "Finish Figure 12(a)!"
# generate Chimera_EndtoEnd_forward_scaling figure (Figure 12(b) in the paper)
python Chimera_EndtoEnd_forward_scaling.py && echo "Finish Figure 12(b)!"
# generate Chimera_EndtoEnd_backward figure (Figure 13(a) in the paper)
python Chimera_EndtoEnd_backward.py && echo "Finish Figure 13(a)!"
# generate Chimera_EndtoEnd_backward_PS figure (Figure 13(b) in the paper)
python Chimera_EndtoEnd_backward_pipeline.py && echo "Finish Figure 13(b)!"
# generate TP+SP, TP+PP, TP+EP, PP+EP, PP+SP, SP+EP figure (Figure 14(a)-(f) in the paper)
python Chimera_RealMachine.py && echo "Finish Figure 14!"