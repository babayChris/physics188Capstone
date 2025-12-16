# A PDE Solver Based On FNO and U-Net
Here we demostrate the function of each code in the following topology graph:
```text
physics188Capstone/
├─ environment.yml
├─ data/
│  ├─ __pycache__/
│  └─ code/
│     └─ flowbench.ipynb
├─ src/
│  ├─ data_handle/
│  │  ├─ __pycache__/
│  │  ├─ data.py 
│  │  └─ data_final.py
│  ├─ eval/
│  │  ├─ __pycache__/
│  │  ├─ inference.py #Inference code for FNO
│  │  └─ inference.ipynb #Inference notebook for U-Net
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ __pycache__/
│  │  └─ FNO.py #FNO model structure
│  │
│  ├─ training/
│  │  ├─ __pycache__/
│  │  ├─ data.py # Data process file for U-Net
│  │  ├─ trainer_general.py
│  │  ├─ FNO.ipynb
│  │  ├─ UNet.ipynb # notebook for U-Net model structure and training
│  │  └─ epochs/
│  │     ├─ FNO_1/
│  │     │  ├─ model.txt 
│  │     │  ├─ FNO_no_rollout/
│  │     │  │  └─ model_weights_6
│  │     │  └─ rollout/
│  │     │     └─ model_weights_0
│  │     ├─ FNO_large/
│  │     │  ├─ best_model.pt
│  │     │  ├─ best_model_ar.pt
│  │     │  ├─ training_log.txt
│  │     │  └─ training_loss_visualization_1.ipynb
│  │     ├─ FNO_no_rollout/
│  │     │  └─ model_weights_6
│  │     └─ rollout/
│  │        └─ model_weights_0
└─ - Visualize/
   │  └─ Visualize.ipynb
   ├─ train.ipynb
   └─ eval.ipynb

```
