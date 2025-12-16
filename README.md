# A PDE Solver Based On FNO and U-Net
Here we demostrate the function of each code in the following topology graph:
```text
physics188Capstone/
├─ environment.yml
├─ src/
│  ├─ data_handle/
│  │  ├─ __pycache__/
│  │  ├─ data.py #original simple dataclass (NOTE: not use in training) 
│  │  └─ data_final.py #dataclass with lazy loading and preprocessing
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
│  │  └─ epochs/ #Model Weights
|  |     |UNET WEIGHTS LINK: https://drive.google.com/file/d/10-9bz0AmOyYhf6EY8mVWiEOpisVBgruZ/view (file was too large for github)
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
   ├─ train.ipynb #FNO training notebook
   └─ eval.ipynb #FNO visualizations

```
