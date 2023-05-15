#!/bin/bash

project=ff-convmixer-deeper 
up_lrs=(5e-4 1e-4 1e-4 1e-5)
down_lrs=(5e-4 1e-4 1e-3 1e-5)

for (( i=0; i<${#up_lrs[@]}; i++ )); do
    up_lr=${up_lrs[i]}
    down_lr=${down_lrs[i]}
    echo "up_lr=$up_lr, down_lr=$down_lr"
    # 在这里添加你想要执行的命令
    CUDA_VISIBLE_DEVICES=5 python main.py wandb.setup.project=$project model.depth=12 model.num_groups_each_layer=64 training.learning_rate=$up_lr training.downstream_learning_rate=$down_lr
done

