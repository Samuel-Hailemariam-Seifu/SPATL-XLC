#!/bin/bash

# Set shared arguments
COMMON_ARGS="--dataset cifar10 --model resnet20 --n_parties 2 --comm_round 200 --sample 1.0 --is_same_initial 1 --target_acc 8.0 --batch-size 64 --lr 0.01 --epochs 5"

# Create a logs directory if it doesn't exist
mkdir -p logs

# Run FedAvg
nohup python3 spatl_federated_learning.py --alg fedavg --logdir ./logs/fedavg $COMMON_ARGS > fedavg.log 2>&1 &

# # Run FedProx
# nohup python3 spatl_federated_learning.py --alg fedprox --logdir ./logs/fedprox $COMMON_ARGS > logs/fedprox.log 2>&1 &

# # Run SPATL
# nohup python3 spatl_federated_learning.py --alg spatl --logdir ./logs/spatl $COMMON_ARGS > logs/spatl.log 2>&1 &

# # Run SPATL-XCGrad
# nohup python3 spatl_federated_learning.py --alg spatl-xcGrad --logdir ./logs/spatl-xcGrad  $COMMON_ARGS --explainer_type gradient > logs/spatl_xcGrad.log 2>&1 &

# # Run SPATL-XCDeep
# nohup python3 spatl_federated_learning.py --alg spatl-xcDeep --logdir ./logs/spatl-xcDeep $COMMON_ARGS --explainer_type deep > logs/spatl_xcDeep.log 2>&1 &

# # Run SCAFFOLD
# nohup python3 spatl_federated_learning.py --alg sccafold --logdir ./logs/sccafold $COMMON_ARGS > logs/sccafold.log 2>&1 &

# echo "All algorithms launched in background with nohup."
