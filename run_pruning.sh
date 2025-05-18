 

nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=random --prune_ratio=0.2 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_random_0.2/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=random --prune_ratio=0.4 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_random_0.4/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=random --prune_ratio=0.6 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_random_0.6/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=magnitude --prune_ratio=0.2 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_magnitude_0.2/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=magnitude --prune_ratio=0.4 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_magnitude_0.4/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=magnitude --prune_ratio=0.6 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_magnitude_0.6/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=gradxinput --prune_ratio=0.2 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_gradxinput_0.2/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=gradxinput --prune_ratio=0.4 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_gradxinput_0.4/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=gradxinput --prune_ratio=0.6 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_gradxinput_0.6/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=shap --prune_ratio=0.2 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_shap_0.2/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=shap --prune_ratio=0.4 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_shap_0.4/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=shap --prune_ratio=0.6 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_shap_0.6/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=classwise_shap --prune_ratio=0.2 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_classwise_shap_0.2/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=classwise_shap --prune_ratio=0.4 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_classwise_shap_0.4/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=classwise_shap --prune_ratio=0.6 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_classwise_shap_0.6/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=entropy --prune_ratio=0.2 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_entropy_0.2/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=entropy --prune_ratio=0.4 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_entropy_0.4/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10

# nohup python spatl_federated_learning.py --model=resnet20 --dataset=cifar10 --alg=fedavg --pruning=entropy --prune_ratio=0.6 --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/fedavg_entropy_0.6/' --noise=0 --sample=0.4 --rho=0.9 --partition=noniid-labeldir --comm_round=200 --init_seed=0 --is_same_initial --num_classes=10
