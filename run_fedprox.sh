nohup python spatl_federated_learning.py \
  --datadir ./data \
  --logdir ./logs/fedprox \
  --model resnet20 \
  --dataset cifar10 \
  --mu=0.01 \
  --alg fedprox \
  --n_parties 3 \
  --sample 1.0 \
  --partition noniid-labeldir \
  --noise 0 \
  --rho 0.9 \
  --lr 0.01 \
  --batch-size 64 \
  --epochs 5 \
  --comm_round 1 \
  --optimizer sgd \
  --init_seed 0 \
  --device cuda \
  > fedprox_train.log 2>&1 &