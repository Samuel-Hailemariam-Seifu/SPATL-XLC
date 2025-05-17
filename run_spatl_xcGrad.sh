nohup python spatl_federated_learning.py \
  --datadir ./data \
  --logdir ./logs/spatl-xcGrad \
  --model resnet20 \
  --dataset cifar10 \
  --alg spatl-xcGrad \
  --n_parties 3 \
  --sample 1.0 \
  --partition noniid-labeldir \
  --noise 0 \
  --rho 0.9 \
  --lr 0.01 \
  --batch-size 64 \
  --num_clusters 2 \
  --epochs 2 \
  --comm_round 2 \
  --optimizer sgd \
  --init_seed 0 \
  --device cuda \
  > spatl-xcGrad_train.log 2>&1 &