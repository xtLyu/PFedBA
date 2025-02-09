# PFedBA
In this repository, code is for our Usenix Security 2024 paper [Lurking in the shadows: Unveiling Stealthy Backdoor Attacks against Personalized Federated Learning](https://www.usenix.org/conference/usenixsecurity24/presentation/lyu)

## Installation
Install Pytorch

- run experiments for the Fashion-MNIST dataset:
  
python -u main.py --dataset FashionMnist --model dnn --learning_rate 0.1 --numusers 10 --local_epochs 20 --num_global_iters 150 --algorithm FedAvg --per_epoch 1 --poisoning_per_batch 16 --attack_method attackall --attack_start 30 --defense none --descrip avg_attackall

nohup python -u main.py --dataset FashionMnist --model dnn --learning_rate 0.1 --numusers 10 --local_epochs 20 --num_global_iters 150 --algorithm FedAvg --per_epoch 1 --poisoning_per_batch 16 --attack_method attackall --attack_start 30 --defense mkrum --descrip avg_attackall 

python -u main.py --dataset FashionMnist --model dnn --learning_rate 0.1 --numusers 10 --local_epochs 20 --num_global_iters 150 --algorithm FedAvg --per_epoch 1 --poisoning_per_batch 16 --attack_method attackall --attack_start 30 --defense trim --descrip avg_attackall 

python -u main.py --dataset FashionMnist --model dnn --learning_rate 0.1 --numusers 10 --local_epochs 20 --num_global_iters 150 --algorithm FedProx --per_epoch 1 --poisoning_per_batch 16 --attack_method attackall --attack_start 30 --defense none --descrip prox_attackall 

python -u main.py --dataset FashionMnist --model dnn --learning_rate 0.1 --numusers 10 --local_epochs 20 --num_global_iters 150 --algorithm FedProx --per_epoch 1 --poisoning_per_batch 16 --attack_method attackall --attack_start 30 --defense mkrum --descrip prox_attackall

python -u main.py --dataset FashionMnist --model dnn --learning_rate 0.1 --numusers 10 --local_epochs 20 --num_global_iters 150 --algorithm FedProx --per_epoch 1 --poisoning_per_batch 16 --attack_method attackall --attack_start 30 --defense trim --descrip prox_attackall

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings {299621,
author = {Xiaoting Lyu and Yufei Han and Wei Wang and Jingkai Liu and Yongsheng Zhu and Guangquan Xu and Jiqiang Liu and Xiangliang Zhang},
title = {Lurking in the shadows: Unveiling Stealthy Backdoor Attacks against Personalized Federated Learning},
booktitle = {33rd USENIX Security Symposium (USENIX Security 24)},
year = {2024},
isbn = {978-1-939133-44-1},
address = {Philadelphia, PA},
pages = {4157--4174},
url = {https://www.usenix.org/conference/usenixsecurity24/presentation/lyu},
publisher = {USENIX Association},
month = aug
}
```
