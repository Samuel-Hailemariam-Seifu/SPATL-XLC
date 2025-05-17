import json
import logging
import os
import time
import torch
import logging
import sys
import datetime
import sys
import numpy as np
from torch.utils import data
import warnings
from federated_learning.network_training import local_update, local_update_scaffold, \
    local_update_scaffold_notransfer,local_update_shap, local_update_fedprox, train_net,train_net_fedavg  
from utils.load_neural_networks import init_nets
from utils.log_utils import mkdirs
from utils.parameters import get_parameter
from utils.data.prepare_data import partition_data, get_dataloader
from utils.save_model import save_checkpoint
import matplotlib.pyplot as plt
import copy
import shap
import numpy as np
import torch
import torch.nn as nn
import warnings
from utils.metrics import MetricsTracker
from sklearn.cluster import KMeans
import pandas as pd
from utils.accuracy import compute_acc
import warnings
warnings.filterwarnings("ignore")

def get_dataloader_from_map(args, net_dataidx_map, net_id):
    dataidxs = net_dataidx_map[net_id]
    noise_level = args.noise if net_id != args.n_parties - 1 else 0

    if args.noise_type == 'space':
        return get_dataloader(
            args.dataset, args.datadir, args.batch_size, 32,
            dataidxs, noise_level, net_id, args.n_parties - 1
        )
    else:
        adjusted_noise = args.noise / (args.n_parties - 1) * net_id
        return get_dataloader(
            args.dataset, args.datadir, args.batch_size, 32,
            dataidxs, adjusted_noise
        )


def plot_final_model_explainability(global_model, data_loader, device, logdir, explainer_type="gradient"):
    global_model.eval()
    batch = next(iter(data_loader))
    inputs, _ = batch
    inputs = inputs.to(device)

    # Choose SHAP explainer
    if explainer_type == "gradient":
        explainer = shap.GradientExplainer(global_model, inputs)
    elif explainer_type == "deep":
        explainer = shap.DeepExplainer(global_model, inputs)
    else:
        raise ValueError("Unsupported explainer type")

    # Compute SHAP values (just first 20 samples for speed)
    shap_values = explainer.shap_values(inputs[:20])
    mean_abs_shap = np.mean(np.abs(shap_values[0]), axis=0)

    # Bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(mean_abs_shap.flatten())), mean_abs_shap.flatten())
    plt.title("Final Model Global SHAP Feature Importance")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, "final_shap_importance.png"))
    plt.close()
    print(f"[Explainability] Saved SHAP bar plot to {logdir}/final_shap_importance.png")


def get_client_dataloader(client_idx, args, X_train, y_train, net_dataidx_map):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset_name = args.dataset.lower()
    if dataset_name == 'cifar10':
        dataset_class = datasets.CIFAR10
    elif dataset_name == 'mnist':
        dataset_class = datasets.MNIST
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    dataset = dataset_class(args.datadir, train=True, download=True, transform=transform)
    subset_indices = net_dataidx_map[client_idx]
    subset = Subset(dataset, subset_indices)
    return DataLoader(subset, batch_size=args.batch_size, shuffle=True)
    
def evaluate_global_model(global_model, args, device, logger):
    """
    Evaluate the test accuracy of the global model after aggregation.

    Args:
        global_model (nn.Module): The global model.
        args (Namespace): Arguments containing dataset config.
        device (str): Device to run the evaluation on.
        logger (Logger): Logger instance for recording accuracy.

    Returns:
        float: Global model test accuracy.
    """
    _, test_dl_global, _, _ = get_dataloader(
        args.dataset, args.datadir, args.batch_size, 32
    )

    criterion = torch.nn.CrossEntropyLoss().to(device)
    global_model.to(device)
    global_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_dl_global:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = global_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    global_acc = correct / total
    logger.info(f"[Global Model] Post-aggregation test accuracy: {global_acc:.4f}")
    return global_acc



def run_fedavg(args, device, logger, net_dataidx_map, test_dl_global):
    
    logger.info("Initializing nets for FedAvg")
    nets, _, _ = init_nets(args.n_parties, args.model, args)
    global_models, _, _ = init_nets(1, args.model, args)
    global_model = global_models[0]
    global_para = global_model.module.encoder.state_dict()
    successful_rounds = 0


    if args.is_same_initial:
        for net_id, net in nets.items():
            net.module.encoder.load_state_dict(global_para)

    t_start = time.time()
    lr = args.lr
    achieved_target = False

    tracker = MetricsTracker(num_rounds=args.comm_round, log_dir=args.logdir)

    for round in range(args.comm_round):
        tracker.rounds_completed += 1 ;
        if achieved_target:
            break

        tracker.start_round()
        logger.info(f"[FedAvg] in comm round: {round} " + "#" * 100)

        selected = np.random.choice(range(args.n_parties), int(args.n_parties * args.sample), replace=False)

        for idx in selected:
            nets[idx].module.encoder.load_state_dict(global_para)

        client_accs = []
        client_deltas = []

        train_dl, test_dl, *_ = get_dataloader_from_map(args, net_dataidx_map, idx)

        for k in nets:
            nets[k] = nets[k].to(device)


        for idx in selected:
            train_acc, test_acc, pre_train_acc, pre_test_acc = train_net_fedavg(
                idx, nets[idx], train_dl, test_dl, args.epochs, lr, args.optimizer, logger, args
            )

          

            client_accs.append(test_acc)

            local_para = nets[idx].module.encoder.state_dict()
            delta = {k: (local_para[k] - global_para[k].to(local_para[k].device)) for k in global_para}
            flat_delta = torch.cat([v.flatten() for v in delta.values()]).detach()
            client_deltas.append(flat_delta.cpu().numpy())

        client_drift = [np.linalg.norm(delta) for delta in client_deltas]
        tracker.update_client_metrics(client_accs=client_accs, client_drift=client_drift)


        total_data_points = sum(len(net_dataidx_map[r]) for r in selected)
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for i, idx in enumerate(selected):
            local_para = nets[idx].cpu().module.encoder.state_dict()
            if i == 0:
                for key in global_para:
                    global_para[key] = local_para[key] * fed_avg_freqs[i]
            else:
                for key in global_para:
                    global_para[key] += local_para[key] * fed_avg_freqs[i]

        global_model.module.encoder.load_state_dict(global_para)
        acc, top5 = compute_acc(test_dl_global, device, global_model, nn.CrossEntropyLoss())
        tracker.update_global_metrics(acc, top5)

        logger.info(f"[FedAvg] Round accuracy {acc:.2f}% at round {round}.")

        if acc >= args.target_acc:
            successful_rounds += 1
            logger.info(f"[FedAvg] Target accuracy {args.target_acc:.2f}% reached at round {round} ({successful_rounds}/3).")
        else:
            successful_rounds = 0  
        
        if successful_rounds >= 3:
            logger.info(f"[FedAvg] Target accuracy sustained for 3 rounds. Stopping.")
            achieved_target = True


        tracker.end_round()

    t_end = time.time()
    tracker.finalize(final_accuracy=acc)
    tracker.save_summary()

    logger.info("total communication time: %.2f seconds" % (t_end - t_start))
    logger.info("avg time per round: %.2f seconds" % ((t_end - t_start) / tracker.rounds_completed))


def run_fedprox(args, device, logger, net_dataidx_map, test_dl_global):
    tracker.rounds_completed += 1 ;
    logger.info("Initializing nets for FedProx")
    nets, _, _ = init_nets(args.n_parties, args.model, args)
    global_models, _, _ = init_nets(1, args.model, args)
    global_model = global_models[0]
    global_para = global_model.module.encoder.state_dict()

    if args.is_same_initial:
        for net_id, net in nets.items():
            net.module.encoder.load_state_dict(global_para)

    t_start = time.time()
    lr = args.lr
    mu = getattr(args, 'mu', 0.01)
    successful_rounds = 0
    tracker = MetricsTracker(num_rounds=args.comm_round, log_dir=args.logdir)

    for round in range(args.comm_round):
        tracker.rounds_completed += 1 ;
        if successful_rounds >= 3:
            logger.info(f"[FedProx] Target accuracy sustained for 3 consecutive rounds. Stopping.")
            break

        tracker.start_round()
        logger.info(f"[FedProx] in comm round: {round} " + "#" * 100)

        selected = np.random.choice(range(args.n_parties), int(args.n_parties * args.sample), replace=False)

        for idx in selected:
            nets[idx].module.encoder.load_state_dict(global_para)

        client_accs = []
        client_deltas = []

        for idx in selected:
            train_dl, test_dl, _, _ = get_dataloader_from_map(args, net_dataidx_map, idx)
            net = nets[idx]
            net.to(device)

            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss().to(device)

            net.train()
            for epoch in range(args.epochs):
                for batch_x, batch_y in train_dl:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    output = net(batch_x)
                    loss = criterion(output, batch_y)

                    prox_reg = 0.0
                    for (_, param), (_, g_param) in zip(net.named_parameters(), global_model.named_parameters()):
                        prox_reg += ((param - g_param.detach()) ** 2).sum()
                    loss += (mu / 2) * prox_reg

                    loss.backward()
                    optimizer.step()

            acc, _ = compute_acc(test_dl, device, net, criterion)
            client_accs.append(acc)

            local_para = net.module.encoder.state_dict()
            delta = {k: (local_para[k] - global_para[k].to(local_para[k].device)) for k in global_para}
            flat_delta = torch.cat([v.flatten() for v in delta.values()]).detach()
            client_deltas.append(flat_delta.cpu().numpy())

        # Compute client drift (L2 norm of deltas)
        client_drift = [np.linalg.norm(delta) for delta in client_deltas]
        tracker.update_client_metrics(client_accs=client_accs, client_drift=client_drift)

        total_data_points = sum(len(net_dataidx_map[r]) for r in selected)
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for i, idx in enumerate(selected):
            local_para = nets[idx].cpu().module.encoder.state_dict()
            if i == 0:
                for key in global_para:
                    global_para[key] = local_para[key] * fed_avg_freqs[i]
            else:
                for key in global_para:
                    global_para[key] += local_para[key] * fed_avg_freqs[i]

        global_model.module.encoder.load_state_dict(global_para)
        acc, top5 = compute_acc(test_dl_global, device, global_model, nn.CrossEntropyLoss())
        tracker.update_global_metrics(acc, top5)

        if acc >= args.target_acc:
            successful_rounds += 1
            logger.info(f"[FedProx] Target accuracy {args.target_acc:.2f}% reached at round {round} ({successful_rounds}/3).")
        else:
            successful_rounds = 0

        tracker.end_round()

    t_end = time.time()
    tracker.finalize(final_accuracy=acc)
    tracker.save_summary()

    logger.info("total communication time: %.2f seconds" % (t_end - t_start))
    logger.info("avg time per round: %.2f seconds" % ((t_end - t_start) / tracker.rounds_completed))


def run_spatl(args, device, logger, net_dataidx_map, test_dl_global):
    logger.info("Initializing nets for SPATL")
    nets, _, _ = init_nets(args.n_parties, args.model, args)
    global_models, _, _ = init_nets(1, args.model, args)
    global_model = global_models[0]
    global_para = global_model.module.encoder.state_dict()

    if args.is_same_initial:
        for net_id, net in nets.items():
            net.module.encoder.load_state_dict(global_para)

    t_start = time.time()
    lr = args.lr
    tracker = MetricsTracker(num_rounds=args.comm_round, log_dir=args.logdir)
    achieved_target = False

    for round in range(args.comm_round):
        if achieved_target:
            break

        tracker.start_round()
        logger.info("[SPATL] in comm round: " + str(round) + "#" * 100)

        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.sample)]

        for idx in selected:
            nets[idx].module.encoder.load_state_dict(global_para)

        prune = True
        local_update(nets, selected, args, net_dataidx_map, logger, lr, test_dl=None, device=device, Prune=prune)
        lr *= 0.99

        round_pruning = []
        round_flops = []
        round_comm_cost = 0

        for idx in selected:
            kept_idx = nets[idx].kept_idx
            flops = nets[idx].flops_reduction
            pruning_ratio = 1 - sum([len(v) for v in kept_idx.values()]) / sum([v.numel() for v in nets[idx].module.encoder.parameters()])
            round_pruning.append(pruning_ratio)
            round_flops.append(flops)
            round_comm_cost += sum([len(v) for v in kept_idx.values()])

        tracker.update_pruning_metrics(np.mean(round_pruning), np.mean(round_flops))
        tracker.update_comm_cost(round_comm_cost)

        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for i, idx in enumerate(selected):
            local_para = nets[idx].cpu().module.encoder.state_dict()
            if i == 0:
                for key in global_para:
                    global_para[key] = local_para[key] * fed_avg_freqs[i]
            else:
                for key in global_para:
                    global_para[key] += local_para[key] * fed_avg_freqs[i]

        global_model.module.encoder.load_state_dict(global_para)

        acc, top5 = compute_acc(test_dl_global, device, global_model, nn.CrossEntropyLoss())
        tracker.update_global_metrics(acc, top5)

        if acc >= args.target_acc:
            logger.info(f"[SPATL] Target accuracy {args.target_acc:.2f}% reached at round {round}.")
            achieved_target = True

        tracker.end_round()

    t_end = time.time()
    tracker.finalize(final_accuracy=acc)
    tracker.save_summary()

    logger.info("total communication time: %.2f seconds" % (t_end - t_start))
    logger.info("avg time per round: %.2f seconds" % ((t_end - t_start) / tracker.rounds_completed))

def run_spatl_xcGrad(args, device, logger, net_dataidx_map, test_dl_global):
    logger.info("Initializing nets for SPATL-XC Gradient")
    nets, _, _ = init_nets(args.n_parties, args.model, args)
    global_models, _, _ = init_nets(1, args.model, args)
    global_model = global_models[0]
    global_para = global_model.module.encoder.state_dict()

    if args.is_same_initial:
        for net_id, net in nets.items():
            net.module.encoder.load_state_dict(global_para)

    t_start = time.time()
    lr = args.lr
    tracker = MetricsTracker(num_rounds=args.comm_round, log_dir=args.logdir)
    achieved_target = False

    for round in range(args.comm_round):
        if achieved_target:
            break

        tracker.start_round()
        logger.info(f"[SPATL-XCGrad] in comm round: {round} " + "#" * 100)

        selected = np.random.choice(range(args.n_parties), int(args.n_parties * args.sample), replace=False)

        for idx in selected:
            nets[idx].module.encoder.load_state_dict(global_para)

        updated_nets, client_accs = local_update_shap(
            nets, selected, args, net_dataidx_map, logger, lr,
            test_dl=None, device=device, Prune=True
        )

        round_pruning, round_flops = [], []
        shap_entropy_list, shap_time_list = [], []
        comm_cost = 0

        for idx in selected:
            nets[idx] = updated_nets[idx]
            kept_idx = nets[idx].kept_idx
            flops = nets[idx].flops_reduction

            pruning_ratio = 1 - sum(len(v) for v in kept_idx.values()) / sum(p.numel() for p in nets[idx].module.encoder.parameters())
            round_pruning.append(pruning_ratio)
            round_flops.append(flops)
            comm_cost += sum(len(v) for v in kept_idx.values())

            entropy = getattr(nets[idx], 'shap_entropy', None)
            if entropy is not None:
                shap_entropy_list.append(entropy)

            shap_time = getattr(nets[idx], 'shap_time', None)
            if shap_time is not None:
                shap_time_list.append(shap_time)

        tracker.update_client_metrics(client_accs=client_accs)
        tracker.update_pruning_metrics(np.mean(round_pruning), np.mean(round_flops))
        tracker.update_comm_cost(comm_cost)
        if shap_entropy_list:
            tracker.update_shap_entropy(np.mean(shap_entropy_list))
        if shap_time_list:
            tracker.update_shap_time(np.mean(shap_time_list))

        # Aggregation
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for i, idx in enumerate(selected):
            local_para = nets[idx].cpu().module.encoder.state_dict()
            if i == 0:
                for key in global_para:
                    global_para[key] = local_para[key] * fed_avg_freqs[i]
            else:
                for key in global_para:
                    global_para[key] += local_para[key] * fed_avg_freqs[i]

        global_model.module.encoder.load_state_dict(global_para)
        acc, top5 = compute_acc(test_dl_global, device, global_model, nn.CrossEntropyLoss())
        tracker.update_global_metrics(acc, top5)

        if acc >= args.target_acc:
            logger.info(f"[SPATL-XCGrad] Target accuracy {args.target_acc:.2f}% reached at round {round}.")
            achieved_target = True

        tracker.end_round()

    t_end = time.time()
    tracker.finalize(final_accuracy=acc)
    tracker.save_summary()

    logger.info("total communication time: %.2f seconds" % (t_end - t_start))
    logger.info("avg time per round: %.2f seconds" % ((t_end - t_start) / tracker.rounds_completed))

def run_spatl_xcDeep(args, device, logger, net_dataidx_map, test_dl_global):
    logger.info("Initializing nets for SPATL-XC Deep")
    nets, _, _ = init_nets(args.n_parties, args.model, args)
    global_models, _, _ = init_nets(1, args.model, args)
    global_model = global_models[0]
    global_para = global_model.module.encoder.state_dict()

    if args.is_same_initial:
        for net_id, net in nets.items():
            net.module.encoder.load_state_dict(global_para)

    t_start = time.time()
    lr = args.lr
    tracker = MetricsTracker(num_rounds=args.comm_round, log_dir=args.logdir)
    achieved_target = False

    for round in range(args.comm_round):
        if achieved_target:
            break

        tracker.start_round()
        logger.info(f"[SPATL-XCDeep] in comm round: {round} " + "#" * 100)

        selected = np.random.choice(range(args.n_parties), int(args.n_parties * args.sample), replace=False)

        for idx in selected:
            nets[idx].module.encoder.load_state_dict(global_para)

        args.explainer_type = "deep"
        updated_nets, client_accs = local_update_shap(
            nets, selected, args, net_dataidx_map, logger, lr,
            test_dl=None, device=device, Prune=True
        )

        round_pruning, round_flops = [], []
        shap_entropy_list, shap_time_list = [], []
        comm_cost = 0

        for idx in selected:
            nets[idx] = updated_nets[idx]
            kept_idx = nets[idx].kept_idx
            flops = nets[idx].flops_reduction

            pruning_ratio = 1 - sum(len(v) for v in kept_idx.values()) / sum(p.numel() for p in nets[idx].module.encoder.parameters())
            round_pruning.append(pruning_ratio)
            round_flops.append(flops)
            comm_cost += sum(len(v) for v in kept_idx.values())

            entropy = getattr(nets[idx], 'shap_entropy', None)
            if entropy is not None:
                shap_entropy_list.append(entropy)

            shap_time = getattr(nets[idx], 'shap_time', None)
            if shap_time is not None:
                shap_time_list.append(shap_time)

        tracker.update_client_metrics(client_accs=client_accs)
        tracker.update_pruning_metrics(np.mean(round_pruning), np.mean(round_flops))
        tracker.update_comm_cost(comm_cost)
        if shap_entropy_list:
            tracker.update_shap_entropy(np.mean(shap_entropy_list))
        if shap_time_list:
            tracker.update_shap_time(np.mean(shap_time_list))

        # Aggregation
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for i, idx in enumerate(selected):
            local_para = nets[idx].cpu().module.encoder.state_dict()
            if i == 0:
                for key in global_para:
                    global_para[key] = local_para[key] * fed_avg_freqs[i]
            else:
                for key in global_para:
                    global_para[key] += local_para[key] * fed_avg_freqs[i]

        global_model.module.encoder.load_state_dict(global_para)
        acc, top5 = compute_acc(test_dl_global, device, global_model, nn.CrossEntropyLoss())
        tracker.update_global_metrics(acc, top5)

        if acc >= args.target_acc:
            logger.info(f"[SPATL-XCDeep] Target accuracy {args.target_acc:.2f}% reached at round {round}.")
            achieved_target = True

        tracker.end_round()

    t_end = time.time()
    tracker.finalize(final_accuracy=acc)
    tracker.save_summary()

    logger.info("total communication time: %.2f seconds" % (t_end - t_start))
    logger.info("avg time per round: %.2f seconds" % ((t_end - t_start) / tracker.rounds_completed))

def run_sccafold(args, device, logger, net_dataidx_map, test_dl_global):
    logger.info("Initializing nets for SCAFFOLD")
    nets, _, _ = init_nets(args.n_parties, args.model, args)
    global_models, _, _ = init_nets(1, args.model, args)
    global_model = global_models[0]
    global_para = global_model.module.encoder.state_dict()

    if args.is_same_initial:
        for net_id, net in nets.items():
            net.module.encoder.load_state_dict(global_para)

    # Initialize control variates
    c_global, _, _ = init_nets(1, args.model, args)
    c_global = c_global[0]
    c_nets = {}
    for idx in range(args.n_parties):
        c_local, _, _ = init_nets(1, args.model, args)
        c_nets[idx] = c_local[0]

    t_start = time.time()
    lr = args.lr
    tracker = MetricsTracker(num_rounds=args.comm_round, log_dir=args.logdir)
    achieved_target = False

    for round in range(args.comm_round):
        if achieved_target:
            break

        tracker.start_round()
        logger.info(f"[SCAFFOLD] in comm round: {round} " + "#" * 100)

        selected = np.random.choice(range(args.n_parties), int(args.n_parties * args.sample), replace=False)

        for idx in selected:
            nets[idx].module.encoder.load_state_dict(global_para)

        updated_nets = local_update_scaffold(
            nets, selected, global_model, c_nets, c_global,
            args, net_dataidx_map, logger,
            test_dl=test_dl_global, device=device, Prune=False
        )

        for idx in selected:
            nets[idx] = updated_nets[idx]

        # Aggregation
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for i, idx in enumerate(selected):
            local_para = nets[idx].cpu().module.encoder.state_dict()
            if i == 0:
                for key in global_para:
                    global_para[key] = local_para[key] * fed_avg_freqs[i]
            else:
                for key in global_para:
                    global_para[key] += local_para[key] * fed_avg_freqs[i]

        global_model.module.encoder.load_state_dict(global_para)
        acc, top5 = compute_acc(test_dl_global, device, global_model, nn.CrossEntropyLoss())
        tracker.update_global_metrics(acc, top5)

        if acc >= args.target_acc:
            logger.info(f"[SCAFFOLD] Target accuracy {args.target_acc:.2f}% reached at round {round}.")
            achieved_target = True

        tracker.end_round()

    t_end = time.time()
    tracker.finalize(final_accuracy=acc)
    tracker.save_summary()

    logger.info("total communication time: %.2f seconds" % (t_end - t_start))
    logger.info("avg time per round: %.2f seconds" % ((t_end - t_start) / tracker.rounds_completed))


if __name__ == '__main__':

    args = get_parameter()

    # Create log directory if it doesn't exist
    mkdirs(args.logdir)
    
    # Set log file name
    if args.log_file_name is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.log_file_name = f"{args.model}_{args.n_parties}_sample{args.sample}_{args.alg}_experiment_log-{timestamp}"
    log_path = os.path.join(args.logdir, args.log_file_name + '.log')
    
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create handlers
    file_handler = logging.FileHandler(log_path, mode='w')
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Force flushing after each log message
    file_handler.flush = sys.stdout.flush
    console_handler.flush = sys.stdout.flush
    
    # Save args to JSON
    argument_path = os.path.join(args.logdir, args.log_file_name + "_args.json")
    with open(argument_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Confirm device and start logging
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Partitioning data")

    '''
    prepare data
    '''
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    n_classes = len(np.unique(y_train))
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.batch_size,
                                                                                      32)
    data_size = len(test_ds_global)

    train_all_in_list = []
    test_all_in_list = []

    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)



    if args.alg == 'fedavg':
        run_fedavg(args, device, logger, net_dataidx_map, test_dl_global)
    elif args.alg == 'fedprox':
        run_fedprox(args, device, logger, net_dataidx_map, test_dl_global)
    elif args.alg == 'spatl':
        run_spatl(args, device, logger, net_dataidx_map, test_dl_global)
    elif args.alg == 'spatl-xcGrad':
        run_spatl_xcGrad(args, device, logger, net_dataidx_map, test_dl_global)
    elif args.alg == 'spatl-xcDeep':
        run_spatl_xcDeep(args, device, logger, net_dataidx_map, test_dl_global)
    elif args.alg == 'sccafold':
        run_sccafold(args, device, logger, net_dataidx_map, test_dl_global)

    else:
        print('Algorithm not found!!!!')

    


