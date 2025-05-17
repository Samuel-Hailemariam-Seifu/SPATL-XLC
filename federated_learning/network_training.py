import copy

import torch
from sklearn.metrics import confusion_matrix
from torch import optim, nn
import numpy as np
from pruning_head.gnnrl_network_pruning import gnnrl_pruning
from pruning_head.graph_env.network_pruning import channel_pruning
from utils.accuracy import compute_acc
from utils.data.prepare_data import get_dataloader
from utils.loss import LossCalculator
from torch.nn.utils import prune
import shap
from copy import deepcopy
import torch.nn as nn

import time
import datetime

################################fedavg################################
def train_net_fedavg(net_id, net, train_dataloader, test_dataloader, epochs, lr, optimizer_name, logger, args):
    import torch
    from torch import nn, optim

    device = args.device if hasattr(args, "device") else "cpu"
    logger.info(f"[Client {net_id}] Training on {len(train_dataloader.dataset)} samples for {epochs} epochs")

    criterion = nn.CrossEntropyLoss().to(device)
    net.to(device)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=args.reg)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args.rho, weight_decay=args.reg)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Pre-training accuracy
    pre_train_acc, _ = compute_acc(train_dataloader, device, net, criterion)
    pre_test_acc, _ = compute_acc(test_dataloader, device, net, criterion)

    for epoch in range(epochs):
        net.train()
        total_loss = 0.0
        total_samples = 0
    
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = net(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item() * batch_x.size(0)  # accumulate total loss over all samples
            total_samples += batch_x.size(0)
    
        avg_loss = total_loss / total_samples
        logger.info(f"[Client {net_id}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Post-training accuracy
    train_acc, _ = compute_acc(train_dataloader, device, net, criterion)
    test_acc, _ = compute_acc(test_dataloader, device, net, criterion)

    logger.info(f"[Client {net_id}] Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f}")

    return train_acc, test_acc, pre_train_acc, pre_test_acc


################################fedprox################################

def local_update_fedprox(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, optimizer_type, logger, args, device="cpu", mu=0.01):
    logger.info('Training network %s [FedProx]' % str(net_id))

    criterion = nn.CrossEntropyLoss().to(device)
    pre_train_acc, _ = compute_acc(train_dataloader, device, net, criterion)
    pre_test_acc, _ = compute_acc(test_dataloader, device, net, criterion)

    logger.info('>> Pre-Training Training accuracy: %.2f' % (pre_train_acc/100))
    logger.info('>> Pre-Training Test accuracy: %.2f' % (pre_test_acc/100))

    if optimizer_type == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif optimizer_type == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)

    global_params = deepcopy(global_model.module.encoder.state_dict())
    net.train()

    for epoch in range(epochs):
        for x, target in train_dataloader:
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            output = net(x)
            loss = criterion(output, target)

            # === FedProx Proximal Term ===
            prox_term = 0.0
            for name, param in net.module.encoder.named_parameters():
                prox_term += ((param - global_params[name].to(device)) ** 2).sum()
            loss += (mu / 2) * prox_term

            loss.backward()
            optimizer.step()

        logger.info(f'[Client {net_id}] Epoch {epoch} Loss: {loss.item():.4f}')

    train_acc, _ = compute_acc(train_dataloader, device, net, criterion)
    test_acc, _ = compute_acc(test_dataloader, device, net, criterion)

    logger.info('>> Training accuracy: %.2f' % train_acc)
    logger.info('>> Test accuracy: %.2f' % test_acc)
    return train_acc, test_acc, pre_train_acc, pre_test_acc

################################spatl-xc################################

def local_update_shap(nets, selected, args, net_dataidx_map, logger, lr=0.01, test_dl=None, device="cpu", Prune=True):
    client_accs = []
    avg_acc = 0.0
    pre_avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        dataidxs = net_dataidx_map[net_id]
        noise_level = args.noise if net_id != args.n_parties - 1 else 0

        # Get dataloaders
        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(
                args.dataset, args.datadir, args.batch_size, 32,
                dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            adjusted_noise = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(
                args.dataset, args.datadir, args.batch_size, 32,
                dataidxs, adjusted_noise)

        net.to(device)
        n_epoch = args.epochs

        logger.info(f"[Client {net_id}] Training for {n_epoch} epochs")
        train_acc, test_acc, pre_train_acc, pre_test_acc = train_net(
            net_id, net, train_dl_local, test_dl_local,
            n_epoch, lr, args.optimizer, logger, args, device=device)

        logger.info(f"[Client {net_id}] Final Test Accuracy: {test_acc:.2f}")

        if Prune:
            logger.info(f"[Client {net_id}] Starting SHAP-based pruning")
            pruned_state, kept_indices, pruning_info = local_update_shap_pruning(
                model=net,
                dataloader=train_dl_local,
                logger=logger,
                loss_fn=torch.nn.CrossEntropyLoss(),
                device=device,
                local_epochs=0,  # already trained
                batch_size=args.batch_size,
                net_id=net_id,
                args=args
            )
            net.module.encoder.load_state_dict(pruned_state)
            net.kept_idx = kept_indices
            net.flops_reduction = pruning_info['flops']
            logger.info(f"[Client {net_id}] Pruned: {pruning_info['pruning'] * 100:.2f}%, FLOPs: {pruning_info['flops'] * 100:.2f}%")

        # Track client accuracy
        criterion = torch.nn.CrossEntropyLoss().to(device)
        acc, _ = compute_acc(test_dl_local, device, net, criterion)
        client_accs.append(acc)

        avg_acc += test_acc
        pre_avg_acc += pre_test_acc

    avg_acc /= len(selected)
    pre_avg_acc /= len(selected)

    logger.info(f"[Round Summary] Pre-update Avg Test Accuracy: {pre_avg_acc:.4f}")
    logger.info(f"[Round Summary] Post-update Avg Test Accuracy: {avg_acc:.4f}")

    return list(nets.values()), client_accs

def local_update_shap_pruning(model, dataloader, logger, loss_fn, device, 
                       local_epochs=0, batch_size=32, net_id=None, args=None):
    import shap
    import time
    from copy import deepcopy

    model.to(device)
    model.eval()

    # === Step 1: Prepare SHAP background and input ===
    local_data = list(dataloader)
    background = torch.cat([local_data[i][0] for i in range(min(len(local_data), 5))]).to(device)
    test_batch = local_data[0][0][:batch_size].to(device)

    # === Step 2: Choose Explainer with fallback ===
    explainer_type = getattr(args, "explainer_type", "gradient").lower()

    try:
        if explainer_type == "deep":
            explainer = shap.DeepExplainer(model, background)
            logger.info(f"[Client {net_id}] Using DeepExplainer")
        else:
            explainer = shap.GradientExplainer(model, background)
            logger.info(f"[Client {net_id}] Using GradientExplainer")
    except Exception as e:
        logger.warning(f"[Client {net_id}] {explainer_type.capitalize()}Explainer failed: {e}. Falling back to GradientExplainer.")
        try:
            explainer = shap.GradientExplainer(model, background)
            logger.info(f"[Client {net_id}] Fallback to GradientExplainer succeeded.")
        except Exception as fallback_e:
            logger.error(f"[Client {net_id}] GradientExplainer also failed: {fallback_e}")
            raise RuntimeError("Both SHAP explainers failed.")

    # === Step 3: SHAP Computation ===
    start_time = time.time()
    shap_vals = explainer.shap_values(test_batch)
    elapsed_time = time.time() - start_time

    if isinstance(shap_vals, list):
        shap_vals = np.mean(np.abs(np.stack(shap_vals)), axis=0)

    # === Step 4: Dynamic Pruning Ratio ===
    n_samples = len(local_data) * batch_size
    max_samples = 10000
    ratio = combined_dynamic_ratio(shap_vals, model, n_samples, max_samples, logger=logger)

    logger.info(f"[Client {net_id}] SHAP-guided pruning ratio: {ratio:.4f}")

    # === Step 5: Compute Importance Scores ===
    encoder_params = dict(model.module.encoder.named_parameters())
    importance_scores = {}
    for name, param in encoder_params.items():
        if param.requires_grad and param.dim() >= 2:
            score = param.abs().detach().cpu().numpy()
            score_per_unit = np.mean(score, axis=tuple(range(1, param.dim())))
            importance_scores[name] = score_per_unit

    # === Step 6: Apply Pruning ===
    pruned_state = deepcopy(model.module.encoder.state_dict())
    kept_indices = {}

    for name, scores in importance_scores.items():
        param = encoder_params[name]
        k = int((1 - ratio) * scores.shape[0])
        topk_idx = np.argsort(scores)[-k:]
        kept_indices[name] = topk_idx

        mask = torch.zeros_like(param)
        if param.dim() == 4:
            mask[topk_idx, :, :, :] = 1
        elif param.dim() == 2:
            mask[topk_idx, :] = 1
        elif param.dim() == 1:
            mask[topk_idx] = 1

        pruned_state[name] = (param * mask.to(param.device)).detach()

    model.module.encoder.load_state_dict(pruned_state)

    # === Step 7: FLOPs and Params Reduction ===
    total_params, kept_params, total_flops, kept_flops = 0, 0, 0, 0
    for name, param in model.module.encoder.named_parameters():
        num_total = param.shape[0]
        shape = param.shape
        is_pruned = name in kept_indices
        num_kept = len(kept_indices[name]) if is_pruned else num_total

        total_params += param.numel()
        kept_params += (num_kept * int(np.prod(shape[1:]))) if param.dim() >= 2 else num_kept
        flops_total = 2 * param.numel() if param.dim() >= 2 else param.numel()
        flops_kept = 2 * (num_kept * int(np.prod(shape[1:]))) if param.dim() >= 2 else num_kept

        total_flops += flops_total
        kept_flops += flops_kept

    param_prune_ratio = 1 - (kept_params / total_params)
    flops_prune_ratio = 1 - (kept_flops / total_flops)

    logger.info(f"[Client {net_id}] Final param reduction: {param_prune_ratio * 100:.2f}%")
    logger.info(f"[Client {net_id}] Final FLOPs reduction: {flops_prune_ratio * 100:.2f}%")

    # === Step 8: SHAP Metrics ===
    flat_shap = np.abs(shap_vals).reshape(-1)
    shap_dist = flat_shap / (np.sum(flat_shap) + 1e-8)
    entropy = scipy.stats.entropy(shap_dist)

    model.shap_entropy = entropy
    model.shap_time = elapsed_time

    return pruned_state, kept_indices, {
        'pruning': param_prune_ratio,
        'flops': flops_prune_ratio
    }


def shap_entropy_based_ratio(shap_values, base_ratio=0.2, min_ratio=0.05, max_ratio=0.5, logger=None):
    import numpy as np
    import scipy.stats

    # Step 1: Ensure (B, N) shape or reduce
    if isinstance(shap_values, list):
        shap_values = np.mean(np.abs(np.stack(shap_values)), axis=0)  # (B, N)
    elif shap_values.ndim > 2:
        shap_values = np.mean(np.abs(shap_values), axis=0)            # (B, N) or (N,)

    # Step 2: Flatten across all axes → importance for each feature
    flat_shap = np.abs(shap_values).reshape(-1)                       # Flatten to 1D

    # Step 3: Normalize into a probability distribution
    shap_dist = flat_shap / (np.sum(flat_shap) + 1e-8)

    # Step 4: Scalar entropy
    entropy = scipy.stats.entropy(shap_dist)

    # Step 5: Normalize entropy and get pruning ratio
    entropy_norm = entropy / np.log(len(shap_dist) + 1e-8)
    ratio = base_ratio * (1 - entropy_norm)
    ratio = float(np.clip(ratio, min_ratio, max_ratio))

    if logger:
        logger.info(f"[SHAP Entropy] Entropy={entropy:.4f}, Normalized={entropy_norm:.4f}, Ratio={ratio:.4f}")

    return ratio


def gradient_norm_based_ratio(model, base_ratio=0.2, min_ratio=0.05, max_ratio=0.5):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += torch.norm(p.grad.detach(), p=2).item()
    total_norm = total_norm / (len(list(model.parameters())) + 1e-6)

    # Normalize with heuristic threshold (tune as needed)
    norm_ratio = min(1.0, total_norm / 10.0)
    ratio = base_ratio * (1 - norm_ratio)
    ratio = np.clip(ratio, min_ratio, max_ratio)
    return float(ratio)

def data_size_based_ratio(n_samples, max_samples, base_ratio=0.2, min_ratio=0.05, max_ratio=0.5):
    scale = n_samples / max_samples
    ratio = base_ratio * scale
    ratio = np.clip(ratio, min_ratio, max_ratio)
    return float(ratio)

def combined_dynamic_ratio(shap_vals, model, n_samples, max_samples,
                           base_ratio=0.2, min_ratio=0.05, max_ratio=0.5, logger=None):
    r1 = shap_entropy_based_ratio(shap_vals, base_ratio, min_ratio, max_ratio, logger)
    r2 = gradient_norm_based_ratio(model, base_ratio, min_ratio, max_ratio)
    r3 = data_size_based_ratio(n_samples, max_samples, base_ratio, min_ratio, max_ratio)
    # Weighted average
    combined = float(np.clip((r1 + r2 + r3) / 3, min_ratio, max_ratio))
    if logger:
        logger.info(f"[Dynamic Ratio] SHAP={r1:.4f}, GradNorm={r2:.4f}, DataSize={r3:.4f} → Combined={combined:.4f}")
    return combined



################################fedavg################################
def local_update(nets, selected, args, net_dataidx_map,logger, lr=0.01,test_dl = None, device="cpu", Prune=True):
    avg_acc = 0.0
    pre_avg_acc = 0.0
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs



        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)




        trainacc, testacc, pre_trainacc, pre_testacc = train_net(net_id, net, train_dl_local, test_dl_local, n_epoch, lr, args.optimizer, logger,args,device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))

        if Prune:
            logger.info("--------------------------------------Pruning network %s.--------------------------------------" % (str(net_id)))
            net,_,sparsity = gnnrl_pruning(net,logger, test_dl_local,args)
            logger.info("Flops ratio: %s." % (str(_)))
            logger.info("--------------------------------------End pruning %s.--------------------------------------" % (str(net_id)))
            for name, module in net.module.encoder.named_modules(): #remove mask
                if isinstance(module, nn.Conv2d):
                    module = prune.remove(module,name='weight')


        avg_acc += testacc
        pre_avg_acc += pre_testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)

    avg_acc /= len(selected)
    pre_avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)
    logger.info("avg test acc after aggregate %f" % pre_avg_acc)
    logger.info("avg test acc after local update %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

    # raise NotImplementedError

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer,logger,args, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    # pre_train_acc = compute_accuracy(net, train_dataloader, device=device)
    # pre_test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    criterion = nn.CrossEntropyLoss().to(device)
    pre_train_acc,_ = compute_acc(train_dataloader, device, net, criterion)
    pre_test_acc, _ = compute_acc(test_dataloader, device, net, criterion)

    logger.info('>> Pre-Training Training accuracy: {}'.format(pre_train_acc/100))
    logger.info('>> Pre-Training Test accuracy: {}'.format(pre_test_acc/100))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    # criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()
    loss_calculator = LossCalculator()


    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                # loss = criterion(out, target)
                loss = loss_calculator.calc_loss(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    train_acc,_ = compute_acc(train_dataloader, device, net, criterion)
    test_acc, _ = compute_acc(test_dataloader, device, net, criterion)
    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc, pre_train_acc,pre_test_acc


################################Scaffold notransfer################################
def local_update_scaffold_notransfer(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, logger, test_dl = None, device="cpu", Prune=True):
    avg_acc = 0.0
    pre_avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.module.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]


        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)




        trainacc, testacc,  pre_trainacc, pre_testacc, c_delta_para = train_net_scaffold_notransfer(net_id, net, global_model,c_nets[net_id], c_global, train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, logger, args, device=device)

        if Prune:
            # env.reset()
            # env.best_pruned_model=None
            # env.model = net
            logger.info("--------------------------------------Pruning network %s.--------------------------------------" % (str(net_id)))
            # net,_ = gnnrl_pruning(net, env,logger,args)
            net,_,sparsity = gnnrl_pruning(net,logger,test_dl_local,args)
            logger.info("Flops ratio: %s." % (str(_)))
            logger.info("Sparcity of salient parameters: %s." % (str(sparsity)))
            logger.info("--------------------------------------End pruning %s.--------------------------------------" % (str(net_id)))
            for name, module in net.module.named_modules(): #remove mask
                if isinstance(module, nn.Conv2d):
                    module = prune.remove(module,name='weight')

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]


        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        pre_avg_acc += pre_testacc

    for key in total_delta:
        total_delta[key] /= len(selected)
    c_global_para = c_global.module.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.module.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    pre_avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)
    logger.info("avg test acc after aggregate %f" % pre_avg_acc)
    logger.info("avg test acc after local update %f" % avg_acc)


    nets_list = list(nets.values())
    return nets_list
def train_net_scaffold_notransfer(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer,logger, args, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    criterion = nn.CrossEntropyLoss().to(device)
    pre_train_acc,_ = compute_acc(train_dataloader, device, net, criterion)
    pre_test_acc, _ = compute_acc(test_dataloader, device, net, criterion)

    logger.info('>> Pre-Training Training accuracy: {}'.format(pre_train_acc/100))
    logger.info('>> Pre-Training Test accuracy: {}'.format(pre_test_acc/100))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    # criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()


    c_global_para = c_global.module.state_dict()
    c_local_para = c_local.module.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.module.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])*0.0005
                net.module.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))



    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    c_new_para = c_local.module.state_dict()
    c_delta_para = copy.deepcopy(c_local.module.state_dict())
    global_model_para = global_model.module.state_dict()
    net_para = net.module.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.module.load_state_dict(c_new_para)


    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc, pre_train_acc,pre_test_acc, c_delta_para


################################Scaffold################################
def local_update_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, logger, test_dl = None, device="cpu", Prune=True):
    avg_acc = 0.0
    pre_avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.module.encoder.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]


        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)




        trainacc, testacc,  pre_trainacc, pre_testacc, c_delta_para = train_net_scaffold(net_id, net, global_model,c_nets[net_id], c_global, train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, logger, args, device=device)

        if Prune:
            # env.reset()
            # env.best_pruned_model=None
            # env.model = net
            logger.info("--------------------------------------Pruning network %s.--------------------------------------" % (str(net_id)))
            # net,_ = gnnrl_pruning(net, env,logger,args)
            net,_,sparsity = gnnrl_pruning(net,logger,test_dl_local,args)
            logger.info("Flops ratio: %s." % (str(_)))
            logger.info("Sparcity of salient parameters: %s." % (str(sparsity)))
            logger.info("--------------------------------------End pruning %s.--------------------------------------" % (str(net_id)))
            for name, module in net.module.named_modules(): #remove mask
                if isinstance(module, nn.Conv2d):
                    module = prune.remove(module,name='weight')

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]


        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        pre_avg_acc += pre_testacc

    for key in total_delta:
        total_delta[key] /= len(selected)
    c_global_para = c_global.module.encoder.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.module.encoder.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    pre_avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)
    logger.info("avg test acc after aggregate %f" % pre_avg_acc)
    logger.info("avg test acc after local update %f" % avg_acc)


    nets_list = list(nets.values())
    return nets_list

def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer,logger, args, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    criterion = nn.CrossEntropyLoss().to(device)
    pre_train_acc,_ = compute_acc(train_dataloader, device, net, criterion)
    pre_test_acc, _ = compute_acc(test_dataloader, device, net, criterion)

    logger.info('>> Pre-Training Training accuracy: {}'.format(pre_train_acc/100))
    logger.info('>> Pre-Training Test accuracy: {}'.format(pre_test_acc/100))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    # criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()


    c_global_para = c_global.module.encoder.state_dict()
    c_local_para = c_local.module.encoder.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.module.encoder.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])/100
                net.module.encoder.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))



    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    c_new_para = c_local.module.encoder.state_dict()
    c_delta_para = copy.deepcopy(c_local.module.encoder.state_dict())
    global_model_para = global_model.module.encoder.state_dict()
    net_para = net.module.encoder.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.module.encoder.load_state_dict(c_new_para)


    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc, pre_train_acc,pre_test_acc, c_delta_para



def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device, dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix

    return correct / float(total)

