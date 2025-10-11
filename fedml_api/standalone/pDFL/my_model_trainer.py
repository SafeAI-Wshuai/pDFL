import copy
import logging
import time
import pdb
import numpy as np
import torch
from torch import nn
from fedml_api.model.cv.cnn_meta import Meta_net
import torch.nn.functional as F
from fedml_api.utils.cosScheduler import CosineSchedulerPair

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer
def infer_head_prefixes(model: nn.Module, dataset: str, tail_linear: int = 1):
    """
    返回一组前缀，表示哪些子模块属于“head”
    - emnist: output_layer
    - resnet: fc
    - vgg: classifier
    - 否则：自动取“最后的 tail_linear 个 nn.Linear 模块”的名字（如 fc3 / fc2 等）
    """
    # 1) 兼容你现有的特殊命名
    if dataset == "emnist":
        return ["output_layer"]
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        return ["fc"]
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        return ["classifier"]

    # 2) 自动探测：收集所有 nn.Linear 子模块的名字（保持拓扑顺序）
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_names.append(name)

    # 没有 linear（极少见），回退到 'linear' 关键词
    if not linear_names:
        return ["linear"]

    # 取最后 tail_linear 个作为 head
    tail_linear = max(1, min(tail_linear, len(linear_names)))
    return linear_names[-tail_linear:]


def is_head_name(name: str, dataset: str, model: nn.Module, tail_linear: int = 1):
    """
    给定 state_dict 的键 name，判断它是否属于 head 参数
    """
    head_prefixes = infer_head_prefixes(model, dataset, tail_linear=tail_linear)
    # state_dict 键形如 'fc3.weight'，所以判断前缀 + '.'
    return any(name.startswith(p + ".") for p in head_prefixes)

def split_head_body_params(model: nn.Module, dataset: str, tail_linear: int = 1):
    head_prefixes = set(infer_head_prefixes(model, dataset, tail_linear))
    body, head = [], []
    for n, p in model.named_parameters():
        if any(n.startswith(pref + ".") for pref in head_prefixes):
            head.append(p)
        else:
            body.append(p)
    return body, head


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args=args
        self.logger = logger
        self.body_params = [p  for name, p in model.named_parameters() if 'linear' not in name]
        self.head_params = [p  for name, p in model.named_parameters() if 'linear' in name]

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters,strict=False) 

    def get_trainable_params(self):
        dict= {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    def train(self, train_data, test_data, device, args, round, teachers=None, attn=None, kd_mode="uni"):

        # test
        test_local_metrics = self.test(test_data,device,args)
        p_test_acc = np.array(test_local_metrics['test_correct']) / np.array(test_local_metrics['test_total'])
        self.logger.info('acc_before: {:.5f}'.format(p_test_acc))

        # train
        model = self.model
        # ---- KD: 构建教师模型列表（仅用于前向蒸馏，不参与训练）----
        teacher_models = []
        if teachers is not None and len(teachers) > 0:
            for sd in teachers:
                t_model = copy.deepcopy(self.model).to(device)
                t_model.load_state_dict(sd, strict=False)
                t_model.eval()
                for p in t_model.parameters():
                    p.requires_grad_(False)
                teacher_models.append(t_model)
        # α 注意力权重张量
        alpha = None
        if attn is not None:
            alpha = attn.detach().to(device)
            if alpha.dim() == 0:  # 单值时
                alpha = alpha.unsqueeze(0)

        model = self.model
        model.to(device)
        model.train()

        # ---- KD 计算：单向 & 双向（保持之前实现）----
        def _kd_loss_uni(student_logits, xs, T=2.0, alpha_vec=None):
            if len(teacher_models) == 0:
                return torch.tensor(0.0, device=student_logits.device)
            with torch.no_grad():
                logits_list = [tm(xs) for tm in teacher_models]
                stacked = torch.stack(logits_list, dim=0)  # [K,B,C]
                if alpha_vec is None or alpha_vec.numel() != stacked.size(0):
                    alpha_use = torch.full((stacked.size(0),), 1.0 / stacked.size(0), device=stacked.device)
                else:
                    alpha_use = (alpha_vec.to(stacked.device)).clamp_min(1e-12)
                    alpha_use = alpha_use / alpha_use.sum()
                q_list = [F.softmax(stacked[k] / T, dim=1) for k in range(stacked.size(0))]
                q_teacher = torch.stack(q_list, dim=0)
                q_teacher = (alpha_use.view(-1, 1, 1) * q_teacher).sum(dim=0)
            log_p = F.log_softmax(student_logits / T, dim=1)
            return F.kl_div(log_p, q_teacher, reduction="batchmean") * (T * T)

        def _kd_loss_bi(student_logits, xs, T=2.0, alpha_vec=None):
            if len(teacher_models) == 0:
                return torch.tensor(0.0, device=student_logits.device)
            with torch.no_grad():
                logits_list = [tm(xs) for tm in teacher_models]
                stacked = torch.stack(logits_list, dim=0)
                if alpha_vec is None or alpha_vec.numel() != stacked.size(0):
                    alpha_use = torch.full((stacked.size(0),), 1.0 / stacked.size(0), device=stacked.device)
                else:
                    alpha_use = (alpha_vec.to(stacked.device)).clamp_min(1e-12)
                    alpha_use = alpha_use / alpha_use.sum()
                q_list = [F.softmax(stacked[k] / T, dim=1) for k in range(stacked.size(0))]
                q_teacher = torch.stack(q_list, dim=0).mul_(alpha_use.view(-1, 1, 1)).sum(dim=0)
            p_log = F.log_softmax(student_logits / T, dim=1)
            p = p_log.exp()
            kl_pq = F.kl_div(p_log, q_teacher, reduction="batchmean")
            log_q = torch.log(q_teacher.clamp_min(1e-12))
            kl_qp = F.kl_div(log_q, p, reduction="batchmean")
            return 0.5 * (kl_pq + kl_qp) * (T * T)

        def _kd_loss(student_logits, xs, T, alpha_vec):
            if kd_mode == "bi":
                return _kd_loss_bi(student_logits, xs, T, alpha_vec)
            else:
                return _kd_loss_uni(student_logits, xs, T, alpha_vec)

        criterion = nn.CrossEntropyLoss().to(device)

        # ========= 新增：一阶段分支 =========
        if getattr(args, "one_stage", 0) == 1:
            # 统一训练整网，不拆 head/body；epoch 数与 lr 可单独指定
            total_epochs = args.one_stage_epochs
            if total_epochs is None:
                total_epochs = int(args.head_epochs + args.body_epochs)
            base_lr = args.lr_one_stage if args.lr_one_stage is not None else args.lr_body

            # 所有参数都训练
            for p in model.parameters():
                p.requires_grad = True

            optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                        momentum=args.momentum, weight_decay=args.wd)
            # 余弦衰减（每阶段内部版本）
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_epochs), eta_min=base_lr * 1e-2
            )

            for epoch in range(total_epochs):
                epoch_loss = []
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    model.zero_grad()
                    logits = model(x)
                    ce = criterion(logits, labels.long())
                    T = getattr(args, "kd_tau", 2.0)
                    lam = getattr(args, "kd_lambda", 0.3)
                    kd = _kd_loss(logits, x, T=T, alpha_vec=attn.to(device) if attn is not None else None) \
                        if len(teacher_models) > 0 else 0.0
                    loss = ce + lam * kd
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    optimizer.step()
                    epoch_loss.append(loss.item())

                scheduler.step()
                cid = getattr(self, "id", getattr(self, "client_idx", -1))
                self.logger.info(
                    f'[OneStage] Client={cid}\tEpoch={epoch}\tLR={scheduler.get_last_lr()[0]:.6e}\tLoss={np.mean(epoch_loss):.6f}')

            return test_local_metrics
        # ========= 旧的两阶段逻辑（原样保留） =========

        # （下面保持你现有的 head→body 两段训练 + KD 逻辑不变）
        body_params, head_params = split_head_body_params(model, self.args.dataset, tail_linear=1)
        if len(head_params) == 0:
            raise RuntimeError("Head params is empty. Check model naming (fc/classifier/output_layer/linear).")

        # 先 head
        for p in body_params: p.requires_grad = False
        for p in head_params: p.requires_grad = True
        head_optimizer = torch.optim.SGD(head_params, lr=args.lr_head, momentum=args.momentum, weight_decay=args.wd)
        head_sched = torch.optim.lr_scheduler.CosineAnnealingLR(head_optimizer, T_max=max(1, args.head_epochs),
                                                                eta_min=args.lr_head * 1e-2)
        for epoch in range(args.head_epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                logits = model(x)
                ce = criterion(logits, labels.long())
                T = getattr(args, "kd_tau", 2.0);
                lam = getattr(args, "kd_lambda", 0.3)
                kd = _kd_loss(logits, x, T=T, alpha_vec=attn.to(device) if attn is not None else None) \
                    if len(teacher_models) > 0 else 0.0
                loss = ce + lam * kd
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_params, 10)
                head_optimizer.step()
                epoch_loss.append(loss.item())
            head_sched.step()
            cid = getattr(self, "id", getattr(self, "client_idx", -1))
            self.logger.info(
                f'Client={cid}\t[Head] Epoch={epoch}\tLR={head_sched.get_last_lr()[0]:.6e}\tLoss={np.mean(epoch_loss):.6f}')

        # 再 body
        for p in body_params: p.requires_grad = True
        for p in head_params: p.requires_grad = False
        body_optimizer = torch.optim.SGD(body_params, lr=args.lr_body, momentum=args.momentum, weight_decay=args.wd)
        body_sched = torch.optim.lr_scheduler.CosineAnnealingLR(body_optimizer, T_max=max(1, args.body_epochs),
                                                                eta_min=args.lr_body * 1e-2)
        for epoch in range(args.body_epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                logits = model(x)
                ce = criterion(logits, labels.long())
                T = getattr(args, "kd_tau", 2.0);
                lam = getattr(args, "kd_lambda", 0.3)
                kd = _kd_loss(logits, x, T=T, alpha_vec=attn.to(device) if attn is not None else None) \
                    if len(teacher_models) > 0 else 0.0
                loss = ce + lam * kd
                loss.backward()
                torch.nn.utils.clip_grad_norm_(body_params, 10)
                body_optimizer.step()
                epoch_loss.append(loss.item())
            body_sched.step()
            cid = getattr(self, "id", getattr(self, "client_idx", -1))
            self.logger.info(
                f'Client={cid}\t[Body] Epoch={epoch}\tLR={body_sched.get_last_lr()[0]:.6e}\tLoss={np.mean(epoch_loss):.6f}')

        return test_local_metrics

    def train_fedavg(self, train_data, test_data, device, args, round):
        # 先测一遍
        test_local_metrics = self.test(test_data, device, args)
        p_test_acc = np.array(test_local_metrics['test_correct']) / np.array(test_local_metrics['test_total'])
        self.logger.info('acc_before(FedAvg): {:.5f}'.format(p_test_acc))

        model = self.model.to(device).train()
        criterion = nn.CrossEntropyLoss().to(device)

        # 复用已有超参，避免再加新参数
        lr = getattr(args, "lr_body", 0.01)
        epochs = getattr(args, "body_epochs", 1)
        momentum = getattr(args, "momentum", 0.9)
        weight_decay = getattr(args, "wd", 5e-4)
        lr_decay = getattr(args, "lr_decay", 1.0)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr * (lr_decay ** round),
                                    momentum=momentum, weight_decay=weight_decay)

        for epoch in range(epochs):
            epoch_loss = []
            for batch_idx, (x, y) in enumerate(train_data):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                epoch_loss.append(loss.item())

            cid = getattr(self, "id", getattr(self, "client_idx", -1))
            self.logger.info(f'[FedAvg] Client={cid}\tEpoch={epoch}\tLoss={np.mean(epoch_loss):.6f}')

        return test_local_metrics

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

