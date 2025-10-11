import copy
import logging
import math
import time
import pdb
import numpy as np
import torch
import numpy as n
from torch import nn
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

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer, logger):
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        # 查看本地样本数量
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number



    def train(self, w_global,w_local,round):

        def is_head_name(name: str, dataset: str, model: nn.Module):
            if dataset == "emnist":
                return "output_layer" in name
            if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
                return name.startswith("fc.")
            if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
                return name.startswith("classifier.")
            return "linear" in name  # 回退

        body_params = {k: v for k, v in w_global.items() if
                       not is_head_name(k, self.args.dataset, self.model_trainer.model)}
        head_params = {k: v for k, v in w_local.items() if is_head_name(k, self.args.dataset, self.model_trainer.model)}

        self.model_trainer.set_model_params(body_params)
        self.model_trainer.set_model_params(head_params)
        self.model_trainer.set_id(self.client_idx)
        
        test_local_metrics = self.model_trainer.train \
            (self.local_training_data, self.local_test_data,self.device, self.args, round)
        weights = self.model_trainer.get_model_params()
        return weights,test_local_metrics

    def local_test(self, w, b_use_test_dataset = True):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
    
