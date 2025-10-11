import copy
import logging
import time
import pdb
import numpy as np
import torch
from torch import nn
from fedml_api.model.cv.cnn_meta import Meta_net

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

    def train(self, train_data,test_data,  device,  args, round):
  
        # test 
        test_local_metrics = self.test(test_data,device,args)
        p_test_acc = np.array(test_local_metrics['test_correct']) / np.array(test_local_metrics['test_total'])
        self.logger.info('acc_before: {:.5f}'.format(p_test_acc))
        
        # train 
        model = self.model
        model.to(device)
        model.train()

        body_params, head_params = split_head_body_params(model, self.args.dataset)
        if len(head_params) == 0:
            # 防御性断言，避免空参数列表
            raise RuntimeError("Head params is empty. Check model naming (fc/classifier/output_layer/linear).")

        criterion = nn.CrossEntropyLoss().to(device)
        """
        先冻结 body，只训练 head → 让分类器快速适应本地数据，实现个性化。
        再冻结 head，只训练 body → 更新共享特征表征，使其逐渐在各客户端间收敛。
        """
        ## train head parameters
        for param in body_params:
            param.requires_grad = False 
        for param in head_params:
            param.requires_grad = True
        head_optimizer = torch.optim.SGD(head_params,lr=args.lr_head* (args.lr_decay**round), 
                                         momentum=args.momentum,weight_decay=args.wd)
        for epoch in range(args.head_epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                head_optimizer.step()
                epoch_loss.append(loss.item())
                
            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
           
           
        ## training body parameters
        for param in body_params:
            param.requires_grad = True 
        for param in head_params:
            param.requires_grad = False
        body_optimizer = torch.optim.SGD(body_params,lr=args.lr_body* (args.lr_decay**round),
                                         momentum=args.momentum,weight_decay=args.wd)
        for epoch in range(args.body_epochs):   
            epoch_loss = []     
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                body_optimizer.step()
                epoch_loss.append(loss.item())

            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
        
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

