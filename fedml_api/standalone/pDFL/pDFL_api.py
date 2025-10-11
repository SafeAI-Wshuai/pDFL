import copy
import logging
import pickle
import random

import numpy as np
import torch
import pdb
import torch.nn.functional as F
import os
import gc

from fedml_api.standalone.pDFL.client import Client
from fedml_api.model.cv.resnet import  customized_resnet18, tiny_resnet18


class pDFLAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger, log_path):
        self.logger = logger
        self.log_path = log_path  # 保存日志目录
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()
        self.client_sample_nums = [dataset[4][cid]
                                   for cid in range(self.args.client_num_in_total)]

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()  # get_model_params
        w_locals = []
  
        for clnt in range(self.args.client_num_in_total):
            w_locals.append(copy.deepcopy(w_global))  # distribute models to each client

        acc_locals = []
        loss_locals = []
        for round_idx in range(self.args.comm_round): 
            print("################Communication round : {}".format(round_idx))
            self.logger.info("################Communication round : {}".format(round_idx))
            
            ## initialize metrics 
            p_test_metrics_before = {
            'num_samples': [],
            'num_correct': [],
            'losses': []}

            w_locals_record = copy.deepcopy(w_locals)
            # 检查下是否循环100轮，把每个client都训练， 按理说只选取10个来训练啊，但是10个也要保证他们的邻居也训练了
            for clnt_idx in range(self.args.client_num_in_total): 
                self.logger.info('##--->Training Client: {} at Round: {}'.format(clnt_idx, round_idx))
                nei_indexs = []
                
                # select neighbours
                nei_indexs = self._benefit_choose(round_idx, clnt_idx, self.args.client_num_in_total,\
                                                  self.args.client_num_per_round, self.args.cs)
                if self.args.client_num_in_total != self.args.client_num_per_round:
                    nei_indexs = np.append(nei_indexs, clnt_idx)
                nei_indexs = np.sort(nei_indexs) 
                if self.args.cs != "full":
                    self.logger.info("client_indexes = " + str(nei_indexs))
                else:
                    self.logger.info("Choose all clients aka FULLY CONNECTED!")

                if self.args.mode == "pdfL":
                    # 1) 注意力 + 加权聚合（仅 body），head 本地
                    alpha = self._compute_neighbor_attention(clnt_idx, nei_indexs, w_locals_record,
                                                             tau=getattr(self.args, "attn_tau", 2.0))
                    self.logger.info('attention weight: {}'.format(alpha))
                    w_local_mdl = self._aggregate_func_attn(clnt_idx, nei_indexs, w_locals_record, alpha)
                    teacher_list = [w_locals_record[j] for j in nei_indexs]  # 作为 KD 教师
                    # 本地训练（你的 pDFL 版本）
                    client = self.client_list[clnt_idx]
                    w_local_mdl, p_test_before_metric = client.train(
                        copy.deepcopy(w_local_mdl),
                        w_locals[clnt_idx],
                        round_idx,
                        teachers=teacher_list,
                        attn=alpha
                    )
                elif self.args.mode == "pdfL_bi":
                    # 注意力：双向KL
                    alpha = self._compute_neighbor_attention(
                        clnt_idx, nei_indexs, w_locals_record,
                        tau=getattr(self.args, "attn_tau", 2.0),
                        metric="symkl"
                    )
                    self.logger.info('bi-attention weight: {}'.format(alpha))
                    w_local_mdl = self._aggregate_func_attn(clnt_idx, nei_indexs, w_locals_record, alpha)
                    teacher_list = [w_locals_record[j] for j in nei_indexs]
                    client = self.client_list[clnt_idx]
                    # 下面会在 trainer 里使用双向KD
                    w_local_mdl, p_test_before_metric = client.train(
                        copy.deepcopy(w_local_mdl),
                        w_locals[clnt_idx],
                        round_idx,
                        teachers=teacher_list,
                        attn=alpha,
                        # 让 trainer 知道用双向 KD
                        kd_mode="bi"
                    )
                elif self.args.mode == "kd_only":
                    # 2) 等权平均（原始聚合），仍传教师，attn=None
                    w_local_mdl = self._aggregate_func(
                        clnt_idx,
                        self.args.client_num_in_total,
                        self.args.client_num_per_round,
                        nei_indexs,
                        w_locals_record
                    )
                    teacher_list = [w_locals_record[j] for j in nei_indexs]
                    client = self.client_list[clnt_idx]
                    w_local_mdl, p_test_before_metric = client.train(
                        copy.deepcopy(w_local_mdl),
                        w_locals[clnt_idx],
                        round_idx,
                        teachers=teacher_list,
                        attn=None
                    )

                elif self.args.mode == "fedavg":
                    # 3) 纯 FedAvg：等权平均 + 整模型训练（无 KD/无冻结）
                    locals_for_agg = []
                    for clnt in range(self.args.client_num_in_total):
                        Ni = self.client_sample_nums[clnt]
                        Wi = copy.deepcopy(w_locals[clnt])  # 这里确保是本轮训练后的权重
                        locals_for_agg.append((Ni, Wi))

                    w_local_mdl = self._aggregate(locals_for_agg)  # 样本数加权
                    client = self.client_list[clnt_idx]
                    w_local_mdl, p_test_before_metric = client.train_fedavg(
                        copy.deepcopy(w_local_mdl),
                        w_locals[clnt_idx],
                        round_idx
                    )
                w_locals[clnt_idx] = copy.deepcopy(w_local_mdl)

                ## metrics for each client      
                p_test_metrics_before['num_samples'].append(copy.deepcopy(p_test_before_metric['test_total']))
                p_test_metrics_before['num_correct'].append(copy.deepcopy(p_test_before_metric['test_correct']))
                p_test_metrics_before['losses'].append(copy.deepcopy(p_test_before_metric['test_loss']))
                
                
            ## metrics in each round
            p_test_acc_before = sum(
            [np.array(p_test_metrics_before['num_correct'][i]) / np.array(p_test_metrics_before['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            p_test_loss = sum([np.array(p_test_metrics_before['losses'][i]) / np.array(p_test_metrics_before['num_samples'][i]) for i in
                            range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            stats = {'Local model person_test_acc_before': p_test_acc_before, 'person_test_loss': p_test_loss}
            self.stat_info["person_test_acc_before"].append(p_test_acc_before)
            self.logger.info(stats)
            
            if (round_idx+1) % 50 ==0:
                self.logger.info('person_test_acc_before50={}'.format(self.stat_info["person_test_acc_before"]))
            acc_locals.append(p_test_acc_before)
            loss_locals.append(p_test_loss)

            ## del  
            del w_local_mdl,p_test_metrics_before,w_locals_record
            gc.collect()

        np.savetxt(os.path.join(self.log_path, "acc_history.txt"), np.array(acc_locals, dtype=np.float32))
        np.savetxt(os.path.join(self.log_path, "loss_history.txt"), np.array(loss_locals, dtype=np.float32))

        self.logger.info('person_test_before_acc499={}'.format(self.stat_info["person_test_acc_before"]))  
 
        # show results
        test_max = max(self.stat_info["person_test_acc_before"])*100
        test_index = np.argmax(self.stat_info["person_test_acc_before"])
        stats = {'max person_test_acc_before': test_max, 'index': test_index}
        self.logger.info(stats)
        print("best person_test_acc_before %.3f" %(test_max))
        print("over")

    def _sample_probe_batch(self, clnt_idx, max_batches=2):
        """
        从当前客户端本地训练集取一个小批次用于注意力估计
        """
        dl = self.train_data_local_dict[clnt_idx]
        it = iter(dl)
        x_list, y_list = [], []
        with torch.no_grad():
            for _ in range(max_batches):
                try:
                    x, y = next(it)
                    x_list.append(x)
                    y_list.append(y)
                except StopIteration:
                    break
        if len(x_list) == 0:
            # 极端情形：数据为空则返回 None
            return None, None
        x = torch.cat(x_list, dim=0).to(self.device)
        y = torch.cat(y_list, dim=0).to(self.device)
        return x, y

    def _benefit_choose(self, round_idx, cur_clnt, client_num_in_total, client_num_per_round, cs = False):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
            return client_indexes
        if cs == "random": 
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx + cur_clnt)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        elif cs == "ring": 
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])
            
        elif cs == "grid":
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            top = (cur_clnt - 9 + client_num_in_total) % client_num_in_total
            down = (cur_clnt + 9 + client_num_in_total) % client_num_in_total
            client_indexes = np.asarray([left, right, top, down])

        elif cs =="exp": # (2^6<100<2^7)
            n1 = (cur_clnt + 1 + client_num_in_total) % client_num_in_total
            n2 = (cur_clnt + 2 + client_num_in_total) % client_num_in_total
            n3 = (cur_clnt + 4 + client_num_in_total) % client_num_in_total
            n4 = (cur_clnt + 8 + client_num_in_total) % client_num_in_total
            n5 = (cur_clnt + 16 + client_num_in_total) % client_num_in_total
            n6 = (cur_clnt + 32 + client_num_in_total) % client_num_in_total
            n7 = (cur_clnt + 64 + client_num_in_total) % client_num_in_total
            client_indexes = np.asarray([n1,n2,n3,n4,n5,n6,n7])

        elif cs == "full":
            client_indexes = np.arange(client_num_in_total)
            client_indexes = np.delete(client_indexes, cur_clnt)

        return client_indexes


    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        w_global ={}
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    w_global[k] = local_model_params[k] * w
                else:
                    w_global[k] += local_model_params[k] * w
        return w_global

    def _avg_aggregate(self, per_mdls):
        w_tmp = copy.deepcopy(per_mdls[0])
        w = 1 / len(per_mdls)
        for k in w_tmp.keys():
            w_tmp[k] = w_tmp[k] - w_tmp[k]
            for clnt in range(len(per_mdls)):
                w_tmp[k] += per_mdls[clnt][k] * w

        return w_tmp

    def _aggregate_func(self, cur_clnt, client_num_in_total, client_num_per_round, nei_indexs, w_locals_record):
        # self.logger.info('Doing local aggregation!')
        w_tmp = copy.deepcopy(w_locals_record[cur_clnt])
        w = 1 / len(nei_indexs)
        for k in w_tmp.keys():
            w_tmp[k] = w_tmp[k] - w_tmp[k]
            for clnt in nei_indexs:
                w_tmp[k] += w_locals_record[clnt][k] * w

        return w_tmp

    # === 修改：支持双向KL ===
    def _compute_neighbor_attention(self, clnt_idx, nei_indexs, w_locals_record, tau=2.0, metric="kl"):
        """
        单/双头注意力：
        - metric="kl"    : s_j = -KL(p_client || p_j)
        - metric="symkl" : s_j = -0.5*(KL(p||q)+KL(q||p))
        仅用于计算注意力，不更新参数
        """
        if len(nei_indexs) == 0:
            return None

        # 采样探针批次
        x, _ = self._sample_probe_batch(clnt_idx, max_batches=1)
        if x is None:
            return torch.full((len(nei_indexs),), 1.0 / len(nei_indexs), device=self.device)

        def _logits_from_state_dict(state_dict):
            model = copy.deepcopy(self.model_trainer.model).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            with torch.no_grad():
                logits = model(x)
            return logits

        # 学生
        student_logits = _logits_from_state_dict(w_locals_record[clnt_idx])
        log_p = F.log_softmax(student_logits / tau, dim=1)
        p = log_p.exp()

        scores = []
        for j in nei_indexs:
            teacher_logits = _logits_from_state_dict(w_locals_record[j])
            q = F.softmax(teacher_logits / tau, dim=1)
            if metric == "symkl":
                # 对称KL：0.5*(KL(p||q)+KL(q||p))
                kl_pq = F.kl_div(log_p, q, reduction="batchmean")  # KL(p||q)
                log_q = torch.log(q.clamp_min(1e-12))
                kl_qp = F.kl_div(log_q, p, reduction="batchmean")  # KL(q||p)
                s = -0.5 * (kl_pq + kl_qp)
            else:
                # 单向KL：KL(p||q)
                kl = F.kl_div(log_p, q, reduction="batchmean")
                s = -kl
            scores.append(s.item())
        scores = torch.tensor(scores, device=self.device)

        alpha = F.softmax(scores / tau, dim=0)
        return alpha.detach().float().cpu()

    def _aggregate_func_attn(self, cur_clnt, nei_indexs, w_locals_record, alpha):
        """
        对共享体（body）做注意力加权聚合；head（linear/output_layer）保持本地
        注意：这里的 state_dict 张量通常在 CPU，所以 alpha 必须是 CPU 标量/向量
        """
        w_out = copy.deepcopy(w_locals_record[cur_clnt])

        if alpha is None or len(nei_indexs) == 0:
            return self._aggregate_func(cur_clnt, self.args.client_num_in_total,
                                        self.args.client_num_per_round, nei_indexs, w_locals_record)

        # ✅ 确保 alpha 在 CPU，并转成 Python float 列表，避免设备/类型冲突
        if torch.is_tensor(alpha):
            alpha_list = alpha.detach().cpu().tolist()
        else:
            alpha_list = list(alpha)

        # 名称规则
        is_emnist = (self.args.dataset == "emnist")

        def is_head(name):
            return ('output_layer' in name) if is_emnist else ('linear' in name)

        # 保险：长度一致
        assert len(alpha_list) == len(nei_indexs), "alpha length != neighbor count"

        for k in w_out.keys():
            if is_head(k):
                # 本地头保持不变（CPU 上操作）
                w_out[k] = w_locals_record[cur_clnt][k]
            else:
                # 清零（在 CPU 上）
                w_out[k] = w_out[k] - w_out[k]
                # 用 Python float 做标量加权（避免把张量搬到 GPU）
                for a, j in zip(alpha_list, nei_indexs):
                    w_out[k] += w_locals_record[j][k] * float(a)

        return w_out

    def _test_on_all_clients(self, w_global, w_locals, round_idx):

        self.logger.info("################global_test_on_all_clients : {}".format(round_idx))

        p_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            # test 
            client = self.client_list[client_idx]
            p_test_local_metrics = client.local_test(w_locals[client_idx], True)
            p_test_metrics['num_samples'].append(copy.deepcopy(p_test_local_metrics['test_total']))
            p_test_metrics['num_correct'].append(copy.deepcopy(p_test_local_metrics['test_correct']))
            p_test_metrics['losses'].append(copy.deepcopy(p_test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
            
        p_test_acc = sum(
            [np.array(p_test_metrics['num_correct'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        p_test_loss = sum([np.array(p_test_metrics['losses'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
                           range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        stats = {'Local model person_test_acc': p_test_acc, 'person_test_loss': p_test_loss}
        self.stat_info["person_test_acc"].append(p_test_acc)
        self.logger.info(stats)


    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["person_test_acc_before"] = []

