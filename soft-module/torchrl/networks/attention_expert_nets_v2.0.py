import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl.networks.init as init
import json
import numpy as np
from torch.distributions import Distribution, Normal


# pf explore
class TanhNormal(Distribution):
    """
    Basically from RLKIT

    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + value) / (1 - value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
                self.normal_mean +
                self.normal_std *
                Normal(
                    torch.zeros(self.normal_mean.size()),
                    torch.ones(self.normal_std.size())
                ).sample().to(self.normal_mean.device)
        )

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def entropy(self):
        return self.normal.entropy()



# new 2 MLPBaseAE
class MLPBaseAE(nn.Module):
    def __init__(self, input_shape, hidden_shapes, expert_nums, expert_hidden_shapes, tower_hidden_shapes,
                 attention_shapes, activation_func=F.relu, init_func=init.basic_init, last_activation_func=None,
                 flag=None, expert_id=None, v_id=None, q_id=None, k_id=None, task_nums = None):
        super().__init__()
        self.flag = flag
        self.activation_func = activation_func
        self.fcs = []
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = activation_func

        if flag == 'baseline':
            input_shape = np.prod(input_shape)
            self.output_shape = input_shape
            for i, next_shape in enumerate(hidden_shapes):
                fc = nn.Linear(input_shape, next_shape)
                init_func(fc)
                self.fcs.append(fc)
                # set attr for pytorch to track parameters( device )
                self.__setattr__("baseline_fc{}".format(i), fc)

                input_shape = next_shape
                self.output_shape = next_shape

        elif flag == 'expert':
            input_shape = hidden_shapes[-1]
            self.output_shape = input_shape

            for i, next_shape in enumerate(expert_hidden_shapes):
                fc = nn.Linear(input_shape, next_shape)
                init_func(fc)
                self.fcs.append(fc)
                # set attr for pytorch to track parameters( device )
                self.__setattr__("expert{}_fc{}".format(expert_id, i), fc)

                input_shape = next_shape
                self.output_shape = next_shape

        elif flag == 'tower':
            input_shape = attention_shapes
            self.output_shape = input_shape

            for i, next_shape in enumerate(tower_hidden_shapes):
                fc = nn.Linear(input_shape, next_shape)
                init_func(fc)
                self.fcs.append(fc)
                # set attr for pytorch to track parameters( device )
                self.__setattr__("tower_fc{}".format(i), fc)

                input_shape = next_shape
                self.output_shape = next_shape

        elif flag == 'attention_v':
            input_shape = expert_hidden_shapes[-1]
            self.output_shape = input_shape

            fc = nn.Linear(input_shape, attention_shapes, bias=False)
            # init_func(fc)
            self.fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("v{}_fc".format(v_id), fc)

            self.output_shape = attention_shapes

        elif flag == 'attention_k':
            input_shape = expert_hidden_shapes[-1]
            self.output_shape = input_shape

            fc = nn.Linear(input_shape, attention_shapes, bias=False)
            # init_func(fc)
            self.fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("k{}_fc".format(k_id), fc)

            self.output_shape = attention_shapes

        elif flag == 'attention_q':
            input_shape = task_nums
            self.output_shape = input_shape

            fc = nn.Linear(input_shape, attention_shapes, bias=False)
            # init_func(fc)
            self.fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("q{}_fc".format(q_id), fc)

            self.output_shape = attention_shapes

    def forward(self, x):
        out = x
        for fc in self.fcs[:-1]:
            out = fc(out)
            out = self.activation_func(out)
        out = self.fcs[-1](out)
        if 'attention' not in self.flag:
            out = self.last_activation_func(out)
        return out

# 1 NetAE
class NetAE(nn.Module):
    def __init__(
            self, output_shape,
            base_type,
            expert_nums,
            task_nums,
            append_hidden_shapes=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            **kwargs):

        super().__init__()
        self.task_nums = task_nums
        self.n = expert_nums
        # 0 baseline network 2 layers mlp 300 300
        self.base = base_type(activation_func=activation_func, expert_nums=self.n, flag='baseline', **kwargs)
        self.activation_func = activation_func

        # 1 experts networks 3 * 2 layers mlp 400 400 output
        self.mlp_list = nn.ModuleList()

        for i in range(self.n):
            self.mlp_list.append(
                base_type(activation_func=activation_func, expert_nums=self.n, flag='expert', expert_id=i, **kwargs))

            # 2 attention module
        self.attention_v_list = nn.ModuleList()
        self.attention_q_list = nn.ModuleList()
        self.attention_k_list = nn.ModuleList()
        for i in range(self.n):
            self.attention_v_list.append(
                base_type(activation_func=activation_func, expert_nums=self.n, flag='attention_v', v_id=i, **kwargs))
            self.attention_k_list.append(
                base_type(activation_func=activation_func, expert_nums=self.n, flag='attention_k', k_id=i, **kwargs))
        for i in range(self.task_nums):
            self.attention_q_list.append(
                base_type(activation_func=activation_func, expert_nums=self.n, flag='attention_q', q_id=i, task_nums = self.task_nums,**kwargs))
            # 3 tower network 1 layers mlp 100
        self.tower = base_type(activation_func=activation_func, expert_nums=self.n, flag='tower', **kwargs)
        self.activation_func = activation_func

        #         append_input_shape = self.experts.output_shape

        #         self.append_fcs = []
        #         for i, next_shape in enumerate(append_hidden_shapes):
        #             fc = nn.Linear(append_input_shape, next_shape)
        #             append_hidden_init_func(fc)
        #             self.append_fcs.append(fc)
        #             # set attr for pytorch to track parameters( device )
        #             self.__setattr__("append_fc{}".format(i), fc)
        #             append_input_shape = next_shape

        # 4 last network
        self.last = nn.Linear(self.tower.output_shape, output_shape)
        net_last_init_func(self.last)

    def forward(self, x, task_id):
        task_idx = torch.zeros((self.task_nums,))
        task_idx[task_id.item()]=1

        # 0 baseline network
        out = self.base(x)

        expkqs = []
        vs = []

        for i in range(self.n):
            # 1 expert networks
            e = self.mlp_list[i](out)
            # 2 attention module
            v = torch.squeeze(self.attention_v_list[i](e))
            k = torch.squeeze(self.attention_k_list[i](e))
            q = self.attention_q_list[task_id.item()](task_idx)
            expkq = torch.dot(k, q)
            expkqs.append(expkq.item())
            vs.append(v)

        res = torch.zeros(vs[0].shape)
        alphas = [i / sum(expkqs) for i in expkqs]

        for i in range(self.n):
            res += vs[i] * expkqs[i]

        # 3 tower network
        out = self.tower(res)

        #         for append_fc in self.append_fcs:
        #             out = append_fc(out)
        #             out = self.activation_func(out)

        # 4 last network
        out = self.last(out)
        return out


# 0 GrassianContPolicyAE
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class GuassianContPolicyAE(NetAE):
    def forward(self, x, task_id):
        x = super().forward(x, task_id)

        mean, log_std = x.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        return mean, std, log_std

    def eval_act(self, x, task_id):
        with torch.no_grad():
            mean, _, _ = self.forward(x, task_id)
        return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()

    def explore(self, x, task_id, return_log_probs=False, return_pre_tanh=False):

        mean, std, log_std = self.forward(x, task_id)

        dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(-1, keepdim=True)

        dic = {
            "mean": mean,
            "log_std": log_std,
            "ent": ent
        }

        if return_log_probs:
            action, z = dis.rsample(return_pretanh_value=True)
            log_prob = dis.log_prob(
                action,
                pre_tanh_value=z
            )
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            dic["pre_tanh"] = z.squeeze(0)
            dic["log_prob"] = log_prob
        else:
            if return_pre_tanh:
                action, z = dis.rsample(return_pretanh_value=True)
                dic["pre_tanh"] = z.squeeze(0)
            action = dis.rsample(return_pretanh_value=False)

        dic["action"] = action.squeeze(0)
        return dic

    def update(self, obs, actions, task_id):
        mean, std, log_std = self.forward(obs, task_id)
        dis = TanhNormal(mean, std)

        log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
        ent = dis.entropy().sum(-1, keepdim=True)

        out = {
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "ent": ent
        }
        return out

# critics:
class FlattenNetAE(NetAE):
    def forward(self, input, task_id):
        out = torch.cat(input, dim = -1)
        return super().forward(out, task_id)