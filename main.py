import os
import glob
import re
import json

from torch import device
from torch.utils.data import DataLoader
from dataset import *
from models import *
from test import *
import numpy as np
import torch
from torch.nn import functional as F
import datetime
import argparse
import math
from tqdm import tqdm
import logging


class Main:
    def __init__(self, args, logger, device):
        self.logger = logger
        self.model_name = args.get('model')
        self.dataset_dir = args.get('dataset')
        self.max_arity = args.get('max_arity')
        self.dataset = DataLoader_(os.path.join("data", self.dataset_dir), self.logger, self.max_arity)
        self.device = device
        self.batch_size = args.get('batch_size')
        self.neg_ratio = args.get('neg_ratio')
        self.num_iterations = args.get('num_iterations')
        self.learning_rate = args.get('learning_rate')
        self.model_conf = args.get("model_hyper_params")
        # if self.model_name != 'ConvR':
        #     self.model_conf['chequer_perm'] = self.get_chequer_perm()
        self.emb_dim = self.model_conf["emb_dim"]
        self.model_conf['device'] = self.device

        self.restartable = args.get('restartable')
        self.pretrained = args.get('pretrained')
        self.best_model = args.get('best_model')

        self.load_model()

    def load_model(self):
        self.logger.info(f'Initializing the model ...')
        self.model = self.get_model_from_name()

        self.opt = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)

        # if self.pretrained is not None:
        #     print("Loading the pretrained model at {} for testing".format(self.pretrained))
        #     self.model.load_state_dict(torch.load(self.pretrained))
        #     # Construct the name of the optimizer file based on the pretrained model path
        #     opt_path = os.path.join(os.path.dirname(self.pretrained),
        #                             os.path.basename(self.pretrained).replace('model', 'opt'))
        #     # If an optimizer exists (needed for training, but not testing), then load it.
        #     if os.path.exists(opt_path):
        #         self.opt.load_state_dict(torch.load(opt_path))
        #     elif not self.test:
        #         # if no optimizer found and 'test` is not set, then raise an error
        #         # (we need a pretrained optimizer in order to continue training
        #         raise Exception("*** NO OPTIMIZER FOUND. SKIPPING. ****")
        # elif self.restartable and os.path.isdir(self.output_dir):
        #     # If the output_dir contains a model, then it will be loaded
        #     self.load_last_saved_model(self.output_dir)
        # else:
        #     # Initilize the model
        self.model.init()

    def get_model_from_name(self):
        if self.model_name == "GRACE":
            model = GRACE(self.logger, self.dataset, self.max_arity, **self.model_conf).to(self.device)
        elif self.model_name == "ConvR":
            model = ConvR(self.logger, self.dataset, self.max_arity, **self.model_conf).to(self.device)
        elif self.model_name == "InteractE":
            model = InteractE(self.logger, self.dataset, self.max_arity, **self.model_conf).to(self.device)
        elif self.model_name == "HyperConvD":
            model = HyperConvD(self.logger, self.dataset, self.max_arity, **self.model_conf).to(self.device)
        else:
            raise Exception("!!!! No mode called {} found !!!!".format(self.model_name))
        self.logger.info(f'Model {self.model_name} has been instantiated')
        return model

    def fit(self):
        if (self.model.cur_itr.data >= self.num_iterations):
            self.logger.info(f'Number of iterations is the same as that in the pretrained model.')
            self.logger.info(f'Nothing left to train. Exiting.')
            return

        self.logger.info(f'Training the {self.model_name} model...')
        self.logger.info(f'Number of training data points: {len(self.dataset.data["train"])}')

        loss_layer = torch.nn.CrossEntropyLoss()
        loss_layer.to(self.device)
        self.logger.info(f'The loss function is torch.nn.CrossEntropyLoss()')
        self.logger.info(f'Starting training at iteration ... {self.model.cur_itr.data}')
        for it in range(self.model.cur_itr.data, self.num_iterations + 1):
            self.model.train()
            self.model.cur_itr.data += 1
            losses = 0
            arr = 1
            for data in self.dataset.data['train']:
                arr = arr + 1
                self.logger.info(f'Starting training at array ... {arr}')
                if len(data) == 0:
                    continue
                for i in tqdm(range(int(len(data) / self.batch_size) + 1)):
                    r, e1, e2, e3, e4, e5, e6, targets = self.dataset.next_batch(self.batch_size, arr, i, neg_ratio=self.neg_ratio, device=self.device)
                    self.opt.zero_grad()
                    number_of_positive = len(np.where(targets > 0)[0])
                    predictions = self.model.forward(r, e1, e2, e3, e4, e5, e6)
                    predictions = self.padd_and_decompose(targets, predictions, self.neg_ratio * self.max_arity).view(
                    -1, self.max_arity * self.neg_ratio)
                    # self.logger.debug(f'predictions.size()={predictions.size()}')
                    targets = torch.zeros(number_of_positive).long().to(self.device)
                    # self.logger.debug(f'targets.size()={targets.size()}')
                    loss = loss_layer(predictions, targets)
                    loss.backward()
                    self.opt.step()
                    losses += loss.item()

            self.logger.info(f'Iteration#: {it}, loss: {losses}')

            if (it % 10 == 0) or (it == self.num_iterations):
                self.model.eval()
                with torch.no_grad():
                    self.logger.info(f'validation:')
                    print("validation:")
                    tester = Tester(self.dataset, self.model, "valid", self.model_name, self.device, self.logger)
                    measure_valid, _ = tester.test()
                    # mrr = measure_valid.mrr["fil"]
                    # This is the best model we have so far if
                    # no "best_model" exists yes, or if this MRR is better than what we had before
                    # is_best_model = (self.best_model is None) or (mrr > self.best_model.best_mrr)
                    # if is_best_model:
                    #     self.best_model = self.model
                    #     # Update the best_mrr value
                    #     self.best_model.best_mrr.data = torch.from_numpy(np.array([mrr]))
                    #     self.best_model.best_itr.data = torch.from_numpy(np.array([it]))
                    # Save the model at checkpoint
                    # self.save_model(it, "valid", is_best_model=is_best_model)

    def padd_and_decompose(self, targets, predictions, max_length):
        seq = self.decompose_predictions(targets, predictions, max_length)
        return torch.stack(seq)

    def decompose_predictions(self, targets, predictions, max_length):
        positive_indices = np.where(targets > 0)[0]
        seq = []
        for ind, val in enumerate(positive_indices):
            if ind == len(positive_indices) - 1:
                seq.append(self.padd(predictions[val:], max_length))
            else:
                seq.append(self.padd(predictions[val:positive_indices[ind + 1]], max_length))
        return seq

    def padd(self, a, max_length):
        a = a.view(1, -1)
        b = F.pad(a, (0, max_length - len(a[0])), 'constant', -math.inf)
        return b

    def get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self.model_conf['emb_dim']) for _ in range(self.model_conf['perm'])])
        rel_perm = np.int32([np.random.permutation(self.model_conf['emb_dim']) for _ in range(self.model_conf['perm'])])
        comb_idx = []
        for k in range(self.model_conf['perm']):  # 生成不同的排列数
            temp = []
            ent_idx, rel_idx = 0, 0
            for i in range(self.model_conf['k_h']):  # 这个k是什么的长度呢
                for j in range(self.model_conf['k_w']):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.model_conf['emb_dim'])
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.model_conf['emb_dim'])
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.model_conf['emb_dim'])
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.model_conf['emb_dim'])
                            rel_idx += 1
            comb_idx.append(temp)
        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm


def load_json_config(config_path, logger, device):
    # 将json文件导入
    if not os.path.exists(config_path):
        logger.error(f'File {config_path} does not exist, empty list is returned.')
        return
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['device'] = device
    return config


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    # 所有超参数通过json文件来传入
    # 所有超参数通过json文件来传入
    parser.add_argument('-c', '--config', action='store', default='config/HyperConvD.json', required=False)
    args = parser.parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    args = load_json_config(args.config, logger, device)
    logger.info(f'The hyperparameters are imported. The hyperparameters are {args}')

    model = Main(args, logger, device)
    model.fit()
