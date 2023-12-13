import os
import numpy as np
import random
import torch
import math
import logging
from tqdm import tqdm


class DataLoader_():
    def __init__(self, dir, logger, max_arity=6):
        self.data = {}
        self.dir = dir
        self.max_arity = int(max_arity)
        self.logger = logger
        self.logger.info(f'Loading the dataset {os.path.basename(self.dir)} .....')
        self.ent2id = {"": 0}
        self.rel2id = {"": 0}
        self.data["train"] = self.read(os.path.join(self.dir, "train.txt"))
        # self.data["train_split"] = self.data_split()
        # np.random.shuffle(self.data['train'])
        self.data["valid"] = self.read(os.path.join(self.dir, "valid.txt"))
        # self.data["valid_split"] = self.data_split("valid")
        self.data["test"] = self.read_test(os.path.join(self.dir, "test.txt"))
        # self.data["test_split"] = self.data_split("test")

        # for i in range(2, self.max_arity + 1):  # 将二元关系和多元关系分类进行测试
        #     test_arity = "test_{}".format(i)
        #     file_path = os.path.join(self.dir, "test_{}.txt".format(i))
        #     self.data[test_arity] = self.read_test(file_path)
        self.batch_index = 0

    def read(self, file):
        self.logger.info(
            f'Starts Begins loading part {os.path.basename(file)} of data set {os.path.basename(self.dir)}.')
        if not os.path.exists(file):
            self.logger.error(f"*** {file} not found. Skipping. ***")
            return ()
        ary_2 = []
        ary_3 = []
        ary_4 = []
        ary_5 = []
        ary_6 = []
        with open(file, "r") as f:
            lines = f.readlines()
        for line in lines:
            if len(line.strip().split('\t'))==3:
                ary_2.append(line)
            elif len(line.strip().split('\t'))==4:
                ary_3.append(line)
            elif len(line.strip().split('\t'))==5:
                ary_4.append(line)
            elif len(line.strip().split('\t'))==6:
                ary_5.append(line)
            elif len(line.strip().split('\t'))==7:
                ary_6.append(line)
        tuples = []
        for i in range(5):
            tuples.append(np.zeros((len(eval("ary_{}".format(i+2))), self.max_arity + 3), dtype=int))
            # self.logger.debug(f'ary_{i+2}={lines[2]}.')
            for l in tqdm(range(len(eval("ary_{}".format(i+2))))):
                tuples[i][l] = self.tuple2ids(eval("ary_{}".format(i+2))[l].strip().split("\t"))
            # self.logger.debug(f'tuples[0]={tuples[0]}.')
            # self.logger.debug(f'tuples[1]={tuples[1]}.')
        for i in range(5):
            np.random.shuffle(tuples[i])
        return tuples

    def tuple2ids(self, tuple):
        output = np.zeros(self.max_arity + 3, dtype=int)
        for ind, t in enumerate(tuple):
            if ind == 0:
                output[ind] = self.get_rel_id(t)
            else:
                output[ind] = self.get_ent_id(t)
            output[-2] = ind
        return output

    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]

    def get_ent_id(self, ent):
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        return self.ent2id[ent]

    def read_test(self, file):
        self.logger.info(
            f'Starts Begins loading part {os.path.basename(file)} of data set {os.path.basename(self.dir)}.')
        if not os.path.exists(file):
            self.logger.error(f"*** {file} not found. Skipping. ***")
            return ()
        ary_2 = []
        ary_3 = []
        ary_4 = []
        ary_5 = []
        ary_6 = []
        with open(file, "r") as f:
            lines = f.readlines()
        for line in lines:
            if len(line.strip().split('\t'))==3:
                ary_2.append(line)
            elif len(line.strip().split('\t'))==4:
                ary_3.append(line)
            elif len(line.strip().split('\t'))==5:
                ary_4.append(line)
            elif len(line.strip().split('\t'))==6:
                ary_5.append(line)
            elif len(line.strip().split('\t'))==7:
                ary_6.append(line)
        tuples = []
        for i in range(5):
            tuples.append(np.zeros((len(eval("ary_{}".format(i+2))), self.max_arity + 3), dtype=int))
            # self.logger.debug(f'ary_{i+2}={lines[2]}.')
            for l in tqdm(range(len(eval("ary_{}".format(i+2))))):
                split = eval("ary_{}".format(i+2))[l].strip().split("\t")[1:]
                tuples[i][l] = self.tuple2ids(split)
            # self.logger.debug(f'tuples[0]={tuples[0]}.')
            # self.logger.debug(f'tuples[1]={tuples[1]}.')
            tuples[i][:, -1] = 1  # 标识全部为正样例
        return tuples



    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def rand_ent_except(self, ent):
        # id 0 is reserved for nothing. randint should return something between zero to len of entities
        rand = random.randint(1, self.num_ent() - 1)
        while ent == rand:
            rand = random.randint(1, self.num_ent() - 1)
        return rand

    def next_pos_batch(self, batch_size, arr, ll):
        if ll * batch_size + batch_size < len(self.data["train"][arr-2]):
            batch = self.data["train"][arr-2][ll * batch_size: ll * batch_size + batch_size]
        else:
            batch = self.data["train"][arr-2][ll * batch_size:]
            np.random.shuffle(self.data['train'][arr - 2])
        return batch

    def next_batch(self, batch_size, arr, ll, neg_ratio, device):
        pos_batch = self.next_pos_batch(batch_size, arr, ll)
        batch = self.generate_neg(pos_batch, neg_ratio)
        r = torch.tensor(batch[:, 0]).long().to(device)
        e1 = torch.tensor(batch[:, 1]).long().to(device)
        e2 = torch.tensor(batch[:, 2]).long().to(device)
        e3 = torch.tensor(batch[:, 3]).long().to(device)
        e4 = torch.tensor(batch[:, 4]).long().to(device)
        e5 = torch.tensor(batch[:, 5]).long().to(device)
        e6 = torch.tensor(batch[:, 6]).long().to(device)
        labels = batch[:, -1]
        return r, e1, e2, e3, e4, e5, e6, labels

    def generate_neg(self, pos_batch, neg_ratio):
        ar = pos_batch[:, -2]
        neg_batch = np.concatenate(
            [self.neg_each(np.repeat([c], neg_ratio * ar[i] + 1, axis=0), ar[i], neg_ratio) for i, c in
             enumerate(pos_batch)], axis=0)
        return neg_batch

    def neg_each(self, arr, arity, nr):
        arr[0, -1] = 1
        for a in range(arity):
            arr[a * nr + 1:(a + 1) * nr + 1, a + 1] = np.array([self.rand_ent_except(arr[0, a+1]) for i in range(nr)])
        return arr

    def was_last_batch(self):
        return self.batch_index == 0

    # def data_split(self, dataset = "train"):
    #     ary_2 = []
    #     ary_3 = []
    #     ary_4 = []
    #     ary_5 = []
    #     ary_6 = []
    #     ary_7 = []
    #     ary_8 = []
    #     ary_9 = []
    #
    #     data = self.data[dataset]
    #     for line in data:
    #         if len(line)==3:
    #             ary_2.append(line)
    #         elif len(line)==4:
    #             ary_3.append(line)
    #         elif len(line)==5:
    #             ary_4.append(line)
    #         elif len(line)==6:
    #             ary_5.append(line)
    #         elif len(line)==7:
    #             ary_6.append(line)
    #         elif len(line)==8:
    #             ary_7.append(line)
    #         elif len(line)==9:
    #             ary_8.append(line)
    #         elif len(line)==10:
    #             ary_9.append(line)
    #
    #     # self.read(os.path.join(self.dir, "train.txt"))
    #     for i in range(8):
    #         with open(os.path.join(self.dir, "{}_{}.txt".format(dataset, i+2)), "w") as f:
    #             for tuple_ in eval("ary_{}".format(i+2)):
    #                 for k in range(len(tuple_)):
    #                     if k != len(tuple_) - 1:
    #                         f.write("{}\t".format(tuple_[k]))
    #                     else:
    #                         f.write("{}".format(tuple_[k]))
    #                 f.write("\n")
    #             f.close()
    #
    #     da = []
    #     for i in range(8):
    #         da.append(eval("ary_{}".format(i+2)))
    #     return da