import torch
import torch.nn as nn
import math


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        # self.known_set = [i for i in (0, 9, 10, 11, 13, 16, 18, 20, 22, 24, 26, 27, 36, 43, 48, 50, 55, 60, 73, 75)]
        self.t1know_set = [42, 64, 34, 54, 55, 33, 24, 66, 47, 45, 43, 23, 3, 20, 74, 15, 41, 25, 16, 53]
        self.t2know_set =[42, 64, 34, 54, 55, 33, 24, 66, 47, 45, 43, 23, 3, 20, 74, 15, 41, 25, 16, 53, 8, 38, 75, 63, 49, 11, 30, 32, 39, 17, 36, 62, 78, 46, 12, 35, 26, 13, 37, 0]
        self.t3know_set = [42, 64, 34, 54, 55, 33, 24, 66, 47, 45, 43, 23, 3, 20, 74, 15, 41, 25, 16, 53, 8, 38, 75, 63, 49, 11, 30, 32, 39, 17, 36, 62, 78, 46, 12, 35, 26, 13, 37, 0,57,59,70,73,52,2,21,14,67,22,6,60,28,19,76,44,1,10,29,4]

        # self.mask = self.mask_correlated_samples(batch_size)
        # self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.criterion = nn.BCEWithLogitsLoss()
    def mask_correlated_samples(self, batch_size, labels):
        N = 2 * batch_size
        pos_mask = torch.zeros((N, N))
        for i in range(batch_size):
            pos_mask[i, batch_size + i] = 1
            pos_mask[batch_size + i, i] = 1
        known_index = []
        unknown_index = []
        for index, label in enumerate(labels):
            if label in self.t1know_set:
                known_index.append(index)
            else:
                unknown_index.append(index)
        for i in known_index:
            indices = [j for j in known_index[(i+1):] if labels[i] == labels[j]]
            for index in indices:
                pos_mask[i,index] = 1
                pos_mask[i, batch_size+i] = 1
                pos_mask[i ,batch_size+index ] = 1
                pos_mask[batch_size+i ,batch_size+index ] = 1
                pos_mask[batch_size+i,i] = 1
                pos_mask[batch_size+i,index] = 1

        # pos_mask = pos_mask.bool()
        mask = ~torch.eye(pos_mask.shape[0], dtype=bool)
        pos_mask = pos_mask[mask].view(pos_mask.shape[1], -1)
        return pos_mask

    def forward(self, z_i, z_j, labels):
        mask = self.mask_correlated_samples(self.batch_size,labels)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        mask =mask.to(sim.device)
        diaf_mask = ~torch.eye(sim.shape[0], dtype=bool)
        logits = sim[diaf_mask].view(sim.shape[1], -1)
        # loss = self.criterion(logits, mask)


        # fei
        exp_logits = torch.exp(logits)
        # print(exp_logits.shape)
        fenmu = torch.sum(exp_logits*(1-mask), dim=1, keepdim=True)
        # print(fenmu.shape)
        fenmu_a = fenmu.repeat(1, N-1)
        # print(fenmu_a.shape)
        fenmu_b = fenmu_a + exp_logits
        # print(fenmu_b.shape)
        loss = -torch.log(exp_logits / fenmu_b)
        # print(loss.shape)
        loss = torch.mean(loss[mask.type(torch.bool)])
        # print(loss)
        # fei
        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss
