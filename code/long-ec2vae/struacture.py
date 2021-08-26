import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import model4 as models


class Musicmodel(nn.Module):
    def __init__(self,
                 roll_dims,
                 hidden_dims,
                 condition_dims,
                 z1_dims,
                 z2_dims,
                 n_step,
                 p_queuesize,
                 k=1000):
        super(Musicmodel, self).__init__()

        self.queue_size4 = p_queuesize
        # self.momentum = momentum
        # self.temperature = temperature

        # assert self.queue_size % args.batch_size == 0  # for simplicity
        self.int_model2 = models.VAEbar2(130, hidden_dims, 3, 12, 128, 128, 32)
        self.int_model4.eval()
        self.int_model4.requires_grad = False
        # Load model
        self.musicvae4 = models.VAEbar4(130, hidden_dims, 3, 12, z1_dims, z2_dims, 64)  # Query Encoder
        self.Wp14 = nn.Parameter(torch.randn(z1_dims, 128))
        self.Wp24 = nn.Parameter(torch.randn(z1_dims, 128))
        self.Wp34 = nn.Parameter(torch.randn(z1_dims, 128))
        self.Wr14 = nn.Parameter(torch.randn(z2_dims, 128))
        self.Wr24 = nn.Parameter(torch.randn(z2_dims, 128))
        self.Wr34 = nn.Parameter(torch.randn(z2_dims, 128))

        # self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        self.triplet_loss14 = torch.nn.CosineEmbeddingLoss(margin=0, size_average=None, reduce=None)
        self.triplet_loss24 = torch.nn.CosineEmbeddingLoss(margin=0, size_average=None, reduce=None)

        # Create the queue to store negative samples
        self.register_buffer("p1_queue4", torch.randn(self.queue_size4, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("p1_queue_ptr4", torch.zeros(1, dtype=torch.long))

        self.register_buffer("r1_queue4", torch.randn(self.queue_size4, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("r1_queue_ptr4", torch.zeros(1, dtype=torch.long))

        self.register_buffer("p2_queue4", torch.randn(self.queue_size4, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("p2_queue_ptr4", torch.zeros(1, dtype=torch.long))

        self.register_buffer("r2_queue4", torch.randn(self.queue_size4, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("r2_queue_ptr4", torch.zeros(1, dtype=torch.long))
        self.register_buffer("p3_queue4", torch.randn(self.queue_size4, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("p3_queue_ptr4", torch.zeros(1, dtype=torch.long))

        self.register_buffer("r3_queue4", torch.randn(self.queue_size4, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("r3_queue_ptr4", torch.zeros(1, dtype=torch.long))

        # @torch.no_grad()

    # def momentum_update(self):
    # '''
    # Update the key_encoder parameters through the momentum update:
    # key_params = momentum * key_params + (1 - momentum) * query_params
    # '''

    # # For each of the parameters in each encoder
    # for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
    # p_k.data = p_k.data * self.momentum + p_q.detach().data * (1. - self.momentum)

    @torch.no_grad()
    def shuffled_idx(self, batch_size):
        '''
        Generation of the shuffled indexes for the implementation of ShuffleBN.
        https://github.com/HobbitLong/CMC.
        args:
            batch_size (Tensor.int()):  Number of samples in a batch
        returns:
            shuffled_idxs (Tensor.long()): A random permutation index order for the shuffling of the current minibatch
            reverse_idxs (Tensor.long()): A reverse of the random permutation index order for the shuffling of the
                                            current minibatch to get back original sample order
        '''

        # Generate shuffled indexes
        shuffled_idxs = torch.randperm(batch_size).long().cuda()

        reverse_idxs = torch.zeros(batch_size).long().cuda()

        value = torch.arange(batch_size).long().cuda()

        reverse_idxs.index_copy_(0, shuffled_idxs, value)

        return shuffled_idxs, reverse_idxs

    @torch.no_grad()
    def update_p1queue4(self, feat_p):
        '''
        Update the memory / queue.
        Add batch to end of most recent sample index and remove the oldest samples in the queue.
        Store location of most recent sample index (ptr).
        Taken from: https://github.com/facebookresearch/moco
        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_p.size(0)

        ptr = int(self.p1_queue_ptr4)

        # replace the keys at ptr (dequeue and enqueue)
        self.p1_queue4[ptr:ptr + batch_size, :] = feat_p

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size4

        # Store queue pointer as register_buffer
        self.p1_queue_ptr4[0] = ptr

    @torch.no_grad()
    def update_r1queue4(self, feat_p):
        '''
        Update the memory / queue.
        Add batch to end of most recent sample index and remove the oldest samples in the queue.
        Store location of most recent sample index (ptr).
        Taken from: https://github.com/facebookresearch/moco
        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_p.size(0)

        ptr = int(self.r1_queue_ptr4)

        # replace the keys at ptr (dequeue and enqueue)
        self.r1_queue4[ptr:ptr + batch_size, :] = feat_p

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size4

        # Store queue pointer as register_buffer
        self.r1_queue_ptr4[0] = ptr

    @torch.no_grad()
    def update_p2queue4(self, feat_p):
        '''
        Update the memory / queue.
        Add batch to end of most recent sample index and remove the oldest samples in the queue.
        Store location of most recent sample index (ptr).
        Taken from: https://github.com/facebookresearch/moco
        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_p.size(0)

        ptr = int(self.p2_queue_ptr4)

        # replace the keys at ptr (dequeue and enqueue)
        self.p2_queue4[ptr:ptr + batch_size, :] = feat_p

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size4

        # Store queue pointer as register_buffer
        self.p2_queue_ptr4[0] = ptr

    @torch.no_grad()
    def update_r2queue4(self, feat_p):
        '''
        Update the memory / queue.
        Add batch to end of most recent sample index and remove the oldest samples in the queue.
        Store location of most recent sample index (ptr).
        Taken from: https://github.com/facebookresearch/moco
        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_p.size(0)

        ptr = int(self.r2_queue_ptr4)

        # replace the keys at ptr (dequeue and enqueue)
        self.r2_queue4[ptr:ptr + batch_size, :] = feat_p

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size4

        # Store queue pointer as register_buffer
        self.r2_queue_ptr4[0] = ptr

    @torch.no_grad()
    def update_p3queue4(self, feat_p):
        '''
        Update the memory / queue.
        Add batch to end of most recent sample index and remove the oldest samples in the queue.
        Store location of most recent sample index (ptr).
        Taken from: https://github.com/facebookresearch/moco
        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_p.size(0)

        ptr = int(self.p3_queue_ptr4)

        # replace the keys at ptr (dequeue and enqueue)
        self.p3_queue4[ptr:ptr + batch_size, :] = feat_p

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size4

        # Store queue pointer as register_buffer
        self.p3_queue_ptr4[0] = ptr

    @torch.no_grad()
    def update_r3queue4(self, feat_p):
        '''
        Update the memory / queue.
        Add batch to end of most recent sample index and remove the oldest samples in the queue.
        Store location of most recent sample index (ptr).
        Taken from: https://github.com/facebookresearch/moco
        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_p.size(0)

        ptr = int(self.r3_queue_ptr4)

        # replace the keys at ptr (dequeue and enqueue)
        self.r3_queue4[ptr:ptr + batch_size, :] = feat_p

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size4

        # Store queue pointer as register_buffer
        self.r3_queue_ptr4[0] = ptr

    def InfoNCE_logitsp14(self, f_a, f_1, f_2, f_3, f_4, f_5, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)
        f_5 = F.normalize(f_5, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wp14)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wp14)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wp14)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wp14)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau
        neg4 = torch.mm(f_a, self.Wp14)
        neg4 = torch.mm(neg4, f_5.transpose(1, 0)) / tau
        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)
        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)
        neg4 = torch.sum(torch.exp(neg4), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3 + neg4)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsp24(self, f_a, f_1, f_2, f_3, f_4, f_5, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)
        f_5 = F.normalize(f_5, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wp24)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wp24)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wp24)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wp24)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau
        neg4 = torch.mm(f_a, self.Wp24)
        neg4 = torch.mm(neg4, f_5.transpose(1, 0)) / tau
        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)
        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)
        neg4 = torch.sum(torch.exp(neg4), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3 + neg4)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsp34(self, f_a, f_1, f_2, f_3, f_4, f_5, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)
        f_5 = F.normalize(f_5, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wp34)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wp34)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wp34)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wp34)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau
        neg4 = torch.mm(f_a, self.Wp34)
        neg4 = torch.mm(neg4, f_5.transpose(1, 0)) / tau
        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)
        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)
        neg4 = torch.sum(torch.exp(neg4), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3 + neg4)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsr14(self, f_a, f_1, f_2, f_3, f_4, f_5, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)
        f_5 = F.normalize(f_5, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wr14)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wr14)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wr14)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wr14)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau
        neg4 = torch.mm(f_a, self.Wr14)
        neg4 = torch.mm(neg4, f_5.transpose(1, 0)) / tau
        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)
        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)
        neg4 = torch.sum(torch.exp(neg4), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3 + neg4)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsr24(self, f_a, f_1, f_2, f_3, f_4, f_5, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)
        f_5 = F.normalize(f_5, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wr24)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wr24)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wr24)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wr24)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau
        neg4 = torch.mm(f_a, self.Wr24)
        neg4 = torch.mm(neg4, f_5.transpose(1, 0)) / tau
        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)
        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)
        neg4 = torch.sum(torch.exp(neg4), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3 + neg4)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsr34(self, f_a, f_1, f_2, f_3, f_4, f_5, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)
        f_5 = F.normalize(f_5, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wr34)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wr34)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wr34)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wr34)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau
        neg4 = torch.mm(f_a, self.Wr34)
        neg4 = torch.mm(neg4, f_5.transpose(1, 0)) / tau
        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)
        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)
        neg4 = torch.sum(torch.exp(neg4), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3 + neg4)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def forward(self, x_a, c):
        batch_size = x_a.size(0)
        x_1 = x_a[:, 0:32, :]  ##batch seq_len dim
        x_2 = x_a[:, 32:64, :]
        c_1 = c[:, 0:32, :]
        c_2 = c[:, 32:64, :]
        x_3 = x_a[:, 16:48, :]
        c_3 = c[:, 16:48, :]

        # shuffled_idxs, reverse_idxs = self.shuffled_idx(batch_size)
        with torch.no_grad():
            dis1, dis2 = self.int_model2.encoder(x_1, c_1)
            z11 = dis1.mean
            z21 = dis2.mean
            dis1, dis2 = self.int_model2.encoder(x_2, c_2)
            z12 = dis1.mean
            z22 = dis2.mean
            dis1, dis2 = self.int_model2.encoder(x_3, c_3)
            z13 = dis1.mean
            z23 = dis2.mean


            # Shuffle minibatch 提取前两小节的负例
            z_11 = self.r1_queue4.clone().detach()
            z_21 = self.p1_queue4.clone().detach()
            # cc = self.c_queue.clone().detach()
            zr1 = torch.zeros(z12.size(0), 128)
            zr1[:] = z_11[0:z12.size(0)]
            zp1 = torch.zeros(z12.size(0), 128)
            zp1[:] = z_21[0:z12.size(0)]

            ####提取后两小节的负例
            z_12 = self.r2_queue4.clone().detach()
            z_22 = self.p2_queue4.clone().detach()
            # cc = self.c_queue.clone().detach()
            zr2 = torch.zeros(z12.size(0), 128)
            zr2[:] = z_12[0:z12.size(0)]
            zp2 = torch.zeros(z12.size(0), 128)
            zp2[:] = z_22[0:z12.size(0)]

            z_13 = self.r3_queue4.clone().detach()
            z_23 = self.p3_queue4.clone().detach()
            # cc = self.c_queue.clone().detach()
            zr3 = torch.zeros(z13.size(0), 128)
            zr3[:] = z_13[0:z13.size(0)]
            zp3 = torch.zeros(z13.size(0), 128)
            zp3[:] = z_23[0:z13.size(0)]

            zr1 = zr1.cuda()
            zp1 = zp1.cuda()
            zp2 = zp2.cuda()
            zr2 = zr2.cuda()
            zp3 = zp3.cuda()
            zr3 = zr3.cuda()

            # Feature representations of the shuffled key view from the key encoder
            # z2 = z2[shuffled_idxs]
            # reverse the shuffled samples to original position
            # z2 = z2[reverse_idxs]
        if (z11.size(0)) % 128 == 0:
            self.update_p2queue4(z12)
            self.update_p1queue4(z11)
            self.update_r2queue4(z22)
            self.update_r1queue4(z21)
            self.update_r3queue4(z23)
            self.update_p3queue4(z13)
            # Feature representations of the query view from the query encoder
        recon0, recon0_r, dis1ma, dis1sa, dis2ma, dis2sa, z1a, z2a = self.musicvae4(x_a, c)

        # Compute the logits for the InfoNCE contrastive loss.
        pNCEloss1 = self.InfoNCE_logitsp14(z1a, z11, zp1, z21, z22, z23, 1)
        pNCEloss2 = self.InfoNCE_logitsp24(z1a, z12, zp2, z22, z21, z23, 1)
        pNCEloss3 = self.InfoNCE_logitsp34(z1a, z13, zp3, z23, z22, z21, 1)
        rNCEloss1 = self.InfoNCE_logitsr14(z2a, z21, zr1, z11, z13, z12, 1)
        rNCEloss2 = self.InfoNCE_logitsr24(z2a, z22, zr2, z12, z11, z13, 1)
        rNCEloss3 = self.InfoNCE_logitsr34(z2a, z23, zr3, z13, z12, z11, 1)
        # Update the queue/memory with the current key_encoder minibatch.
        out = (recon0, recon0_r, dis1ma, dis1sa, dis2ma, dis2sa, pNCEloss1, pNCEloss2, pNCEloss3, rNCEloss1, rNCEloss2,
               rNCEloss3)

        return out
