import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import model as models
#import model_hc2 as models_h
import model_h as models_h


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

        self.queue_size_total = p_queuesize
        # self.momentum = momentum
        # self.temperature = temperature

        # assert self.queue_size % args.batch_size == 0  # for simplicity
        self.int_model_8 = models.VAEbar8(130, hidden_dims, 3, 12, 128, 128, 128)
        self.int_model_8.eval()
        self.int_model_8.requires_grad = False
        self.int_model_4 = models.VAEbar4(130, hidden_dims, 3, 12, 128, 128, 64)
        self.int_model_4.train()
        #self.int_model_4.encoder.parameters().requires_grad = False
        self.int_model_2 = models.VAEbar2(130, hidden_dims, 3, 12, 128, 128, 32)
        self.int_model_2.train()
        #self.int_model_2.encoder.parameters().requires_grad = False

        # Load model
        self.decoder84 = models_h.Decoder8to4(1024,128,128,128,128,2)  # Query Encoder
        self.decoder42 = models_h.Decoder4to2(1024, 128, 128, 128,128,2)
        
        self.Wp84 = nn.Parameter(torch.randn(z1_dims, z1_dims))
        self.Wp84_ = nn.Parameter(torch.randn(z1_dims, z1_dims))
        self.Wr84 = nn.Parameter(torch.randn(z1_dims, z1_dims))
        self.Wr84_ = nn.Parameter(torch.randn(z1_dims, z1_dims))

        self.Wp42_1 = nn.Parameter(torch.randn(z2_dims, z2_dims))
        self.Wp42_2 = nn.Parameter(torch.randn(z2_dims, z2_dims))
        self.Wp42_3 = nn.Parameter(torch.randn(z2_dims, z2_dims))
        self.Wp42_4 = nn.Parameter(torch.randn(z2_dims, z2_dims))

        self.Wr42_1 = nn.Parameter(torch.randn(z2_dims, z2_dims))
        self.Wr42_2 = nn.Parameter(torch.randn(z2_dims, z2_dims))
        self.Wr42_3 = nn.Parameter(torch.randn(z2_dims, z2_dims))
        self.Wr42_4 = nn.Parameter(torch.randn(z2_dims, z2_dims))

        # Create the queue to store negative samples
        self.register_buffer("p_queue84", torch.randn(self.queue_size_total, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("p_queue_ptr84", torch.zeros(1, dtype=torch.long))

        self.register_buffer("r_queue84", torch.randn(self.queue_size_total, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("r_queue_ptr84", torch.zeros(1, dtype=torch.long))

        self.register_buffer("p_queue42", torch.randn(self.queue_size_total, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("p_queue_ptr42", torch.zeros(1, dtype=torch.long))

        self.register_buffer("r_queue42", torch.randn(self.queue_size_total, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("r_queue_ptr42", torch.zeros(1, dtype=torch.long))




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
    def update_pqueue84(self, feat_p):
        '''
        Update the memory / queue.
        Add batch to end of most recent sample index and remove the oldest samples in the queue.
        Store location of most recent sample index (ptr).
        Taken from: https://github.com/facebookresearch/moco
        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_p.size(0)

        ptr = int(self.p_queue_ptr84)

        # replace the keys at ptr (dequeue and enqueue)
        self.p_queue84[ptr:ptr + batch_size, :] = feat_p

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size_total

        # Store queue pointer as register_buffer
        self.p_queue_ptr84[0] = ptr

    @torch.no_grad()
    def update_rqueue84(self, feat_p):
        '''
        Update the memory / queue.
        Add batch to end of most recent sample index and remove the oldest samples in the queue.
        Store location of most recent sample index (ptr).
        Taken from: https://github.com/facebookresearch/moco
        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_p.size(0)

        ptr = int(self.r_queue_ptr84)

        # replace the keys at ptr (dequeue and enqueue)
        self.r_queue84[ptr:ptr + batch_size, :] = feat_p

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size_total

        # Store queue pointer as register_buffer
        self.r_queue_ptr84[0] = ptr

    @torch.no_grad()
    def update_pqueue42(self, feat_p):
        '''
        Update the memory / queue.
        Add batch to end of most recent sample index and remove the oldest samples in the queue.
        Store location of most recent sample index (ptr).
        Taken from: https://github.com/facebookresearch/moco
        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_p.size(0)

        ptr = int(self.p_queue_ptr42)

        # replace the keys at ptr (dequeue and enqueue)
        self.p_queue42[ptr:ptr + batch_size, :] = feat_p

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size_total

        # Store queue pointer as register_buffer
        self.p_queue_ptr42[0] = ptr

    @torch.no_grad()
    def update_rqueue42(self, feat_p):
        '''
        Update the memory / queue.
        Add batch to end of most recent sample index and remove the oldest samples in the queue.
        Store location of most recent sample index (ptr).
        Taken from: https://github.com/facebookresearch/moco
        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_p.size(0)

        ptr = int(self.r_queue_ptr42)

        # replace the keys at ptr (dequeue and enqueue)
        self.r_queue42[ptr:ptr + batch_size, :] = feat_p

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size_total

        # Store queue pointer as register_buffer
        self.r_queue_ptr42[0] = ptr


    def InfoNCE_logitsp84_1(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wp84)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wp84)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wp84)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wp84)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsp84_2(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wp84_)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wp84_)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wp84_)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wp84_)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch




    def InfoNCE_logitsr84_1(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wr84)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wr84)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wr84)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wr84)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau


        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3 )
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch


    def InfoNCE_logitsr42_2(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wr42_2)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wr42_2)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wr42_2)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wr42_2)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch


    def InfoNCE_logitsr42_1(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wr42_1)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wr42_1)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wr42_1)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wr42_1)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch




    def InfoNCE_logitsr84_2(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wr84_)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wr84_)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wr84_)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wr84_)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau


        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3 )
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch


    def InfoNCE_logitsp42_1(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wp42_1)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wp42_1)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wp42_1)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wp42_1)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsp42_2(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wp42_2)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wp42_2)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wp42_2)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wp42_2)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsp42_3(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wp42_3)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wp42_3)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wp42_3)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wp42_3)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsp42_4(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wp42_4)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wp42_4)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wp42_4)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wp42_4)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsr42_1(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wr42_1)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wr42_1)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wr42_1)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wr42_1)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsr42_2(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wr42_2)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wr42_2)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wr42_2)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wr42_2)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsr42_3(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wr42_3)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wr42_3)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wr42_3)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wr42_3)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsr42_4(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        pos1 = torch.mm(f_a, self.Wr42_4)
        pos1 = torch.mm(pos1, f_1.transpose(1, 0)) / tau
        neg1 = torch.mm(f_a, self.Wr42_4)
        neg1 = torch.mm(neg1, f_2.transpose(1, 0)) / tau
        neg2 = torch.mm(f_a, self.Wr42_4)
        neg2 = torch.mm(neg2, f_3.transpose(1, 0)) / tau
        neg3 = torch.mm(f_a, self.Wr42_4)
        neg3 = torch.mm(neg3, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)


        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch






    def forward(self, x_a, c,epoch):
        batch_size = x_a.size(0)
        x_1 = x_a[:, 0:64, :]  ##batch seq_len dim
        x_2 = x_a[:, 64:128, :]
        c_1 = c[:, 0:64, :]
        c_2 = c[:, 64:128, :]

        x_3 = x_a[:, 0:32, :]
        c_3 = c[:, 0:32, :]
        x_4 = x_a[:, 32:64, :]
        c_4 = c[:, 32:64, :]
        x_5 = x_a[:, 64:96, :]
        c_5 = c[:, 64:96, :]
        x_6 = x_a[:, 96:128, :]
        c_6 = c[:, 96:128, :]


        # shuffled_idxs, reverse_idxs = self.shuffled_idx(batch_size)
        with torch.no_grad():
            dis_81, dis_82 = self.int_model_8.encoder(x_a, c)
            z8_1 = dis_81.rsample()
            z8_2 = dis_82.rsample()


            dis4_11, dis4_12 = self.int_model_4.encoder(x_1, c_1)
            z4_11 = dis4_11.rsample()
            z4_12 = dis4_12.rsample()

            dis4_21, dis4_22 = self.int_model_4.encoder(x_2, c_2)
            z4_21 = dis4_21.rsample()
            z4_22 = dis4_22.rsample()

            dis2_11, dis2_12 = self.int_model_2.encoder(x_3, c_3)
            z2_11 = dis2_11.rsample()
            z2_12 = dis2_12.rsample()

            dis2_21, dis2_22 = self.int_model_2.encoder(x_4, c_4)
            z2_21 = dis2_21.rsample()
            z2_22 = dis2_22.rsample()

            dis2_31, dis2_32 = self.int_model_2.encoder(x_5, c_5)
            z2_31 = dis2_31.rsample()
            z2_32 = dis2_32.rsample()

            dis2_41, dis2_42 = self.int_model_2.encoder(x_6, c_6)
            z2_41 = dis2_41.rsample()
            z2_42 = dis2_42.rsample()

            z4_11_ = torch.unsqueeze(z4_11,1)
            z4_21_ = torch.unsqueeze(z4_21,1)
            z4_12_ = torch.unsqueeze(z4_12,1)
            z4_22_ = torch.unsqueeze(z4_22,1)
            z4p = torch.cat((z4_11_,z4_21_),1)
            z4r = torch.cat((z4_12_, z4_22_), 1)

            z2_11_ = torch.unsqueeze(z2_11,1)
            z2_21_ = torch.unsqueeze(z2_21,1)
            z2_12_ = torch.unsqueeze(z2_12, 1)
            z2_22_ = torch.unsqueeze(z2_22,1)
            z2p1 = torch.cat((z2_11_,z2_21_),1)
            z2r1 = torch.cat((z2_12_,z2_22_),1)

            z2_31_ = torch.unsqueeze(z2_31,1)
            z2_41_ = torch.unsqueeze(z2_41,1)
            z2_32_ = torch.unsqueeze(z2_32,1)
            z2_42_ = torch.unsqueeze(z2_42,1)

            z2p2 = torch.cat((z2_31_,z2_41_),1)
            z2r2 = torch.cat((z2_32_,z2_42_),1)


            z_4psa = self.p_queue84.clone().detach()
            z_4rsa = self.r_queue84.clone().detach()

            z_2psa = self.p_queue42.clone().detach()
            z_2rsa = self.r_queue42.clone().detach()

            z_4p = torch.zeros(z4p.size(0), 64)
            z_4p = z_4psa[0:z4p.size(0)]

            z_4r = torch.zeros(z4p.size(0), 64)
            z_4r = z_4rsa[0:z4p.size(0)]

            z_2p = torch.zeros(z4p.size(0), 128)
            z_2p = z_2psa[0:z4p.size(0)]
            z_2r = torch.zeros(z4p.size(0), 128)
            z_2r = z_2rsa[0:z4p.size(0)]

            z_4p = z_4p.cuda()
            z_4r = z_4r.cuda()
            z_2p = z_2p.cuda()
            z_2r = z_2r.cuda()


        z4p_re1,z4r_re1 = self.decoder84(z8_1,z8_2,z4p,z4r)
        outp, outr, hxp, hxr = [None],[None],[None],[None]
        z2p_re1, z2r_re1  = self.decoder42(z4p_re1[:,0],z4r_re1[:,0],z2p1,z2r1)
        z2p_re2, z2r_re2 = self.decoder42(z4p_re1[:,1], z4r_re1[:,1], z2p2, z2r2)

        recon_rhythm_41 = self.int_model_4.rhythm_decoder(z4r_re1[:,0],x_1)
        recon_41 = self.int_model_4.final_decoder(z4p_re1[:,0], recon_rhythm_41, c_1,x_1)
        recon_rhythm_42 = self.int_model_4.rhythm_decoder(z4r_re1[:,1],x_2)
        recon_42 = self.int_model_4.final_decoder(z4p_re1[:,1], recon_rhythm_42, c_2,x_2)


        recon_rhythm_1 = self.int_model_2.rhythm_decoder(z2r_re1[:,0],x_3)
        recon_1 = self.int_model_2.final_decoder(z2p_re1[:,0], recon_rhythm_1, c_3,x_3)
        recon_rhythm_2 = self.int_model_2.rhythm_decoder(z2r_re1[:,1],x_4)
        recon_2 = self.int_model_2.final_decoder(z2p_re1[:,1], recon_rhythm_2, c_4,x_4)
        recon_rhythm_3 = self.int_model_2.rhythm_decoder(z2r_re2[:,0],x_5)
        recon_3 = self.int_model_2.final_decoder(z2p_re2[:,0], recon_rhythm_3, c_5,x_5)
        recon_rhythm_4 = self.int_model_2.rhythm_decoder(z2r_re2[:,1],x_6)
        recon_4 = self.int_model_2.final_decoder(z2p_re2[:,1], recon_rhythm_4, c_6,x_6)




        if (x_a.size(0)) % 32 == 0:
            self.update_pqueue84(z4_11)
            self.update_rqueue84(z4_12)
            self.update_pqueue42(z2_11)
            self.update_rqueue42(z2_12)

        # Compute the logits for the InfoNCE contrastive loss.
        print(z4p_re1[:,0].shape)
        pNCEloss41 = self.InfoNCE_logitsp84_1(z4p_re1[:,0], z4_11, z4r_re1[:,0], z4_12, z_4p , 1)
        pNCEloss42 = self.InfoNCE_logitsp84_2(z4p_re1[:,1], z4_21, z4r_re1[:,1], z4_22, z_4p, 1)
        rNCEloss41 = self.InfoNCE_logitsr84_1(z4r_re1[:,0], z4_12, z4p_re1[:,0], z4_11, z_4r , 1)
        rNCEloss42 = self.InfoNCE_logitsr84_2(z4r_re1[:,1], z4_22, z4p_re1[:,1], z4_21, z_4r, 1)

        pNCEloss21 = self.InfoNCE_logitsp42_1(z2p_re1[:,0], z2_11, z2r_re1[:,0], z2_12, z_2p , 1)
        pNCEloss22 = self.InfoNCE_logitsp42_2(z2p_re1[:,1], z2_21, z2r_re1[:,1], z2_22, z_2p, 1)
        pNCEloss23 = self.InfoNCE_logitsp42_3(z2p_re2[:,0], z2_31, z2r_re2[:,0], z2_32, z_2p , 1)
        pNCEloss24 = self.InfoNCE_logitsp42_4(z2p_re2[:,1], z2_41, z2r_re2[:,1], z2_42, z_2p, 1)

        rNCEloss21 = self.InfoNCE_logitsr42_1(z2r_re1[:,0], z2_12, z2p_re1[:,0], z2_11, z_2r , 1)
        rNCEloss22 = self.InfoNCE_logitsr42_2(z2r_re1[:,1], z2_22, z2p_re1[:,1], z2_21, z_2r, 1)
        rNCEloss23 = self.InfoNCE_logitsr42_3(z2r_re2[:,0], z2_32, z2p_re2[:,0], z2_31, z_2r , 1)
        rNCEloss24 = self.InfoNCE_logitsr42_4(z2r_re2[:,1], z2_42, z2p_re2[:,1], z2_41, z_2r, 1)
                
        out = (recon_1, recon_rhythm_1,recon_2, recon_rhythm_2, recon_3, recon_rhythm_3,recon_4, recon_rhythm_4,z4p_re1,z4r_re1,z2p_re1, z2r_re1,z2p_re2, z2r_re2,z4p,z4r,z2p1,z2r1,z2p2,z2r2,pNCEloss41,pNCEloss42,rNCEloss41,rNCEloss42,pNCEloss21,pNCEloss22,pNCEloss23,pNCEloss24,rNCEloss21,rNCEloss22,rNCEloss23,rNCEloss24,recon_41, recon_rhythm_41,recon_42, recon_rhythm_42)
        return out
