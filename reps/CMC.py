import torch
import torch.nn as nn

from .submodels import DetEncoder
# from .utils import prior_expert, reparameterize, l2_loss, kl_loss, bce_loss

class CMC(nn.Module):
    def __init__(self, rec_coeff, activation, n_latents, view_sizes, hidden_size, view_binaries, use_prior_expert=False):
        super(CMC, self).__init__()
        self.encoders = nn.ModuleList()
        self.linears = nn.ModuleList()

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        for i in range(len(view_sizes)):
            self.encoders.append(DetEncoder(activation, n_latents, view_sizes[i], hidden_size))
            self.linears.append(nn.Linear(n_latents, hidden_size))

        self.use_prior_expert = use_prior_expert
        self.n_latents = n_latents
        self.view_sizes = view_sizes
        self.hidden_size = hidden_size
        self.view_binaries = view_binaries
        self.method = 'CMC'
        self.rec_coeff = rec_coeff


    def get_method(self):
        return self.method

    def forward(self, views):
        mu_J, mu_perview_list = self.infer(views)

        return mu_J, mu_perview_list

    def infer(self, views, use_subset=None):
        batch_size = views[0].size(0)
        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        mu_perview_list = []

        if use_subset is None:
            use_subset = [True]*len(views)

        
        mus = None
        view_count = 0.0
        for i in range(len(views)):
            if use_subset[i] is False:
                continue
            view_count += 1.0
            mu_perview = self.encoders[i](views[i])
            mu_perview_list.append(mu_perview)

            if mus is not None:
                mus += mu_perview

            else:
                mus = mu_perview

        # Average pooling
        mu_J = (1.0/view_count) * mus

        return mu_J, mu_perview_list

    def compute_loss(self, mu_J,  mu_perview_list, lambda_views, anneal_factor=1):
        n_views = len(self.view_sizes)
        loss = 0.0
        loss_dict = {}
        batch_size = mu_J.shape[0]
        labels = torch.arange(batch_size).long()
        cnt = 0
        for i in range(n_views):
            v1 = self.linears[i](mu_perview_list[i])

            for j in range(i+1, n_views):
                cnt += 1
                
                v2 = self.linears[j](mu_perview_list[j])

                # Compute symmetric loss of every pair of views
                logits = torch.matmul(v1, v2.T)
                logits = logits - torch.max(logits, 1)[0][:, None]
                nce_loss = self.cross_entropy_loss(logits, labels)

                logits = torch.matmul(v2, v1.T)
                logits = logits - torch.max(logits, 1)[0][:, None]
                nce_loss += self.cross_entropy_loss(logits, labels)
                
                loss_dict['NCE_'+str(i)+'_'+str(j)] = nce_loss
                loss += nce_loss
                

        # print(cnt)
        loss_dict['NCE_full_graph'] = loss
        
        return loss, loss_dict