import torch
import torch.nn as nn

from .submodels import AttributeEncoder, AttributeDecoder, IVWAvg
from .utils import prior_expert, reparameterize, l2_loss, kl_loss, bce_loss

class MVSSM(nn.Module):
    def __init__(self, rec_coeff, activation, n_latents, view_sizes, hidden_size, action_size, view_binaries, use_prior_expert=False, method='MVSSM'):
        super(MVSSM, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.sa_encoder = AttributeEncoder(activation, n_latents, n_latents + action_size)

        for i in range(len(view_sizes)):
            self.encoders.append(AttributeEncoder(activation, n_latents, view_sizes[i], hidden_size))
            self.decoders.append(AttributeDecoder(activation, n_latents, view_sizes[i], hidden_size, is_binary=view_binaries[i]))

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduce=False)
        self.linear_z_a = nn.Linear(n_latents + action_size, hidden_size)
        self.linear_z_prime = nn.Linear(n_latents, hidden_size)

        self.experts = IVWAvg()
        self.use_prior_expert = use_prior_expert
        self.n_latents = n_latents
        self.view_sizes = view_sizes
        self.hidden_size = hidden_size
        self.view_binaries = view_binaries
        self.method = method
        self.rec_coeff = rec_coeff

    def get_method(self):
        return self.method

    def forward(self, views, action):
        # Enc
        mu_J, logvar_J, mu_perview_list, logvar_perview_list, mu_prior, logvar_prior, z = self.infer(views, action, None)

        # Dec
        z_reshape = z.view(-1, self.n_latents)
        views_recon = []
        for i in range(len(mu_perview_list)):
            views_recon.append( self.decoders[i](z_reshape))

        return views_recon, mu_J, logvar_J, mu_perview_list, logvar_perview_list, mu_prior, logvar_prior, z

    def infer(self, views, action, use_prior, use_subset=None):
        batch_size = views[0].size(0)
        seq_len = views[0].size(1)

        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        mu_perview_list = []
        logvar_perview_list = []

        mu_prior_seq = []
        logvar_prior_seq = []
        z_seq = []
        mu_J_seq = []
        logvar_J_seq = []

        if use_subset is None:
            use_subset = [True]*len(views)

        n_views = 0
        for i in range(len(views)):
            if use_subset[i] is False:
                continue
            n_views += 1
            rest = views[i].size(2)
            mu_perview, logvar_perview = self.encoders[i](views[i].view(-1, rest))
            mu_perview_list.append(mu_perview.view(1, batch_size, seq_len, self.n_latents))
            logvar_perview_list.append(logvar_perview.view(1, batch_size, seq_len, self.n_latents))

        mus = torch.cat(mu_perview_list, dim=0)
        logvars = torch.cat(logvar_perview_list, dim=0)

        mu_perview_list = [mu.view(batch_size, seq_len, -1) for mu in mu_perview_list]
        logvar_perview_list = [logvar.view(batch_size, seq_len, -1) for logvar in logvar_perview_list]


        if use_prior is not None:
            # initialize the universal prior expert
            mu_prior_per_step, logvar_prior_per_step = self.sa_encoder(torch.cat([use_prior, action[:, 0, :]], dim=-1))
        else:
            mu_prior_per_step = None
            logvar_prior_per_step = None


        for t in range(seq_len):
            mus_per_step = mus[:, :, t, :]
            logvars_per_step = logvars[:, :, t, :]

            if mu_prior_per_step is not None:
                # include to IVW the prior information inferred from the previous state-action
                mus_per_step = torch.cat([mu_prior_per_step.unsqueeze(0), mus_per_step], dim=0)
                logvars_per_step = torch.cat([logvar_prior_per_step.unsqueeze(0), logvars_per_step], dim=0)
            else:
                # Only latents inferred from V views will be included to IVW computation
                mu_prior_per_step, logvar_prior_per_step = prior_expert((batch_size, self.n_latents), use_cuda=use_cuda)

            mu_prior_seq.append(mu_prior_per_step.unsqueeze(1))
            logvar_prior_seq.append(logvar_prior_per_step.unsqueeze(1))

            # Apply IVW to combine per-view latents
            mu_J_per_step, logvar_J_per_step = self.experts(mus_per_step, logvars_per_step)
            mu_J_seq.append(mu_J_per_step.unsqueeze(1))
            logvar_J_seq.append(logvar_J_per_step.unsqueeze(1))

            z_per_step = reparameterize(mu_J_per_step, logvar_J_per_step, self.training)
            z_seq.append(z_per_step.unsqueeze(1))

            if seq_len > 1:
                action_per_step = action[:, t, :]
                mu_prior_per_step, logvar_prior_per_step = self.sa_encoder(torch.cat([z_per_step, action_per_step], dim=-1))


        # Stack per-view latents to compute loss
        mu_J = torch.cat(mu_J_seq, dim=1)                   # batch_size, seq_len, latent_dim
        logvar_J = torch.cat(logvar_J_seq, dim=1)           # batch_size, seq_len, latent_dim
        z = torch.cat(z_seq, dim=1)                         # batch_size, seq_len, latent_dim
        mu_prior = torch.cat(mu_prior_seq, dim=1)           # batch_size, seq_len, latent_dim
        logvar_prior = torch.cat(logvar_prior_seq, dim=1)   # batch_size, seq_len, latent_dim

        return mu_J, logvar_J, mu_perview_list, logvar_perview_list, mu_prior, logvar_prior, z#, mu_prior_per_step, logvar_prior_per_step

    def compute_loss(self, views, action, views_recon, mu_J, logvar_J, mu_perview_list, logvar_perview_list, mu_prior, logvar_prior, z,
                     lambda_views, anneal_factor=1):

        n_views = len(views)
        if self.method == 'MVSSM':
            rec_weight = n_views / (n_views+1.0)
            vib_weight = 1.0 / (n_views+1.0)
            cvib_weight= 1.0 / (n_views+1.0)
            contrast_weight = rec_weight
        elif self.method == 'SLAC':
            rec_weight = 1.0
            vib_weight = 1.0
            cvib_weight = 0.0
            contrast_weight = 0.0

        batch_size = action.size(0)
        seq_len = action.size(1)
        action_dim = action.size(2)
        # action = action.view(-1, action_dim)
        if len(views[0].shape) > len(views_recon[0].shape):
            # seq_len = views[0].size(1)
            views = [ views[i].view(views_recon[i].shape) for i in range(len(views)) ]

        # Recon & KL losses
        views_rec_loss =[]
        views_kl_loss = []
        TC = 0
        loss_dict = {}
        for i in range(len(views)):
            if self.view_binaries[i]:
                views_rec_loss.append( bce_loss(views_recon[i], views[i]).view(batch_size, seq_len) )
            else:
                views_rec_loss.append( l2_loss(views_recon[i], views[i]).view(batch_size, seq_len) )

            views_kl_loss.append( kl_loss(mu_J, logvar_J, mu_perview_list[i], logvar_perview_list[i]) )

            TC += self.rec_coeff * rec_weight * lambda_views[i] * views_rec_loss[i]\
                  + anneal_factor * cvib_weight * views_kl_loss[i]

            loss_dict['view' + str(i+1) + '_rec'] = torch.mean(views_rec_loss[i])
            loss_dict['KL' + str(i+1)] = torch.mean(views_kl_loss[i])

        prior_kl_loss = kl_loss(mu_J, logvar_J, mu_prior, logvar_prior)
        TC += anneal_factor * vib_weight * prior_kl_loss
        loss_dict['KL0'] = torch.mean(prior_kl_loss)

        # InfoNCE loss
        labels = torch.arange(batch_size).long()
        nce_loss = 0.0
        for t in range(1, seq_len):
            emb_z_prime = self.linear_z_prime(z[:, t, :])
            emb_z_a = self.linear_z_a(torch.cat([z[:, t-1, :], action[:, t-1, :]], dim=-1))

            logits = torch.matmul(emb_z_a, emb_z_prime.T)
            logits = logits - torch.max(logits, 1)[0][:, None]
            nce_loss += (1.0 / (seq_len - 1.0)) * self.cross_entropy_loss(logits, labels)

        loss_dict['NCE'] = torch.mean(nce_loss)
        TC = torch.mean(TC) + contrast_weight*loss_dict['NCE']


        return TC, loss_dict
