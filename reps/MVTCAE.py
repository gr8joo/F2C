import torch
import torch.nn as nn

from .submodels import AttributeEncoder, AttributeDecoder, IVWAvg
from .utils import prior_expert, reparameterize, l2_loss, kl_loss, bce_loss

class MVTCAE(nn.Module):
    def __init__(self, rec_coeff, activation, n_latents, view_sizes, hidden_size, view_binaries, use_prior_expert=False):
        super(MVTCAE, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(len(view_sizes)):
            self.encoders.append(AttributeEncoder(activation, n_latents, view_sizes[i], hidden_size))
            self.decoders.append(AttributeDecoder(activation, n_latents, view_sizes[i], hidden_size, is_binary=view_binaries[i]))

        self.experts = IVWAvg()
        self.use_prior_expert = use_prior_expert
        self.n_latents = n_latents
        self.view_sizes = view_sizes
        self.hidden_size = hidden_size
        self.view_binaries = view_binaries
        self.method = 'MVTCAE'
        self.rec_coeff = rec_coeff

    def get_method(self):
        return self.method

    def forward(self, views):
        # Enc
        mu_J, logvar_J, mu_perview_list, logvar_perview_list, z = self.infer(views, self.use_prior_expert)

        # Dec
        views_recon = []
        for i in range(len(mu_perview_list)):
            views_recon.append( self.decoders[i](z) )

        return views_recon, mu_J, logvar_J, mu_perview_list, logvar_perview_list

    def infer(self, views, use_prior_expert, use_subset=None):
        batch_size = views[0].size(0)
        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        mu_perview_list = []
        logvar_perview_list = []

        if use_subset is None:
            use_subset = [True]*len(views)

        if use_prior_expert:
            # initialize the universal prior expert
            mus, logvars = prior_expert((1, batch_size, self.n_latents), use_cuda=use_cuda)
        else:
            mus = None
            logvars = None

        for i in range(len(views)):
            if use_subset[i] is False:
                continue

            mu_perview, logvar_perview = self.encoders[i](views[i])
            mu_perview_list.append(mu_perview)
            logvar_perview_list.append(logvar_perview)

            if mus is not None:
                mus = torch.cat((mus, mu_perview.unsqueeze(0)), dim=0)
                logvars = torch.cat((logvars, logvar_perview.unsqueeze(0)), dim=0)
            else:
                mus = mu_perview.unsqueeze(0)
                logvars = logvar_perview.unsqueeze(0)

        # Apply IVW to combine per-view latents
        mu_J, logvar_J = self.experts(mus, logvars)
        z = reparameterize(mu_J, logvar_J, self.training)
        return mu_J, logvar_J, mu_perview_list, logvar_perview_list, z

    def compute_loss(self, views, views_recon, mu_J, logvar_J, mu_perview_list, logvar_perview_list,
                     lambda_views, anneal_factor=1):

        n_views = len(views)
        alpha = n_views / (n_views+1.0)
        rec_weight = (n_views-alpha) / n_views
        vib_weight = 1.0 - alpha
        cvib_weight= alpha / n_views

        views_rec_loss =[]
        views_kl_loss = []
        TC = 0
        loss_dict = {}
        for i in range(len(views)):
            if self.view_binaries[i]:
                views_rec_loss.append( bce_loss(views_recon[i], views[i]) )
            else:
                views_rec_loss.append( l2_loss(views_recon[i], views[i]) )

            views_kl_loss.append( kl_loss(mu_J, logvar_J, mu_perview_list[i], logvar_perview_list[i]) )

            loss_dict['view' + str(i+1) + '_rec'] = torch.mean(views_rec_loss[i])
            loss_dict['KL' + str(i+1)] = torch.mean(views_kl_loss[i])

            TC += self.rec_coeff * rec_weight * lambda_views[i] * views_rec_loss[i]\
                  + anneal_factor * cvib_weight * views_kl_loss[i]


        prior_kl_loss = kl_loss(mu_J, logvar_J)
        loss_dict['KL0'] = torch.mean(prior_kl_loss)

        TC += anneal_factor * vib_weight * prior_kl_loss
        TC = torch.mean(TC)

        return TC, loss_dict
