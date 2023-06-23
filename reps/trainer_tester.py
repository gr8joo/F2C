from .utils import AverageMeter


def trainer(epoch,
            model, train_loader, cuda,
            optimizer,
            lambda_views,
            anneal_epochs, N_mini_batches, log_interval):
    model.train()
    method = model.get_method()
    train_loss_meter = AverageMeter()

    for batch_idx, data in enumerate(train_loader):
        views = data[0]
        action = data[1]
        if epoch < anneal_epochs:   # compute the KL annealing factor for the current mini-batch in the current epoch
            anneal_factor = (float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                                float(anneal_epochs * N_mini_batches))
        else:                               # by default the KL annealing factor is unity
            anneal_factor = 1.0

        batch_size = len(views[0])

        # refresh the optimizer
        optimizer.zero_grad()

        # compute loss to train your model
        if method == 'MVTCAE':
            # pass data through model
            views_recon, mu_J, logvar_J, mu_perview_list, logvar_perview_list = model(views)

            # compute TC objective
            joint_loss, logs = model.compute_loss(views, views_recon,
                                                  mu_J, logvar_J, mu_perview_list, logvar_perview_list,
                                                  lambda_views=lambda_views,
                                                  anneal_factor=anneal_factor)
            train_loss = joint_loss
        
        elif method == "CMC":
            # pass data through model
            mu_J, mu_perview_list = model(views)

            # compute CMC objective
            joint_loss, logs = model.compute_loss(mu_J, mu_perview_list,
                                                  lambda_views=lambda_views,
                                                  anneal_factor=anneal_factor)
            train_loss = joint_loss

        elif method in ['MVSSM', 'SLAC']:
            # pass data through model
            views_recon, mu_J, logvar_J, mu_perview_list, logvar_perview_list, mu_prior, logvar_prior, z = model(views, action)

            # compute Sequential TC objective
            joint_loss, logs = model.compute_loss(views, action, views_recon,
                                                  mu_J, logvar_J, mu_perview_list, logvar_perview_list, mu_prior, logvar_prior, z,
                                                  lambda_views=lambda_views,
                                                  anneal_factor=anneal_factor)
            train_loss = joint_loss
        else:
            raise ValueError("Incorrect method name ", method)

        # logging
        train_loss_meter.update(train_loss.data, batch_size)

        # compute gradients and take step
        train_loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tanneal-Factor: {:.3f}'.format(
                epoch, batch_idx * len(views[0]), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss_meter.avg, anneal_factor))

    # tb_logger.write_train_logs(logs)
    print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))
    return logs


def tester(epoch, model, valid_loader, lambda_views):
    model.eval()
    method = model.get_method()
    eval_loss_meter = AverageMeter()

    for batch_idx, data in enumerate(valid_loader):
        views = data[0]
        action = data[1]
        anneal_factor = 1.0
        batch_size = len(views[0])

        # compute loss to train your model
        if method == 'MVTCAE':
            # pass data through model
            views_recon, mu_J, logvar_J, mu_perview_list, logvar_perview_list = model(views)

            # compute TC objective
            joint_loss, logs = model.compute_loss(views, views_recon,
                                                  mu_J, logvar_J, mu_perview_list, logvar_perview_list,
                                                  lambda_views=lambda_views,
                                                  anneal_factor=anneal_factor)
            train_loss = joint_loss
        
        elif method == "CMC":
            # pass data through model
            mu_J, mu_perview_list = model(views)

            # compute CMC objective
            joint_loss, logs = model.compute_loss(mu_J, mu_perview_list,
                                                  lambda_views=lambda_views,
                                                  anneal_factor=anneal_factor)
            train_loss = joint_loss

        elif method in ['MVSSM', 'SLAC']:
            # pass data through model
            views_recon, mu_J, logvar_J, mu_perview_list, logvar_perview_list, mu_prior, logvar_prior, z = model(views, action)

            # compute Sequential TC objective
            joint_loss, logs = model.compute_loss(views, action, views_recon,
                                                  mu_J, logvar_J, mu_perview_list, logvar_perview_list, mu_prior, logvar_prior, z,
                                                  lambda_views=lambda_views,
                                                  anneal_factor=anneal_factor)
            train_loss = joint_loss
            
        else:
            raise ValueError("Incorrect method name ", method)

        # logging
        eval_loss_meter.update(train_loss.data, batch_size)

    avg_loss = eval_loss_meter.avg
    print('======> Valid: {}\tLoss: {:.4f}'.format(epoch, eval_loss_meter.avg))
    return logs, avg_loss
