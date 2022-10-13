import sys
sys.path.append('../')
from torch.utils.data import DataLoader
import torch
import argparse
import os
from util.util import *
from train.eval import *
from clustering.domain_split import domain_split
from dataloader.dataloader import random_split_dataloader
import wandb
import hydra
import logger
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg):
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--data-root', default='/data/unagi0/kurose/data/MS/Facial_expression/')
    # parser.add_argument('--save-root', default='/data/unagi0/kurose/experiment_data/MS/Facial_expression_DG')
    # parser.add_argument('--result-dir', default='default')
    # parser.add_argument('--train', default='deepall')
    # parser.add_argument('--model', default='caffenet')
    # parser.add_argument('--clustering', action='store_true')
    # parser.add_argument('--clustering-method', default='Kmeans')
    # parser.add_argument('--num-clustering', type=int, default=3)
    # parser.add_argument('--clustering-step', type=int, default=1)
    # parser.add_argument('--entropy', choices=['default', 'maximum_square'])

    # parser.add_argument('--exp-num', type=int, default=0)
    # parser.add_argument('--gpu', type=int, default=0)

    # parser.add_argument('--num-epoch', type=int, default=30)
    # parser.add_argument('--eval-step', type=int, default=1)
    # parser.add_argument('--save-step', type=int, default=100)

    # parser.add_argument('--batch-size', type=int, default=128)
    # parser.add_argument('--scheduler', default='step')
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--lr-step', type=int, default=24)
    # parser.add_argument('--lr-decay-gamma', type=float, default=0.1)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--weight-decay', type=float, default=5e-4)
    # parser.add_argument('--nesterov', action='store_true')

    # parser.add_argument('--fc-weight', type=float, default=1.0)
    # parser.add_argument('--disc-weight', type=float, default=1.0)
    # parser.add_argument('--entropy-weight', type=float, default=1.0)
    # parser.add_argument('--grl-weight', type=float, default=1.0)
    # parser.add_argument('--loss-disc-weight', action='store_true')

    # parser.add_argument('--color-jitter', action='store_true')
    # parser.add_argument('--min-scale', type=float, default=0.8)

    # parser.add_argument('--instance-stat', action='store_true')
    # parser.add_argument('--feature-fixed', action='store_true')
    # args = parser.parse_args()

    wandb.init(project='Facial_expression_DG', name='{}'.format(cfg.out))
    wbcfg = wandb.config
    wbcfg['gpu'] = cfg.gpu
    wbcfg['dataset_list_dir'] = cfg.dataset_list_dir
    wbcfg['epoch'] = cfg.epoch
    wbcfg['data'] = cfg.data
    wbcfg['test_domain'] = cfg.test_domain
    wbcfg['train'] = cfg.train
    wbcfg['batch_size'] = cfg.batch_size

    wbcfg['eval_step'] = cfg.eval_step
    wbcfg['save_step'] = cfg.save_step
    wbcfg['color_jitter'] = cfg.color_jitter
    wbcfg['min_scale'] = cfg.min_scale

    wbcfg['clustering'] = cfg.clustering
    wbcfg['clustering_step'] = cfg.clustering_step
    wbcfg['lr_step'] = cfg.lr_step
    wbcfg['num_clustering'] = cfg.num_clustering

    wbcfg['scheduler'] = cfg.scheduler
    wbcfg['lr'] = cfg.lr
    wbcfg['lr_decay_gamma'] = cfg.lr_decay_gamma
    wbcfg['weight_decay'] = cfg.weight_decay
    wbcfg['momentum'] = cfg.momentum
    wbcfg['nesterov'] = cfg.nesterov
    wbcfg['fc_weight'] = cfg.fc_weight
    wbcfg['disc_weight'] = cfg.disc_weight
    wbcfg['loss_disc_weight'] = cfg.loss_disc_weight
    wbcfg['entropy_weight'] = cfg.entropy_weight
    wbcfg['grl_weight'] = cfg.grl_weight
    wbcfg['feature_fixed'] = cfg.feature_fixed
    wbcfg['instance_stat'] = cfg.instance_stat
    wbcfg['data_server'] = cfg.data_server


    # if not os.path.isdir(path):
    #     os.makedirs(path)
    #     os.makedirs(path + '/models')
    #
    # with open(path + '/args.txt', 'w') as f:
    #     f.write(str(args))
    # initialize logger
    log = logger.Logger(cfg, wandb=wandb)
    path = log.out_dir

    domain = get_domain(cfg.data)
    source_domain, target_domain = split_domain(domain, cfg.test_domain)

    device = torch.device("cuda:" + str(cfg.gpu) if torch.cuda.is_available() else "cpu")
    get_domain_label, get_cluster = train_to_get_label(cfg.train, cfg.clustering)

    source_train, source_val, target_test = random_split_dataloader(
        dataset_list_dir=cfg.dataset_list_dir, source_domain=source_domain, target_domain=target_domain,
        batch_size=cfg.batch_size, get_domain_label=get_domain_label, get_cluster=get_cluster, num_workers=3,
        color_jitter=cfg.color_jitter, min_scale=cfg.min_scale, num_class=cfg.num_class, data_server=cfg.data_server)

    #     num_epoch = int(args.num_iteration / len(source_train))
    #     lr_step = int(args.lr_step / min([len(domain) for domain in source_train]))
    #     print(num_epoch)

    num_epoch = cfg.epoch
    lr_step = cfg.lr_step

    disc_dim = get_disc_dim(cfg.train, cfg.clustering, len(source_domain), cfg.num_clustering)

    model = get_model(cfg.model, cfg.train)(
        num_classes=source_train.dataset.num_class, num_domains=disc_dim, pretrained=True)

    model = model.to(device)
    model_lr = get_model_lr(cfg.model, cfg.train, model, fc_weight=cfg.fc_weight, disc_weight=cfg.disc_weight)
    optimizers = [get_optimizer(model_part, cfg.lr * alpha, cfg.momentum, cfg.weight_decay,
                                cfg.feature_fixed, cfg.nesterov, per_layer=False) for model_part, alpha in model_lr]

    if cfg.scheduler == 'inv':
        schedulers = [get_scheduler(cfg.scheduler)(optimizer=opt, alpha=10, beta=0.75, total_epoch=num_epoch)
                      for opt in optimizers]
    elif cfg.scheduler == 'step':
        schedulers = [get_scheduler(cfg.scheduler)(optimizer=opt, step_size=lr_step, gamma=cfg.lr_decay_gamma)
                      for opt in optimizers]
    else:
        raise ValueError('Name of scheduler unknown %s' % cfg.scheduler)

    best_acc = 0.0
    test_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epoch):

        print('Epoch: {}/{}, Lr: {:.6f}'.format(epoch, num_epoch - 1, optimizers[0].param_groups[0]['lr']))
        print('Temporary Best Accuracy is {:.4f} ({:.4f} at Epoch {})'.format(test_acc, best_acc, best_epoch))

        dataset = source_train.dataset

        if cfg.clustering:
            if epoch % cfg.clustering_step == 0:
                pseudo_domain_label = domain_split(dataset, model, device=device,
                                                   cluster_before=dataset.clusters,
                                                   filename=os.path.join(path, 'nmi.txt'), epoch=epoch,
                                                   nmb_cluster=cfg.num_clustering, method=cfg.clustering_method,
                                                   pca_dim=256, whitening=False, L2norm=False,
                                                   instance_stat=cfg.instance_stat, num_workers=3)
                dataset.set_cluster(np.array(pseudo_domain_label))

        if cfg.loss_disc_weight:
            if cfg.clustering:
                hist = dataset.clusters
            else:
                hist = dataset.domains

            weight = 1. / np.histogram(hist, bins=model.num_domains)[0]
            weight = weight / weight.sum() * model.num_domains
            weight = torch.from_numpy(weight).float().to(device)

        else:
            weight = None

        model, optimizers = get_train(cfg.train)(
            model=model, train_data=source_train, optimizers=optimizers, device=device,
            epoch=epoch, num_epoch=num_epoch, logger=log, filename=path + '/source_train.txt', entropy=cfg.entropy,
            disc_weight=weight, entropy_weight=cfg.entropy_weight, grl_weight=cfg.grl_weight)

        if epoch % cfg.eval_step == 0:
            loss, acc = eval_model(model, source_val, device, epoch, path + '/source_eval.txt')
            loss_, acc_ = eval_model(model, target_test, device, epoch, path + '/target_test.txt')
            log.add_scalar('val/source_loss', loss, epoch)
            log.add_scalar('val/target_loss', loss_, epoch)
            log.add_scalar('val/source_acc', acc, epoch)
            log.add_scalar('val/target_acc', acc_, epoch)

        if epoch % cfg.save_step == 0:
            torch.save(model.state_dict(), os.path.join(
                path, 'models',
                "model_{}.pt".format(epoch)))

        if acc >= best_acc:
            best_acc = acc
            test_acc = acc_
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                path, 'models',
                "model_best.pt"))

        for scheduler in schedulers:
            scheduler.step()

    best_model = get_model(cfg.model, cfg.train)(num_classes=source_train.dataset.num_class,
                                                 num_domains=disc_dim, pretrained=False)
    best_model.load_state_dict(torch.load(os.path.join(
        path, 'models',
        "model_best.pt"), map_location=device))
    best_model = best_model.to(device)
    test_acc = eval_model(best_model, target_test, device, best_epoch, path + '/target_best.txt')
    print('Test Accuracy by the best model on the source domain is {} (at Epoch {})'.format(test_acc, best_epoch))


if __name__ == '__main__':

    main()

