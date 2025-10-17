from .cvdataset import cvdataset_read

def get_dataloader(args):
    train_dataloaders, val_dataloaders, test_loader, net_dataidx_map, traindata_cls_counts, data_distributions = cvdataset_read(args.dataset, args.datadir, args.batch_size, args.n_parties, args.partition, args.beta, args.skew_class)
    return train_dataloaders, val_dataloaders, test_loader, net_dataidx_map, traindata_cls_counts, data_distributions