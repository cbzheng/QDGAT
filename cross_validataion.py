import pickle
from torch.utils.data import DataLoader, RandomSampler
from qdgat.drop_dataloader import DropBatchGen
from qdgat.drop_dataloader import create_collate_fn
# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()


def cross_validation(
    network,
    data_dir,
    args,
    tokenizer,
    train_func,
    eval_func,
    logger,
    k=5
):
    collate_fn = create_collate_fn(tokenizer.pad_token_id, args.use_cuda)
    eval_loss_list, eval_f1_list, eval_em_list = [], [], []

    for i in range(k):
        logger.info(
            f'------------------------- Cross validation: fold {i} -------------------------')

        train_paths = [
            '{}/train_cv_{}.pkl'.format(data_dir, j) for j in range(k) if i != j]
        dev_path = '{}/dev_cv_{}.pkl'.format(data_dir, i)

        # prepare the training data
        train_data = []
        for dpath in train_paths:
            with open(dpath, "rb") as f:
                print("Load data from {}.".format(dpath))
                data = pickle.load(f)
                train_data.extend(data)
        train_dataset = DropBatchGen(
            args, data_mode="train", tokenizer=tokenizer, loaded_data=train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataset = DataLoader(train_dataset, sampler=train_sampler,
                                   batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=False)

        # prepare the dev data
        dev_data = []
        with open(dev_path, "rb") as f:
            print("Load data from {}.".format(dev_path))
            dev_data = pickle.load(f)
        dev_dataset = DropBatchGen(
            args, data_mode="dev", tokenizer=tokenizer, loaded_data=dev_data)
        dev_dataset = DataLoader(dev_dataset, batch_size=args.eval_batch_size,
                                 num_workers=0, collate_fn=collate_fn, pin_memory=False, shuffle=False)

        train_func(args, network, train_dataset, dev_dataset)
        eval_loss, eval_f1, eval_em = eval_func(args, network, dev_dataset)
        eval_loss_list.append(eval_loss)
        eval_f1_list.append(eval_f1)
        eval_em_list.append(eval_em)

    logger.info("Cross validation results: F1 {}, EM {}.".format(
        sum(eval_f1_list)/len(eval_f1_list), sum(eval_em_list)/len(eval_em_list)))
    return eval_loss_list, eval_f1_list, eval_em_list
