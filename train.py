import argparse
import shutil
import model_ as M
import nnue_dataset
import nnue_bin_dataset
import features
import torch
import torch.nn.functional as F
from torch import set_num_threads as t_set_num_threads
from torch.utils.data import DataLoader, Dataset

from torch import nn
import copy

import ranger_adabelief
import ranger
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

def data_loader_cc(train_filename, val_filename, feature_set, num_workers, batch_size, filtered, random_fen_skipping,
                   main_device, epoch_size=100000000, val_size=10000000):
    # Epoch and validation sizes are arbitrary
    features_name = feature_set.name
    train_infinite = nnue_dataset.SparseBatchDataset(features_name, train_filename, batch_size, num_workers=num_workers,
                                                     filtered=filtered, random_fen_skipping=random_fen_skipping,
                                                     device=main_device)
    val_infinite = nnue_dataset.SparseBatchDataset(features_name, val_filename, batch_size, filtered=filtered,
                                                   random_fen_skipping=random_fen_skipping, device=main_device)
    # num_workers has to be 0 for sparse, and 1 for dense
    # it currently cannot work in parallel mode but it shouldn't need to
    train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size),
                       batch_size=None, batch_sampler=None)
    val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size),
                     batch_size=None, batch_sampler=None)
    return train, val


def data_loader_py(train_filename, val_filename, batch_size, feature_set, main_device):
    train = DataLoader(nnue_bin_dataset.NNUEBinData(train_filename, feature_set), batch_size=batch_size, shuffle=True, num_workers=4)
    val = DataLoader(nnue_bin_dataset.NNUEBinData(val_filename, feature_set), batch_size=32)
    return train, val


def nnue_loss(q, t, score, lambda_):

    p = (score / 600.0).sigmoid()
    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    result  = lambda_ * teacher_loss    + (1.0 - lambda_) * outcome_loss
    entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    loss = result.mean() - entropy.mean()
    return loss

def save_ckp(state, checkpoint_dir):
    f_path = checkpoint_dir + 'best_model.pt'
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

def main():
    parser = argparse.ArgumentParser(description="Trains the network.")
    parser.add_argument("train", help="Training data (.bin or .binpack)")
    parser.add_argument("val", help="Validation data (.bin or .binpack)")

    parser.add_argument("--tune", action="store_true", help="automated LR search")
    parser.add_argument("--save", action="store_true", help="save after every training epoch (default = False)")
    parser.add_argument("--experiment", default="1", type=str, help="specify the experiment id")
    parser.add_argument("--py-data", action="store_true", help="Use python data loader (default=False)")
    parser.add_argument("--lambda", default=1.0, type=float, dest='lambda_',
                        help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).")
    parser.add_argument("--num-workers", default=1, type=int, dest='num_workers',
                        help="Number of worker threads to use for data loading. Currently only works well for binpack.")
    parser.add_argument("--batch-size", default=-1, type=int, dest='batch_size',
                        help="Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128.")
    parser.add_argument("--threads", default=-1, type=int, dest='threads',
                        help="Number of torch threads to use. Default automatic (cores) .")
    parser.add_argument("--seed", default=42, type=int, dest='seed', help="torch seed to use.")
    parser.add_argument("--smart-fen-skipping", action='store_true', dest='smart_fen_skipping',
                        help="If enabled positions that are bad training targets will be skipped during loading. Default: False")
    parser.add_argument("--random-fen-skipping", default=0, type=int, dest='random_fen_skipping',
                        help="skip fens randomly on average random_fen_skipping before using one.")
    parser.add_argument("--resume-from-model", dest='resume_from_model',
                        help="Initializes training using the weights from the given .pt model")

    features.add_argparse_args(parser)
    args = parser.parse_args()

    print("Training with {} validating with {}".format(args.train, args.val))

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    batch_size = args.batch_size
    if batch_size <= 0:
        batch_size = 128 if args.gpus == 0 else 8192
    print('Using batch size {}'.format(batch_size))

    print('Smart fen skipping: {}'.format(args.smart_fen_skipping))
    print('Random fen skipping: {}'.format(args.random_fen_skipping))

    if args.threads > 0:
        print('limiting torch to {} threads.'.format(args.threads))
        t_set_num_threads(args.threads)

    feature_set = features.get_feature_set_from_name(args.features)

    if args.py_data:
        print('Using python data loader')
        train_data, val_data = data_loader_py(args.train, args.val, batch_size, feature_set, 'cuda:0')

    else:
        print('Using c++ data loader')
        train_data, val_data = data_loader_cc(args.train, args.val, feature_set, args.num_workers, batch_size,
                                    args.smart_fen_skipping, args.random_fen_skipping, 'cuda:0')

    print("Feature set: {}".format(feature_set.name))
    print("Num real features: {}".format(feature_set.num_real_features))
    print("Num virtual features: {}".format(feature_set.num_virtual_features))
    print("Num features: {}".format(feature_set.num_features))

    START_EPOCH = 0
    NUM_EPOCHS = 300

    LEARNING_RATE = 5e-4
    DECAY = 0
    EPS = 1e-8

    best_loss = 1000
    is_best = False

    early_stopping_delay = 30
    early_stopping_count = 0
    early_stopping_flag = False

    summary_location = 'logs/nnue_experiment_' + args.experiment
    save_location = '/home/esigelec/PycharmProjects/nnue-pytorch/save_models/' + args.experiment

    writer = SummaryWriter(summary_location)

    nnue = M.NNUE(feature_set=feature_set, lambda_=args.lambda_, s=1)

    train_params = [{'params': nnue.get_1xlr(), 'lr': LEARNING_RATE},
                    {'params': nnue.get_10xlr(), 'lr': LEARNING_RATE * 10.0}]

    optimizer = ranger.Ranger(train_params,lr=LEARNING_RATE, eps=EPS, betas=(0.9, 0.999), weight_decay=DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, cooldown=1, min_lr=1e-7, verbose=True)

    if args.resume_from_model is not None:
        nnue, optimizer, START_EPOCH = load_ckp(args.resume_from_model, nnue, optimizer)
        nnue.set_feature_set(feature_set)
        for state in optimizer.state.values():
           for k, v in state.items():
               if isinstance(v, torch.Tensor):
                   state[k] = v.cuda()

    nnue = nnue.cuda()

    for epoch in range(START_EPOCH, NUM_EPOCHS + START_EPOCH):

        nnue.train()

        train_interval = 100
        loss_f_sum_interval = 0.0
        loss_f_sum_epoch = 0.0
        loss_v_sum_epoch = 0.0

        if early_stopping_flag:
            print("early end of training at epoch" + str(epoch))
            break

        for batch_idx, batch in enumerate(train_data):

            batch = [_data.cuda() for _data in batch]
            us, them, white, black, outcome, score = batch

            optimizer.zero_grad()
            output = nnue(us, them, white, black)

            loss = nnue_loss(output, outcome, score, args.lambda_)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(nnue.parameters(), 0.5)
            optimizer.step()

            loss_f_sum_interval += loss.float()
            loss_f_sum_epoch += loss.float()

            if batch_idx % train_interval == train_interval - 1:

                writer.add_scalar('train_loss',
                                  loss_f_sum_interval / train_interval,
                                  epoch * len(train_data) + batch_idx)

                loss_f_sum_interval = 0.0

        print("Epoch #{}\t Train_Loss: {:.8f}\t".format(epoch, loss_f_sum_epoch / len(train_data)))

        if epoch % 1 == 0 or (epoch + 1) == NUM_EPOCHS:

            with torch.no_grad():
                nnue.eval()
                for batch_idx, batch in enumerate(val_data):
                    batch = [_data.cuda() for _data in batch]
                    us, them, white, black, outcome, score = batch

                    _output = nnue(us, them, white, black)
                    loss_v = nnue_loss(_output, outcome, score, args.lambda_)
                    loss_v_sum_epoch += loss_v.float()

            scheduler.step(loss_v_sum_epoch / len(val_data))

            writer.add_scalar('val_loss',
                              loss_v_sum_epoch / len(val_data),
                              epoch * len(train_data) + batch_idx)

            if loss_v_sum_epoch / len(val_data) <= best_loss:
                best_loss = loss_v_sum_epoch / len(val_data)
                is_best = True
                early_stopping_count = 0
            else:
                early_stopping_count += 1
                if early_stopping_delay == early_stopping_count:
                    early_stopping_flag = True


            if is_best:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': nnue.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                save_ckp(checkpoint, save_location)
                is_best = False

            print("Epoch #{}\tVal_Loss: {:.8f}\t".format(epoch, loss_v_sum_epoch / len(val_data)))

    writer.close()

if __name__ == '__main__':
    main()
