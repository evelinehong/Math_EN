import sys

from config import *
import pdb
import os
import logging
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import random

from utils import DataLoader
from train import SupervisedTrainer, BackTrainer
from model import EncoderRNN, DecoderRNN, Seq2seq
from utils import NLLLoss, Optimizer, Checkpoint, Evaluator

args = get_args()

if args.wandb:
    import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def train_backsearch():
    if args.mode == 0:
        encoder_cell = 'lstm'
        decoder_cell = 'lstm'
    elif args.mode == 1:
        encoder_cell = 'gru'
        decoder_cell = 'gru'
    elif args.mode == 2:
        encoder_cell = 'gru'
        decoder_cell = 'lstm'
    else:
        encoder_cell = 'lstm'
        decoder_cell = 'gru'

    data_loader = DataLoader(args)
    embed_model = nn.Embedding(data_loader.vocab_len, 128)
    #embed_model.weight.data.copy_(torch.from_numpy(data_loader.word2vec.emb_vectors))
    encode_model = EncoderRNN(vocab_size = data_loader.vocab_len,
                              embed_model = embed_model,
                              emb_size = 128,
                              hidden_size = 512,
                              input_dropout_p = 0.3,
                              dropout_p = 0.4,
                              n_layers = 2,
                              bidirectional = True,
                              rnn_cell = None,
                              rnn_cell_name = encoder_cell,
                              variable_lengths = True)
    decode_model = DecoderRNN(vocab_size = data_loader.vocab_len,
                              class_size = data_loader.classes_len,
                              embed_model = embed_model,
                              emb_size = 128,
                              hidden_size = 1024,
                              n_layers = 2,
                              rnn_cell = None,
                              rnn_cell_name=decoder_cell,
                              sos_id = data_loader.vocab_dict['END_token'],
                              eos_id = data_loader.vocab_dict['END_token'],
                              input_dropout_p = 0.3,
                              dropout_p = 0.4)
    seq2seq = Seq2seq(encode_model, decode_model)

    if args.cuda_use:
        seq2seq = seq2seq.cuda()

    if args.wandb:
        wandb.watch(seq2seq)

    weight = torch.ones(data_loader.classes_len)
    pad = data_loader.decode_classes_dict['PAD_token']
    loss = NLLLoss(weight, pad)

    st = BackTrainer(vocab_dict = data_loader.vocab_dict,
                   vocab_list = data_loader.vocab_list,
                   decode_classes_dict = data_loader.decode_classes_dict,
                   decode_classes_list = data_loader.decode_classes_list,
                   cuda_use = args.cuda_use,
                   loss = loss,
                   print_every = 1,
                   teacher_schedule = False,
                   checkpoint_dir_name = args.checkpoint_dir_name,
                   use_rule = args.use_rule,
                   n_step=args.n_step,
                   wandb=args.wandb)


    print ('start training')
    st.train(model = seq2seq,
             data_loader = data_loader,
             batch_size = 256,
             n_epoch = 200,
             resume = args.resume,
             optimizer = None,
             mode = args.mode,
             teacher_forcing_ratio=args.teacher_forcing_ratio,
             post_flag = args.post_flag)

def train_seq2seq():

    if args.mode == 0:
        encoder_cell = 'lstm'
        decoder_cell = 'lstm'
    elif args.mode == 1:
        encoder_cell = 'gru'
        decoder_cell = 'gru'
    elif args.mode == 2:
        encoder_cell = 'gru'
        decoder_cell = 'lstm'
    else:
        encoder_cell = 'lstm'
        decoder_cell = 'gru'

    data_loader = DataLoader(args)
    embed_model = nn.Embedding(data_loader.vocab_len, 128)
    #embed_model.weight.data.copy_(torch.from_numpy(data_loader.word2vec.emb_vectors))
    encode_model = EncoderRNN(vocab_size = data_loader.vocab_len,
                              embed_model = embed_model,
                              emb_size = 128,
                              hidden_size = 512,
                              input_dropout_p = 0.3,
                              dropout_p = 0.4,
                              n_layers = 2,
                              bidirectional = True,
                              rnn_cell = None,
                              rnn_cell_name = encoder_cell,
                              variable_lengths = True)
    decode_model = DecoderRNN(vocab_size = data_loader.vocab_len,
                              class_size = data_loader.classes_len,
                              embed_model = embed_model,
                              emb_size = 128,
                              hidden_size = 1024,
                              n_layers = 2,
                              rnn_cell = None,
                              rnn_cell_name=decoder_cell,
                              sos_id = data_loader.vocab_dict['END_token'],
                              eos_id = data_loader.vocab_dict['END_token'],
                              input_dropout_p = 0.3,
                              dropout_p = 0.4)
    seq2seq = Seq2seq(encode_model, decode_model)

    if args.cuda_use:
        seq2seq = seq2seq.cuda()

    if args.wandb:
        wandb.watch(seq2seq)

    weight = torch.ones(data_loader.classes_len)
    pad = data_loader.decode_classes_dict['PAD_token']
    loss = NLLLoss(weight, pad)

    st = SupervisedTrainer(vocab_dict = data_loader.vocab_dict,
                           vocab_list = data_loader.vocab_list,
                           decode_classes_dict = data_loader.decode_classes_dict,
                           decode_classes_list = data_loader.decode_classes_list,
                           cuda_use = args.cuda_use,
                           loss = loss,
                           print_every = 10,
                           teacher_schedule = False,
                           checkpoint_dir_name = args.checkpoint_dir_name,
                           use_rule = False)


    print ('start training')
    st.train(model = seq2seq,
             data_loader = data_loader,
             batch_size = 256,
             n_epoch = 200,
             template_flag = True,
             resume = args.resume,
             optimizer = None,
             mode = args.mode,
             teacher_forcing_ratio=args.teacher_forcing_ratio,
             post_flag = args.post_flag)

def test_23k():

    data_loader = DataLoader(args)

    #Checkpoint.CHECKPOINT_DIR_NAME = "0120_0030"
    Checkpoint.CHECKPOINT_DIR_NAME = args.checkpoint_dir_name
    checkpoint_path = os.path.join("./experiment", Checkpoint.CHECKPOINT_DIR_NAME, "best")
    checkpoint = Checkpoint.load(checkpoint_path)

    seq2seq = checkpoint.model
    if args.cuda_use:
        seq2seq = seq2seq.cuda()

    seq2seq.eval()
    evaluator = Evaluator(vocab_dict = data_loader.vocab_dict,
                          vocab_list = data_loader.vocab_list,
                          decode_classes_dict = data_loader.decode_classes_dict,
                          decode_classes_list = data_loader.decode_classes_list,
                          loss = NLLLoss(),
                          cuda_use = args.cuda_use)
    name = args.run_flag
    #if name == 'test_23k':
    beam_sizes = [1, 3,5]
    ans_accs = []
    for beam_size in beam_sizes:
        test_temp_acc, test_ans_acc = evaluator.evaluate(model = seq2seq,
                                                     data_loader = data_loader,
                                                     data_list = data_loader.math23k_test_list,
                                                     batch_size = 1,
                                                     evaluate_type = 0,
                                                     use_rule = False,
                                                     mode = args.mode,
                                                     post_flag=args.post_flag,
                                                     name_save = name,
                                                     use_rule_old=False,
                                                     beam_size=beam_size)
        ans_accs.append(test_ans_acc)

    print (ans_accs)

if __name__ == "__main__":
    if args.resume and args.id is None:
        print('resume must provide id')
        sys.exit(1)

    if args.wandb:
        wandb.init(project="mwp-postfix-ma-final", id=(args.id if args.resume else None), sync_tensorboard=True)
        wandb.config.use_rule = args.use_rule
        wandb.config.run_flag = args.run_flag
        wandb.config.n_step = args.n_step
        wandb.config.seed = args.seed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if 'test_23k' in args.run_flag:
        test_23k()
    elif args.run_flag == 'train_23k':
        train_seq2seq()
    elif args.run_flag == 'backsearch':
        train_backsearch()
    else:
        print ('emmmm..................')
