#encoding: utf-8
import random
import sys

import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .baseRNN import BaseRNN
from .attention_1 import Attention_1

DEFAULT_NUM_STEPS = 21

def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)

def min_len_single(num_list):
    return len(num_list)*2 - 1
def max_len_single(num_list):
    return len(num_list)*2 + 3


def mask_batch(batch_select, class_select):
    return ~(batch_select[:, None] * class_select[None, :])


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output, symbols):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output
        self.symbols = symbols

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class DecoderRNN(BaseRNN):
    def __init__(self, vocab_size, class_size, embed_model=None, emb_size=100, hidden_size=128, \
                 n_layers=1, rnn_cell = None, rnn_cell_name='lstm', \
                 sos_id=1, eos_id=0, input_dropout_p=0, dropout_p=0): #use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size, emb_size, hidden_size,
                                         input_dropout_p, dropout_p,
                                         n_layers, rnn_cell_name)
        self.vocab_size = vocab_size
        self.class_size = class_size
        self.sos_id = sos_id
        self.eos_id = eos_id

        if embed_model == None:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        else:
            self.embedding = embed_model

        if rnn_cell == None:
            self.rnn = self.rnn_cell(emb_size, hidden_size, n_layers, \
                                     batch_first=True, dropout=dropout_p)
        else:
            self.rnn = rnn_cell

        self.out = nn.Linear(self.hidden_size, self.class_size)
        self.attention = Attention_1(hidden_size)

    #def _init_state(self, encoder_hidden, op_type):


    def forward_step(self, input_var, hidden, encoder_outputs, function):
        '''
        normal forward, step by step or all steps together
        '''

        if len(input_var.size()) == 1:
            input_var = torch.unsqueeze(input_var,1)
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)

        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(\
                            output.contiguous().view(-1, self.hidden_size)), dim=1)\
                            .view(batch_size, output_size, -1)
        return predicted_softmax, hidden#, #attn

    def decode(self, step, step_output):
        '''
        step_output: batch x classes , prob_log
        symbols: batch x 1
        '''
        symbols = step_output.topk(1)[1]
        return symbols

    def decode_rule(self, step, sequence_symbols_list, step_output):
        symbols = self.rule_filter(sequence_symbols_list, step_output)
        return symbols

    def symbol_norm(self, symbols):
        symbols = symbols.view(-1).data.cpu().numpy()
        new_symbols = []
        for idx in symbols:
            #print idx,
            #print self.class_list[idx],
            #pdb.set_trace()
            #print self.vocab_dict[self.class_list[idx]]
            new_symbols.append(self.vocab_dict[self.class_list[idx]])
        new_symbols = Variable(torch.LongTensor(new_symbols))
        #print new_symbols
        new_symbols = torch.unsqueeze(new_symbols, 1)
        if self.use_cuda:
            new_symbols = new_symbols.cuda()
        return new_symbols

    def forward_normal_no_teacher_beam(self, decoder_input, decoder_init_hidden, encoder_outputs,\
                                                 function, num_list, target_length, mask_const, beam_size=5):
        '''
        decoder_input: batch x 1
        max_length: including END

        decoder_output: batch x 1 x classes,  probility_log

        Must be batch size 1
        '''

        #attn_list = []
        decoder_hidden = decoder_init_hidden

        batch_size = decoder_input.size(0)
        classes_len = len(self.class_list)
        filters_op = self.filter_op()
        only_op = torch.zeros(classes_len, dtype=torch.bool)
        only_op[filters_op] = 1

        filters_digit = []
        for k,v in self.class_dict.items():
            if 'temp' in k or 'PI' == k or k.isdigit():
                filters_digit.append(v)
        filters_digit = np.array(filters_digit)
        only_digit = torch.zeros(classes_len, dtype=torch.bool)
        only_digit[filters_digit] = 1

        filters_const = []
        for k, v in self.class_dict.items():
            if 'PI' == k or k.isdigit():
                filters_const.append(v)
        filters_const = np.array(filters_const)

        mask_temp = torch.ones((batch_size, classes_len), dtype=torch.bool)
        if num_list is not None:
            for i in range (len(num_list)):
                filters_temp = []

                for k,v in self.class_dict.items():
                    if 'temp' in k:
                        if (ord(k[5]) - ord('a') >= len(num_list[i])):
                            filters_temp.append(v)
                filters_temp = np.array (filters_temp)
                mask_temp[i, filters_temp] = 0

        only_pad = torch.zeros(classes_len, dtype=torch.bool)
        only_pad[self.filter_PAD()] = 1

        only_end = torch.zeros(classes_len, dtype=torch.bool)
        only_end[self.filter_END()] = 1

        ended = torch.zeros(batch_size, dtype=torch.bool)
        generated_ops = torch.zeros(batch_size, dtype=torch.int)
        generated_nums = torch.zeros(batch_size, dtype=torch.int)
        # if target_length is not None:
        #     target_length = np.array(target_length)
        #     target_length = target_length - 1  # exclude END
        #     target_length -= (target_length % 2 == 0) # force odd
        #     target_length[target_length > max_len.numpy()] = 0
        #     target_length = torch.tensor(target_length)
        #     max_len = target_length


        if self.use_cuda:
            ended = ended.cuda()
            generated_ops = generated_ops.cuda()
            generated_nums = generated_nums.cuda()
            only_pad = only_pad.cuda()
            only_end = only_end.cuda()
            only_digit = only_digit.cuda()
            only_op = only_op.cuda()
            # if target_length is not None:
            #     target_length = target_length.cuda()
            mask_temp = mask_temp.cuda()

        max_target_length = DEFAULT_NUM_STEPS + 1
        all_decoder_outputs = torch.zeros(max_target_length, batch_size, classes_len)
        beam_list = list()
        score = torch.zeros(batch_size)
        if self.use_cuda:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs, []))

        for di in range(max_target_length):
            temp_list = list()
            beam_len = len(beam_list)
            for xb in beam_list[:]:
                if di!=0 and int(xb.input_var[0][0]) == self.eos_id:
                    temp_list.append(xb)
                    beam_list.remove(xb)
                    beam_len -= 1
                if di != 0 and int(xb.input_var[0][0]) == 0: #padding without eos
                    beam_list.remove(xb)
                    beam_len -= 1
            if beam_len == 0:
                break

            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, classes_len * beam_len)
            all_hidden = torch.zeros(len(decoder_hidden), decoder_hidden[0].size(0), batch_size*beam_len, decoder_hidden[0].size(2))
            all_outputs =  torch.zeros(max_target_length, batch_size * beam_len, classes_len)
            if self.use_cuda:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                decoder_output, decoder_hidden = self.forward_step(\
                               decoder_input, decoder_hidden, encoder_outputs, function=function)
                if torch.any(torch.isnan(decoder_output)):
                    print("nananananan")
                    sys.exit(1)
                #attn_list.append(attn)
                step_output = decoder_output.squeeze(1)
                step_output = torch.exp(step_output) # batch_size * classes_len

                mask = torch.ones((batch_size, classes_len))
                if self.use_cuda:
                    mask = mask.cuda()

                # if self.use_rule:
                #     mask *= mask_batch(ended, ~only_pad) # if ended, all except pad are 0
                #     mask *= mask_batch(~ended, only_pad) # if not ended, only pad is 0
                #
                #     if di==0 or di==1:
                #         mask[:, filters_op] = 0  # first two elements are numbers
                #     mask *= mask_batch(generated_nums - 1 == generated_ops, only_op) # number of ops cannot be greater than number of nums
                #
                #     mask *= mask_batch((max_len == 0) & (di==0), ~only_end) # if max len is 0, immediately generate END
                #     mask *= mask_batch((max_len != 0) & (generated_nums - 1 != generated_ops), only_end) # otherwise allow END only when nums match ops
                #
                #     if target_length is not None:
                #         mask *= mask_batch(generated_nums == (target_length + 1) / 2, only_digit)  # number of nums cannot be greater than (target_length+1)/2
                #         mask *= mask_batch(di < target_length, only_end)  # min length
                #         mask *= mask_batch(di == target_length, ~only_end)
                #
                #     if mask_const:
                #         mask[:, filters_const] = 0  # for the first iterations, do not generate 1 and 3.14
                #
                #     mask = mask * mask_temp

                masked_step_output = step_output * mask
                masked_step_output[masked_step_output==0] = 1e-30
                masked_step_output = torch.log(masked_step_output)


                score = masked_step_output

                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score = beam_score + (score-beam_score)/(di+1)
                # score += beam_score
                beam_scores[:, b_idx * classes_len: (b_idx + 1) * classes_len] = score
                all_hidden[0, :, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden[0]
                all_hidden[1, :, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden[1]

                beam_list[b_idx].all_output[di] = masked_step_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output

                # if self.use_rule_old == False:
                # symbols = self.decode(di, masked_step_output)
                # else:
                #     step_output, symbols = self.decode_rule(di, sequence_symbols_list, step_output)

                # preds = symbols.flatten()
                # generated_nums += isin(preds, torch.tensor(filters_digit).cuda()).int()
                # generated_ops += isin(preds, torch.tensor(self.filter_op()).cuda()).int()

                # decoder_input = self.symbol_norm(symbols)

                # ended = ended | (preds == self.class_dict['END_token']).bool()
            topv, topi = beam_scores.topk(beam_size, dim=1)

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % classes_len
                temp_input = temp_input.data
                temp_beam_pos = temp_topk / classes_len

                indices = torch.LongTensor(range(batch_size))
                if self.use_cuda:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(2, indices)
                temp_output = all_outputs.index_select(1, indices)

                decoder_input = self.symbol_norm(temp_input)

                old_symbols = beam_list[temp_beam_pos].symbols

                beam = Beam(topv[:, k], decoder_input, (temp_hidden[0], temp_hidden[1]), temp_output, old_symbols + [temp_input.unsqueeze(1)])
                temp_list.append(beam)

            temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

            if len(temp_list) < beam_size:
                beam_list = temp_list
            else:
                beam_list = temp_list[:beam_size]

        beam_list = temp_list
        
        res = []

        for k in range(len(beam_list)):
            all_decoder_outputs = beam_list[k].all_output

            new_symbols = [symbol for symbol in beam_list[k].symbols]
            if k < beam_size:
                diff = True
                for prev_res in res:
                    if prev_res[2] == new_symbols:
                        diff = False
                        break
                if diff:
                    res.append((all_decoder_outputs, beam_list[k].hidden, new_symbols))#, attn_list

        return res


    def forward_normal_no_teacher(self, decoder_input, decoder_init_hidden, encoder_outputs,\
                                                 function, num_list, target_length, mask_const):
        '''
        decoder_input: batch x 1
        max_length: including END

        decoder_output: batch x 1 x classes,  probility_log
        '''

        decoder_outputs_list = []
        sequence_symbols_list = []
        #attn_list = []
        decoder_hidden = decoder_init_hidden

        batch_size = decoder_input.size(0)
        classes_len = len(self.class_list)
        filters_op = self.filter_op()
        only_op = torch.zeros(classes_len, dtype=torch.bool)
        only_op[filters_op] = 1

        filters_digit = []
        for k,v in self.class_dict.items():
            if 'temp' in k or 'PI' == k or k.isdigit():
                filters_digit.append(v)
        filters_digit = np.array(filters_digit)
        only_digit = torch.zeros(classes_len, dtype=torch.bool)
        only_digit[filters_digit] = 1

        filters_const = []
        for k, v in self.class_dict.items():
            if 'PI' == k or k.isdigit():
                filters_const.append(v)
        filters_const = np.array(filters_const)

        mask_temp = torch.ones((batch_size, classes_len), dtype=torch.bool)
        if num_list is not None:
            for i in range (len(num_list)):
                filters_temp = []

                for k,v in self.class_dict.items():
                    if 'temp' in k:
                        if (ord(k[5]) - ord('a') >= len(num_list[i])):
                            filters_temp.append(v)
                filters_temp = np.array (filters_temp)
                mask_temp[i, filters_temp] = 0

        only_pad = torch.zeros(classes_len, dtype=torch.bool)
        only_pad[self.filter_PAD()] = 1

        only_end = torch.zeros(classes_len, dtype=torch.bool)
        only_end[self.filter_END()] = 1

        ended = torch.zeros(batch_size, dtype=torch.bool)
        generated_ops = torch.zeros(batch_size, dtype=torch.int)
        generated_nums = torch.zeros(batch_size, dtype=torch.int)

        if target_length is not None:
            target_length = torch.tensor(target_length)
            if self.use_cuda:
                target_length = target_length.cuda()

        if self.use_cuda:
            ended = ended.cuda()
            generated_ops = generated_ops.cuda()
            generated_nums = generated_nums.cuda()
            only_pad = only_pad.cuda()
            only_end = only_end.cuda()
            only_digit = only_digit.cuda()
            only_op = only_op.cuda()
            mask_temp = mask_temp.cuda()

        total_steps = DEFAULT_NUM_STEPS
        if target_length is not None:
            total_steps = max(target_length)
        for di in range(total_steps+1):
            decoder_output, decoder_hidden = self.forward_step(\
                           decoder_input, decoder_hidden, encoder_outputs, function=function)
            if torch.any(torch.isnan(decoder_output)):
                print("nananananan")
                sys.exit(1)
            #attn_list.append(attn)
            step_output = decoder_output.squeeze(1)
            step_output = torch.exp(step_output) # batch_size * classes_len

            mask = torch.ones((batch_size, classes_len))
            if self.use_cuda:
                mask = mask.cuda()

            if self.use_rule:
                mask *= mask_batch(ended, ~only_pad) # if ended, all except pad are 0
                mask *= mask_batch(~ended, only_pad) # if not ended, only pad is 0

                if di==0 or di==1:
                    mask[:, filters_op] = 0  # first two elements are numbers
                mask *= mask_batch(generated_nums - 1 == generated_ops, only_op) # number of ops cannot be greater than number of nums

                if target_length is not None:
                    mask *= mask_batch((target_length == 0) & (di==0), ~only_end) # if max len is 0, immediately generate END
                    mask *= mask_batch((target_length != 0) & (generated_nums - 1 != generated_ops), only_end) # otherwise allow END only when nums match ops

                    mask *= mask_batch(generated_nums == (target_length + 1) / 2, only_digit)  # number of nums cannot be greater than (target_length+1)/2
                    mask *= mask_batch(di < target_length, only_end)  # min length
                    mask *= mask_batch(di == target_length, ~only_end)
                else:
                    mask *= mask_batch(generated_nums - 1 != generated_ops, only_end)  # allow END only when nums match ops

                if mask_const:
                    mask[:, filters_const] = 0  # for the first iterations, do not generate 1 and 3.14

                mask = mask * mask_temp

            if torch.any(torch.all(mask.bool()==0, 1)):
                for i in range(batch_size):
                    if torch.all(mask[i] == 0):
                        gen_temp = [self.class_list[id] for id in torch.cat(sequence_symbols_list, 1)[i]] if len(sequence_symbols_list)>0 else "[]"
                        print(f"PROBLEM: mask is all zero, di {di}, num_list {num_list[i]}, "
                              f"generated_ops {generated_ops[i]}, generated_nums {generated_nums[i]}, gen_temp {gen_temp}")
                        sys.exit(1)

            masked_step_output = step_output * mask

            masked_step_output[masked_step_output==0] = 1e-30

            masked_step_output = torch.log(masked_step_output)
            step_output = torch.log(step_output)

            # if self.use_rule_old == False:
            symbols = self.decode(di, masked_step_output)
            # else:
            #     step_output, symbols = self.decode_rule(di, sequence_symbols_list, step_output)

            preds = symbols.flatten()
            generated_nums += isin(preds, torch.tensor(filters_digit).cuda()).int()
            generated_ops += isin(preds, torch.tensor(self.filter_op()).cuda()).int()

            decoder_input = self.symbol_norm(symbols)

            decoder_outputs_list.append(masked_step_output)
            sequence_symbols_list.append(symbols)

            ended = ended | (preds == self.class_dict['END_token']).bool()

            if torch.all(ended):
                break

        return decoder_outputs_list, decoder_hidden, sequence_symbols_list#, attn_list

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, 
                function=F.log_softmax, teacher_forcing_ratio=0, use_rule=False, use_cuda=False, \
                vocab_dict = None, vocab_list = None, class_dict = None, class_list = None, num_list = None,
                use_rule_old=False, target_length=None, mask_const=False, beam_size=None):
        '''
        使用rule的时候，teacher_forcing_rattio = 0
        '''
        self.use_rule = use_rule
        self.use_rule_old = use_rule_old
        self.use_cuda = use_cuda
        self.class_dict = class_dict
        self.class_list = class_list
        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        #pdb.set_trace()
        batch_size = encoder_outputs.size(0)
        #batch_size = inputs.size(0)

        pad_var = torch.LongTensor([self.sos_id]*batch_size) # marker

        pad_var = Variable(pad_var.view(batch_size, 1))#.cuda() # marker
        if self.use_cuda:
            pad_var = pad_var.cuda()

        decoder_init_hidden = encoder_hidden

        #inputs = torch.cat((pad_var, inputs), 1) # careful concate  batch x (seq_len+1)
        #inputs = inputs[:, :-1] # batch x seq_len

        if use_teacher_forcing:
            ''' all steps together'''
            inputs = torch.cat((pad_var, inputs), 1) # careful concate  batch x (seq_len+1)
            inputs = inputs[:, :-1] # batch x seq_len
            decoder_inputs = inputs
            return self.forward_normal_teacher(decoder_inputs, decoder_init_hidden, encoder_outputs,\
                                                             function)
        elif beam_size>1:
            #decoder_input = inputs[:,0].unsqueeze(1) # batch x 1
            decoder_input = pad_var#.unsqueeze(1) # batch x 1
            #pdb.set_trace()
            return self.forward_normal_no_teacher_beam(decoder_input, decoder_init_hidden, encoder_outputs,\
                                                  function, num_list, target_length, mask_const, beam_size)
        else:
            # decoder_input = inputs[:,0].unsqueeze(1) # batch x 1
            decoder_input = pad_var  # .unsqueeze(1) # batch x 1
            # pdb.set_trace()
            return self.forward_normal_no_teacher(decoder_input, decoder_init_hidden, encoder_outputs, \
                                                       function, num_list, target_length, mask_const)


    def rule(self, symbol):
        filters = []
        if self.class_list[symbol] in ['+', '-', '*', '/']:
            filters.append(self.class_dict['+'])
            filters.append(self.class_dict['-'])
            filters.append(self.class_dict['*'])
            filters.append(self.class_dict['/'])
            filters.append(self.class_dict[')'])
            filters.append(self.class_dict['='])
        elif self.class_list[symbol] == '=':
            filters.append(self.class_dict['+'])
            filters.append(self.class_dict['-'])
            filters.append(self.class_dict['*'])
            filters.append(self.class_dict['/'])
            filters.append(self.class_dict['='])
            filters.append(self.class_dict[')'])
        elif self.class_list[symbol] == '(':
            filters.append(self.class_dict['('])
            filters.append(self.class_dict[')'])
            filters.append(self.class_dict['+'])
            filters.append(self.class_dict['-'])
            filters.append(self.class_dict['*'])
            filters.append(self.class_dict['/'])
            filters.append(self.class_dict['='])
        elif self.class_list[symbol] == ')':
            filters.append(self.class_dict['('])
            filters.append(self.class_dict[')'])
            for k,v in self.class_dict.items():
                if 'temp' in k:
                    filters.append(v)
        elif 'temp' in self.class_list[symbol]:
            filters.append(self.class_dict['('])
            filters.append(self.class_dict['='])
        return np.array(filters)

    def filter_op(self):
        filters = []
        filters.append(self.class_dict['+'])
        filters.append(self.class_dict['-'])
        filters.append(self.class_dict['*'])
        filters.append(self.class_dict['/'])
        filters.append(self.class_dict['^'])
        return np.array(filters)

    def filter_END(self):
        filters = []
        filters.append(self.class_dict['END_token'])
        return np.array(filters)
    
    def filter_PAD(self):
        filters = []
        filters.append(self.class_dict['PAD_token'])
        return np.array(filters)


    def rule_filter(self, sequence_symbols_list, current):
        '''
        32*28
        '''
        op_list = ['+','-','*','/','^']
        cur_out = current
        #print len(sequence_symbols_list)
        #pdb.set_trace()
        cur_symbols = []
        if sequence_symbols_list == [] or len(sequence_symbols_list) <= 1:
            #filters = self.filter_op()
            filters = np.append(self.filter_op(), self.filter_END())
            for i in range(cur_out.shape[0]):
                cur_out[i][filters] = -float('inf')
                # cur_symbols.append(np.argmax(cur_out[i]))
                cur_symbols = cur_out.topk(1)[1]
        # else:
        #     for i in range(sequence_symbols_list[0].size(0)):
        #         num_var = 0
        #         num_op = 0
        #         for j in range(len(sequence_symbols_list)):
        #             symbol = sequence_symbols_list[j][i].cpu().data[0]
        #             if self.class_list[symbol] in op_list:
        #                 num_op += 1
        #             elif 'temp' in self.class_list[symbol] or self.class_list[symbol] in ['1', 'PI']:
        #                 num_var += 1
        #         if num_var >= num_op + 2:
        #             filters = self.filter_END()
        #             cur_out[i][filters] = -float('inf')
        #         elif num_var == num_op + 1:
        #             filters = self.filter_op() 
        #             cur_out[i][filters] = -float('inf')
        #         cur_symbols.append(np.argmax(cur_out[i]))
        else:
            for i in range(sequence_symbols_list[0].size(0)):
                symbol = sequence_symbols_list[-1][i].cpu().data[0]
                filters = []
                if self.class_list[symbol] in op_list:
                    filters = np.append(self.filter_op(), self.filter_END())

                elif 'temp' in self.class_list[symbol] or self.class_list[symbol] in ['1', 'PI']:
                    for k,v in self.class_dict.items():
                        if 'temp' in k:
                            filters.append(v)
                    filters = np.array(filters)

                cur_out[i][filters] = -float('inf')
                cur_symbols = cur_out.topk(1)[1]

        # cur_symbols = Variable(torch.LongTensor(cur_symbols))
        # cur_symbols = torch.unsqueeze(cur_symbols, 1)
        # cur_symbols = cur_symbols.cuda()
        return cur_out, cur_symbols

