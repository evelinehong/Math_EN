#encoding: utf-8
import random

import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .baseRNN import BaseRNN
from .attention_1 import Attention_1

class DecoderRNN_3(BaseRNN):
    def __init__(self, vocab_size, class_size, embed_model=None, emb_size=100, hidden_size=128, \
                 n_layers=1, rnn_cell = None, rnn_cell_name='lstm', \
                 sos_id=1, eos_id=0, input_dropout_p=0, dropout_p=0): #use_attention=False):
        super(DecoderRNN_3, self).__init__(vocab_size, emb_size, hidden_size,
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

    def forward_normal_teacher_1(self, decoder_inputs, decoder_init_hidden, function):
        '''
        decoder_input: batch x seq_lengths x indices( sub last(-1), add first(sos_id))
        decoder_init_hidden: processed considering encoder layers, bi 
            lstm : h_0 (num_layers * num_directions, batch, hidden_size)
                   c_0 (num_layers * num_directions, batch, hidden_size)
            gru  : 
        decoder_outputs: batch x seq_lengths x classes,  probility_log
            lstm : h_n (num_layers * num_directions, batch, hidden_size)
                   c_n (num_layers * num_directions, batch, hidden_size)
            gru  :
        decoder_hidden: layers x batch x hidden_size 
        '''
        decoder_outputs, decoder_hidden = self.forward_step(\
                          decoder_inputs, decoder_init_hidden, function=function)
        decoder_outputs_list = []
        sequence_symbols_list = []
        for di in range(decoder_outputs.size(1)):
            step_output = decoder_outputs[:, di, :]
            symbols = self.decode(di, step_output)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list

    def forward_normal_teacher(self, decoder_inputs, decoder_init_hidden, encoder_outputs, function):
        decoder_outputs_list = []
        sequence_symbols_list = []
        #attn_list = []
        decoder_hidden = decoder_init_hidden
        seq_len = decoder_inputs.size(1)
        for di in range(seq_len):
            decoder_input = decoder_inputs[:, di]
            #deocder_input = torch.unsqueeze(decoder_input, 1)
            #print '1', deocder_input.size()
            decoder_output, decoder_hidden = self.forward_step(\
                decoder_input, decoder_hidden, encoder_outputs, function=function)
            #attn_list.append(attn)
            step_output = decoder_output.squeeze(1)
            if self.use_rule_old == False:
                symbols = self.decode(di, step_output)
            else:
                symbols = self.decode_rule(di, sequence_symbols_list, step_output)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list#, attn_list

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


    def forward_normal_no_teacher(self, decoder_input, decoder_init_hidden, encoder_outputs,\
                                                 max_length,  function, num_list, fix_rng):
        '''
        decoder_input: batch x 1
        decoder_output: batch x 1 x classes,  probility_log
        '''

        decoder_outputs_list = []
        sequence_symbols_list = []
        #attn_list = []
        decoder_hidden = decoder_init_hidden

        batch_size = decoder_input.size(0)
        classes_len = len(self.class_list)
        mask_op = torch.ones((batch_size, classes_len), dtype=torch.bool)
        filters_op = np.append(self.filter_op(), self.filter_END())
        mask_op[:,filters_op] = 0

        mask_digit = torch.ones((batch_size, classes_len), dtype=torch.bool)
        filters_digit = []
        for k,v in self.class_dict.items():
            if 'temp' in k or 'PI' == k or k.isdigit():
                filters_digit.append(v)
        filters_digit = np.array(filters_digit)
        mask_digit[:, filters_digit] = 0

        mask_temp = torch.ones((batch_size, classes_len), dtype=torch.bool)
        
        if num_list is not None:
            for i in range (len(num_list)):
                filters_temp = []

                for k,v in self.class_dict.items():
                    if 'temp' in k:
                        if (ord(k[5]) - ord('a') >= len(num_list[i])):
                            filters_temp.append(v)
                filter_temp = np.array (filters_temp)
                mask_temp[i, filters_temp] = 0

        ended = torch.zeros(batch_size, dtype=torch.bool)
        for di in range(max_length):
            decoder_output, decoder_hidden = self.forward_step(\
                           decoder_input, decoder_hidden, encoder_outputs, function=function)
            #attn_list.append(attn)
            step_output = decoder_output.squeeze(1)
            step_output = torch.exp(step_output) # batch_size * classes_len

            mask = torch.ones((batch_size, classes_len))

            mask_pad = torch.ones((batch_size, classes_len), dtype=torch.bool)
            for i in range (batch_size):
                if ended[i]:
                    all_except_pad = list(range(classes_len))
                    all_except_pad.remove(self.filter_PAD())
                    mask_pad[i, all_except_pad] = 0
                else:
                    mask_pad[i, self.filter_PAD()] = 0
            
            mask_end = torch.ones((batch_size, classes_len), dtype=torch.bool)
            for i in range (batch_size):
                if not ended[i]:
                    all_except_end = list(range(classes_len))
                    all_except_end.remove(self.filter_END())
                    mask_end[i, all_except_end] = 0

            if self.use_rule:
                if di % 2 == 0:
                    mask = mask * mask_op
                else:
                    mask = mask * mask_digit

                mask = mask * mask_temp
                mask = mask * mask_pad
                if di == max_length - 1:
                    mask = mask * mask_end

            # fixRng-like:
            # force EOS if greater than twice num_list length-1 (+2 for occasional constants)
            # force not EOS if less than half num_list length (make configurable?)
            if fix_rng and num_list is not None:
                max_len = list(map(lambda ls: len(ls)*2-1 + 2, num_list))
                min_len = list(map(lambda ls: len(ls) // 2, num_list))

                mask_rng = torch.ones((batch_size, classes_len), dtype=torch.bool)
                for i in range(batch_size):
                    if di < min_len[i]:
                        mask_rng[i, self.filter_END()] = 0
                    elif di == max_len[i]:
                        if not ended[i]:
                            all_except_end = list(range(classes_len))
                            all_except_end.remove(self.filter_END())
                            mask_rng[i, all_except_end] = 0
                mask = mask * mask_rng
                
            mask[mask == 0] = 1e-12

            if self.use_cuda:
                mask = mask.cuda()
            step_output = mask * step_output
            step_output = torch.log(step_output)

            if self.use_rule_old == False:
                symbols = self.decode(di, step_output)
            else:
                step_output, symbols = self.decode_rule(di, sequence_symbols_list, step_output)

            decoder_input = self.symbol_norm(symbols)

            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)

            ended = ended | (symbols.flatten().cpu() == self.class_dict['END_token']).bool()

        return decoder_outputs_list, decoder_hidden, sequence_symbols_list#, attn_list


    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, template_flag=True,\
                function=F.log_softmax, teacher_forcing_ratio=0, use_rule=False, use_cuda=False, \
                vocab_dict = None, vocab_list = None, class_dict = None, class_list = None, num_list = None,
                fix_rng=False, use_rule_old=False):
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

        if template_flag == False:
            max_length = 40
        else:
            max_length = inputs.size(1)

        #inputs = torch.cat((pad_var, inputs), 1) # careful concate  batch x (seq_len+1)
        #inputs = inputs[:, :-1] # batch x seq_len

        if use_teacher_forcing:
            ''' all steps together'''
            inputs = torch.cat((pad_var, inputs), 1) # careful concate  batch x (seq_len+1)
            inputs = inputs[:, :-1] # batch x seq_len
            decoder_inputs = inputs
            return self.forward_normal_teacher(decoder_inputs, decoder_init_hidden, encoder_outputs,\
                                                             function)
        else:
            #decoder_input = inputs[:,0].unsqueeze(1) # batch x 1
            decoder_input = pad_var#.unsqueeze(1) # batch x 1
            #pdb.set_trace()
            return self.forward_normal_no_teacher(decoder_input, decoder_init_hidden, encoder_outputs,\
                                                  max_length, function, num_list, fix_rng)


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

