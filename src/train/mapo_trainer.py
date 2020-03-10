import sys
import random
import numpy as np
from model import EncoderRNN, DecoderRNN_1, Seq2seq
from utils import NLLLoss, Optimizer, Checkpoint, Evaluator

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import pdb

torch.manual_seed(7)
class MAPOTrainer(object):
    def __init__(self, vocab_dict, vocab_list, decode_classes_dict, decode_classes_list, cuda_use, \
                  loss, print_every, teacher_schedule, checkpoint_dir_name, fix_rng, use_rule):
        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list
        self.decode_classes_dict = decode_classes_dict
        self.decode_classes_list = decode_classes_list
        self.class_dict = decode_classes_dict
        self.class_list = decode_classes_list

        

        self.pad_in_classes_idx = self.decode_classes_dict['PAD_token']
        self.end_in_classes_idx = self.decode_classes_dict['END_token']

        random_seed = 10
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.cuda_use = cuda_use
        self.loss = loss 
        if self.cuda_use == True:
            self.loss.cuda()

        self.fix_rng = fix_rng
        self.use_rule = use_rule

        self.print_every = print_every

        self.teacher_schedule = teacher_schedule

        Checkpoint.CHECKPOINT_DIR_NAME = checkpoint_dir_name

        
    def _convert_f_e_2_d_sybmbol(self, target_variable):
        new_variable = []
        batch,colums = target_variable.size()
        for i in range(batch):
            tmp = []
            for j in range(colums):
                #idx = self.decode_classes_dict[self.vocab_list[target_variable[i][j].data[0]]]
                idx = self.decode_classes_dict[self.vocab_list[target_variable[i][j].item()]]
                tmp.append(idx)
            new_variable.append(tmp)
        return Variable(torch.LongTensor(np.array(new_variable)))

    def inverse_temp_to_num(self, equ_list, num_list):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        new_equ_list = []
        for elem in equ_list:
            if 'temp' in elem:
                index = alphabet.index(elem[-1])
                new_equ_list.append(num_list[index])
            elif 'PI' == elem:
                new_equ_list.append('3.14')
            else:
                new_equ_list.append(elem)
        return new_equ_list

    def inverse_temp_to_num_(self, equ_list, num_list):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        new_equ_list = []
        for elem in equ_list:
            if 'temp' in elem:
                index = alphabet.index(elem[-1])
                try:
                    new_equ_list.append(str(num_list[index]))
                except:
                    return []
            elif 'PI' == elem:
                new_equ_list.append('3.14')
            else:
                new_equ_list.append(elem)
        return new_equ_list

    def get_new_tempalte(self, seq_var, num_list):
        equ_list = []
        for idx in seq_var.data.cpu().numpy():
            if idx == self.pad_in_classes_idx:
                break
            equ_list.append(self.decode_classes_list[idx])
        equ_list = self.inverse_temp_to_num_(equ_list, num_list)
        try:
            equ_list = equ_list[:equ_list.index('END_token')]
        except:
            pass
        return equ_list

    def compute_gen_ans(self, seq_var, num_list, post_flag):

        equ_list = []
        for idx in seq_var.data.cpu().numpy():
            if idx == self.pad_in_classes_idx:
                break
            equ_list.append(self.decode_classes_list[idx])
        equ_list = self.inverse_temp_to_num_(equ_list, num_list)
        try:
            equ_list = equ_list[:equ_list.index('END_token')]
        except:
            pass

        equ_string = ''
        for equ in equ_list:
            if equ.endswith(".0"):
                equ_string += equ[:-2]
            elif equ == "^":
                equ_string += "**"
            else:
                equ_string += equ
        try:
            ans = eval(equ_string)
        except:
            ans = 10000.0
        return ans        


    def _train_batch(self, input_variables, input_lengths,target_variables, target_lengths, model,\
                         template_flag, teacher_forcing_ratio, mode, batch_size, post_flag, num_list, solutions, buffer, train_queue):
        decoder_outputs, decoder_hidden, symbols_list = \
                                      model(input_variable = input_variables, 
                                      input_lengths = input_lengths, 
                                      target_variable = target_variables, 
                                      template_flag = template_flag,
                                      teacher_forcing_ratio = teacher_forcing_ratio, 
                                      mode = mode,
                                      use_rule = self.use_rule,
                                      use_cuda = self.cuda_use,
                                      vocab_dict = self.vocab_dict,
                                      vocab_list = self.vocab_list,
                                      class_dict = self.class_dict,
                                      class_list = self.class_list,
                                      num_list = num_list,
                                      fix_rng = self.fix_rng,
                                      use_rule_old = False)

        # cuda
        
        target_original = target_variables
        target_variables = self._convert_f_e_2_d_sybmbol(target_variables)

        pad_in_classes_idx = self.decode_classes_dict['PAD_token']
        batch_size = len(input_lengths)

        match = 0
        total = 0

        seq = symbols_list
        seq_var = torch.cat(seq, 1)
        batch_pg = []
        #batch_att = []
        selected_probs = torch.zeros(batch_size)



        right = 0
        gen_answers = []
        batch_buffer = buffer.copy()
        batch_queue = train_queue.copy()
        for i in range(batch_size):
            gen_ans = self.compute_gen_ans(seq_var[i], num_list[i], post_flag)
            gen_equ = self.get_new_tempalte(seq_var[i], num_list[i])
            gt_equ = self.get_new_tempalte(target_variables[i], num_list[i])

            try:
                if abs(float(gen_ans) - solutions[i]) < 1e-5:
                    seq_list = seq_var[i].cpu().data.tolist()
                    batch_buffer[i].append(seq_list)
            except:
                pass

            for buffer in batch_buffer[i]:
                batch_queue.append({'input_variables': input_variables[i], 'input_lengths': input_lengths[i], 'target_variables': target_original[i], 'num_list': num_list[i], 'buffer': buffer, 'reward':0}) 
                   

        # reward_decay = 0.9
        # reward_moving_average = np.mean(rewards)
        # reward_moving_average = reward_moving_average * reward_decay \
        #     + np.mean(rewards) * (1 - reward_decay)
        # rewards = rewards - reward_moving_average
        
        # loss = -torch.tensor(rewards) * selected_probs
        # loss = loss.mean()
        # model.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        return batch_buffer, batch_queue

    def _train_mapo_batch(self, input_variables, input_lengths, target_variables, target_lengths, model,\
                         template_flag, teacher_forcing_ratio, mode, batch_size, post_flag, num_list, solutions, buffer_seq):
        if self.cuda_use:
            input_variables = input_variables.cuda()
        
        decoder_outputs, decoder_hidden, symbols_list = \
                                      model(input_variable = input_variables, 
                                      input_lengths = input_lengths, 
                                      target_variable = target_variables, 
                                      template_flag = template_flag,
                                      teacher_forcing_ratio = teacher_forcing_ratio, 
                                      mode = mode,
                                      use_rule = self.use_rule,
                                      use_cuda = self.cuda_use,
                                      vocab_dict = self.vocab_dict,
                                      vocab_list = self.vocab_list,
                                      class_dict = self.class_dict,
                                      class_list = self.class_list,
                                      num_list = num_list,
                                      fix_rng = self.fix_rng,
                                      use_rule_old = False)
        
        target_variables = self._convert_f_e_2_d_sybmbol(target_variables)
        if self.cuda_use:
            target_variables = target_variables.cuda()
            buffer_seq = buffer_seq.cuda()

        pad_in_classes_idx = self.decode_classes_dict['PAD_token']
        batch_size = len(input_lengths)

        match = 0
        total = 0

        seq = symbols_list
        seq_var = torch.cat(seq, 1)
        batch_pg = []
        #batch_att = []
        for i in range(batch_size):
            p_list = []
            #att_tmp_list = []
            total_p = 0
            for j in range(len(decoder_outputs)): 
                #mm_elem_idx = seq_var[i][j].cpu().data.numpy()[0]
                mm_elem_idx = seq_var[i][j].cpu().data.numpy().tolist()
                #cur_att = att_list[j][i,0,:].cpu().data.numpy().tolist()
                #print mm_elem_idx, self.end_in_classes_idx, mm_elem_idx == self.end_in_classes_idx
                if mm_elem_idx == self.end_in_classes_idx:
                    break
                num_p = decoder_outputs[j][i].topk(1)[0].cpu().data.numpy()[0]
                total_p += decoder_outputs[j][i].topk(1)[0]
                p_list.append(str(num_p))
                #att_tmp_list.append(cur_att)
            batch_pg.append(p_list)
            #batch_att.append(att_tmp_list)

        # selected_probs = torch.zeros(batch_size)
        # for i in range(batch_size):
        #     #att_tmp_list = []
        #     total_p = 0
        #     for j in range(len(decoder_outputs)): 
        #         #mm_elem_idx = seq_var[i][j].cpu().data.numpy()[0]
        #         mm_elem_idx = buffer_seq[i][j]
        #         #cur_att = att_list[j][i,0,:].cpu().data.numpy().tolist()
        #         #print mm_elem_idx, self.end_in_classes_idx, mm_elem_idx == self.end_in_classes_idx
        #         if mm_elem_idx == self.end_in_classes_idx:
        #             break
        #         num_p = decoder_outputs[j][i][buffer_seq[i][j]].item()
        #         total_p += decoder_outputs[j][i][buffer_seq[i][j]].item()
        #     selected_probs[i] = total_p

        # if self.cuda_use:
        #     selected_probs = selected_probs.cuda()

        self.loss.reset()
        buffer_seq = torch.transpose(buffer_seq, 0, 1)

        for step, step_output in enumerate(decoder_outputs):
            # cuda step_output = step_output.cuda()
            if self.cuda_use:
                step_output = step_output.cuda()
            target = target_variables[:, step]
            self.loss.eval_batch(step_output.contiguous().view(batch_size, -1), buffer_seq[step])
            non_padding = target.ne(pad_in_classes_idx)
            correct = seq[step].view(-1).eq(target).masked_select(non_padding).sum().item()#data[0]
            match += correct
            total += non_padding.sum().item()#.data[0]

        right = 0
        gen_answers = []

        for i in range(batch_size):
            for j in range(target_variables.size(1)):
                #if target_variables[i][j].data[0] != pad_in_classes_idx  and \
                              #target_variables[i][j].data[0] == seq_var[i][j].data[0]:
                if target_variables[i][j].item() != pad_in_classes_idx  and \
                              target_variables[i][j].item() == seq_var[i][j].item():
                    continue
                #elif target_variables[i][j].data[0] == 1:
                elif target_variables[i][j].item() == 1:
                    right += 1
                    break
                else:
                    break


        # reward_decay = 0.9
        # reward_moving_average = np.mean(rewards)
        # reward_moving_average = reward_moving_average * reward_decay \
        #     + np.mean(rewards) * (1 - reward_decay)
        # rewards = rewards - reward_moving_average
        model.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        return self.loss.get_loss(), [right, match, total]

    def _train_epoches(self, data_loader, model, batch_size, start_epoch, start_step, n_epoch, \
                            mode, template_flag, teacher_forcing_ratio, post_flag):
        print_loss_total = 0

        train_list = data_loader.math23k_train_list
        test_list = data_loader.math23k_test_list
        valid_list = data_loader.math23k_valid_list
        train_list = train_list + valid_list# = data_loader.math23k_valid_list
        steps_per_epoch = len(train_list)/batch_size
        total_steps = steps_per_epoch * n_epoch

        step = start_step
        step_elapsed = 0

        threshold = [0]+[1]*9

        max_ans_acc = 0 

        buffer = dict()
        for i in range (len(train_list)):
            buffer[i] = []       

        for epoch in range(start_epoch, n_epoch + 1):
            print ("epoch:"+str(epoch))
            epoch_loss_total = 0

            #marker if self.teacher_schedule:

            batch_generator = data_loader.get_batch(train_list, batch_size, True)
            
            right_count = 0
            match = 0
            total_m = 0
            total_r = 0
            step_batch = 0
            train_queue = []

            model.train(True)
            for batch_data_dict in batch_generator:
                step_batch += 1
                input_variables = batch_data_dict['batch_encode_pad_idx']
                input_lengths = batch_data_dict['batch_encode_len']
                target_variables = batch_data_dict['batch_decode_pad_idx']
                target_lengths = batch_data_dict['batch_decode_len']
                num_list = batch_data_dict['batch_num_list'] 
                solutions = batch_data_dict['batch_solution']
                
                batch_buffer = dict()
                for i in range(len(input_variables)):
                    batch_buffer[i] = buffer[(step_batch - 1) * batch_size + i]
               
                #cuda
                input_variables = Variable(torch.LongTensor(input_variables))
                target_variables = Variable(torch.LongTensor(target_variables))

                if self.cuda_use:
                    input_variables = input_variables.cuda()
                    target_variables = target_variables.cuda()

                buffer_batch, train_queue = self._train_batch(input_variables = input_variables, 
                                                   input_lengths = input_lengths, 
                                                   target_variables = target_variables, 
                                                   target_lengths = target_lengths, 
                                                   model = model, 
                                                   template_flag = template_flag,
                                                   teacher_forcing_ratio = teacher_forcing_ratio,
                                                   mode = mode, 
                                                   batch_size = batch_size,
                                                   post_flag = post_flag,
                                                   num_list = num_list,
                                                   solutions = solutions,
                                                   buffer = batch_buffer,
                                                   train_queue = train_queue)
                

                for i in range (len(input_variables)):
                    buffer[(step_batch - 1) * batch_size + i] = batch_buffer[i]
                    
            mapo_batch_size = 256
            batch_number = int(len(train_queue)/mapo_batch_size)
            
            for i in range (0, batch_number):
                step += 1
                step_elapsed += 1
                batch_queue = train_queue[i*batch_size:(i*batch_size+batch_size)]           
                
                max_target_len = max([len(queue['target_variables']) for queue in batch_queue])
                target_variables = torch.zeros((len(batch_queue), max_target_len), dtype=torch.long)
                max_input_len = max([len(queue['input_variables']) for queue in batch_queue])
                input_variables = torch.zeros((len(batch_queue), max_input_len), dtype=torch.long)

                max_buffer_len = max([len(queue['buffer']) for queue in batch_queue])
                buffer_seq = torch.zeros((len(batch_queue), max_buffer_len), dtype=torch.long)

                input_lengths = [queue['input_lengths'] for queue in batch_queue]
                sort_idx = np.argsort(-np.array(input_lengths))
                num_list = []
                input_lengths = []
                for j in sort_idx:
                    target_variables[j][0:len(batch_queue[j]['target_variables'])] = (torch.tensor(batch_queue[j]['target_variables']))
                    input_variables[j][0:len(batch_queue[j]['input_variables'])] = (torch.tensor(batch_queue[j]['input_variables']))
                    buffer_seq[j][0:len(batch_queue[j]['buffer'])] = (torch.tensor(batch_queue[j]['buffer']))
                    num_list.append(batch_queue[j]['num_list'])
                    input_lengths.append(batch_queue[j]['input_lengths'])

                model.train(True)
                loss, com_list = self._train_mapo_batch(input_variables = input_variables, 
                                                   input_lengths = input_lengths, 
                                                   target_variables = target_variables, 
                                                   target_lengths = target_lengths, 
                                                   model = model, 
                                                   template_flag = template_flag,
                                                   teacher_forcing_ratio = teacher_forcing_ratio,
                                                   mode = mode, 
                                                   batch_size = mapo_batch_size,
                                                   post_flag = post_flag,
                                                   num_list = num_list,
                                                   solutions = solutions,
                                                   buffer_seq = buffer_seq)

                right_count += com_list[0]
                total_r += batch_size

                match += com_list[1]
                total_m += com_list[2]

                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    print ('step: %d, Progress: %d%%, Train : %.4f, Teacher_r: %.2f' % (
                           step,
                           step*1.0 / total_steps * 100,
                           print_loss_avg,
                           teacher_forcing_ratio))

                if step % 40 == 0:
                    model.eval()
                    train_temp_acc, train_ans_acc =\
                                                self.evaluator.evaluate(model = model,
                                                                        data_loader = data_loader,
                                                                        data_list = train_list,
                                                                        template_flag = True,
                                                                        batch_size = batch_size,
                                                                        evaluate_type = 0,
                                                                        use_rule = self.use_rule,
                                                                        mode = mode,
                                                                        post_flag=post_flag,
                                                                        use_rule_old=False)
            #valid_temp_acc, valid_ans_acc =\
            #                            self.evaluator.evaluate(model = model,
            #                                                    data_loader = data_loader,
            #                                                    data_list = valid_list,
            #                                                    template_flag = True,
            #                                                    batch_size = batch_size,
            #                                                    evaluate_type = 0,
            #                                                    use_rule = self.use_rule,
            #                                                    mode = mode,
            #                                                    post_flag=post_flag,
            #                                                    use-rule_old=False)
                    test_temp_acc, test_ans_acc =\
                                                self.evaluator.evaluate(model = model,
                                                                        data_loader = data_loader,
                                                                        data_list = test_list,
                                                                        template_flag = True,
                                                                        batch_size = batch_size,
                                                                        evaluate_type = 0,
                                                                        use_rule = self.use_rule,
                                                                        mode = mode,
                                                                        post_flag=post_flag,
                                                                        use_rule_old=False,
                                                                        name_save="test")
                    self.train_acc_list.append((epoch, step, train_ans_acc))
                    self.test_acc_list.append((epoch, step, test_ans_acc))
                    self.loss_list.append((epoch, epoch_loss_total/steps_per_epoch))

                    if test_ans_acc > max_ans_acc:
                        max_ans_acc = test_ans_acc
                        th_checkpoint = Checkpoint(model=model,
                                                    optimizer=self.optimizer,
                                                    epoch=epoch,
                                                    step=step,
                                                    train_acc_list = self.train_acc_list,
                                                    test_acc_list = self.test_acc_list,
                                                    loss_list = self.loss_list).\
                                                        save_according_name("./experiment", 'best')
                        print(f"Checkpoint saved! max acc: {max_ans_acc}")

                    #print ("Epoch: %d, Step: %d, train_acc: %.2f, %.2f, validate_acc: %.2f, %.2f, test_acc: %.2f, %.2f"\
                    #      % (epoch, step, train_temp_acc, train_ans_acc, valid_temp_acc, valid_ans_acc, test_temp_acc, test_ans_acc))
                    print ("Epoch: %d, Step: %d, train_acc: %.2f, %.2f, test_acc: %.2f, %.2f"\
                        % (epoch, step, train_temp_acc, train_ans_acc, test_temp_acc, test_ans_acc))


    def train(self, model, data_loader, batch_size, n_epoch, template_flag, \
                        resume=False, optimizer=None, mode=0, teacher_forcing_ratio=0, post_flag=False):
        self.evaluator = Evaluator(vocab_dict = self.vocab_dict,
                                   vocab_list = self.vocab_list,
                                   decode_classes_dict = self.decode_classes_dict,
                                   decode_classes_list = self.decode_classes_list,
                                   loss = NLLLoss(),
                                   cuda_use = self.cuda_use)
        if resume:
            checkpoint_path = Checkpoint.get_certain_checkpoint("./experiment", "best")
            resume_checkpoint = Checkpoint.load(checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            start_step = resume_checkpoint.step
            self.train_acc_list = resume_checkpoint.train_acc_list
            self.test_acc_list = resume_checkpoint.test_acc_list
            self.loss_list = resume_checkpoint.loss_list
        else:
            start_epoch = 1
            start_step = 0
            self.train_acc_list = []
            self.test_acc_list = []
            self.loss_list = []

            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=0)
            self.optimizer = optimizer

        self._train_epoches(data_loader = data_loader, 
                            model = model, 
                            batch_size = batch_size,
                            start_epoch = start_epoch, 
                            start_step = start_step, 
                            n_epoch = n_epoch,
                            mode = mode,
                            template_flag = template_flag,
                            teacher_forcing_ratio=teacher_forcing_ratio,
                            post_flag = post_flag)


