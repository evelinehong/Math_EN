import sys
import random
from collections import defaultdict

import numpy as np
from model import EncoderRNN, DecoderRNN_1, Seq2seq
from model.DecoderRNN_3 import NOISE_SIZE, max_len_single, min_len_single
from utils import NLLLoss, Optimizer, Checkpoint, Evaluator

from .diagnosis_multistep import ExprTree, DIFF_THRESHOLD

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import pdb

WANDB = False
if WANDB:
    import wandb

DEBUG = False

def inverse_temp_to_num(elem, num_list_single):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if 'temp' in elem:
        index = alphabet.index(elem[-1])
        return num_list_single[index]
    elif 'PI' == elem:
        return 3.14
    elif elem.isdigit():
        return float(elem)
    else:
        return elem


class BackTrainer(object):
    def __init__(self, vocab_dict, vocab_list, decode_classes_dict, decode_classes_list, cuda_use, \
                 loss, print_every, teacher_schedule, checkpoint_dir_name, fix_rng, use_rule, n_step):
        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list
        self.decode_classes_dict = decode_classes_dict
        self.decode_classes_list = decode_classes_list
        self.class_dict = decode_classes_dict
        self.class_list = decode_classes_list

        self.cuda_use = cuda_use
        self.loss = loss
        if self.cuda_use == True:
            self.loss.cuda()

        self.fix_rng = fix_rng
        self.use_rule = use_rule
        self.n_step = n_step
        self.fix_buffer = defaultdict(list)

        self.print_every = print_every

        self.teacher_schedule = teacher_schedule

        Checkpoint.CHECKPOINT_DIR_NAME = checkpoint_dir_name

    def find_fix(self, preds, gts, all_probs, num_list, ids, n_step):
        """
        preds: batch_size * expr len                 int - predicted ids
        res: batch_size                              float - labeled correct result
        probs: batch_size * expr len * classes       float - predicted all probabilities
        num_list: batch_size * list
        """
        class_list_expr = self.class_list[2:]

        best_fix_list = []
        for pred, gt, all_prob, item_id, num_list_single in zip(preds, gts, all_probs, ids, num_list):
            end_idx = self.class_dict['END_token']
            fix = []
            if end_idx in pred.tolist() and pred.tolist().index(end_idx) > 0:
                l = pred.tolist().index(end_idx)
                pred = pred[:l]
                all_prob = all_prob[:, 2:]

                fix_source_str = ""
                fix_step = -1


                expr_tree_pred = [x - 2 for x in pred]  # convert index
                # prob = all_prob[range(l), pred]
                # pred_str = [id2sym(x) for x in pred]

                tokens = list(zip(expr_tree_pred, all_prob))
                etree = ExprTree(num_list_single, class_list_expr)
                etree.parse_postfix(tokens)

                if abs(etree.res()[0] - gt) <= DIFF_THRESHOLD:
                    fix = list(pred)
                    fix_source_str = "correct"
                else:
                    output = etree.fix(gt, n_step=n_step)
                    if output:
                        (output, fix_step) = output
                        fix = [int(x + 2) for x in output[0]]
                        fix_source_str = "fix found"

                if DEBUG:
                    old_temp = [self.class_list[id] for id in pred]
                    old_str = [str(x) for x in [inverse_temp_to_num(temp, num_list_single) for temp in old_temp]]
                    if len(fix) > 0:
                        new_ids = fix
                        new_temp = [self.class_list[id] for id in new_ids]
                        new_str = [str(x) for x in [inverse_temp_to_num(temp, num_list_single) for temp in new_temp]]

                        print(f"  {fix_source_str}, {num_list_single}, step {fix_step}: {' '.join(old_str)} => {' '.join(new_str)} = {gt}")
                    # else:
                    #     print(f"  no fix found, {num_list_single}, step {fix_step}: {' '.join(old_str)} != {gt}")

            best_fix_list.append(fix)
        return [[int(x) for x in fix_list] for fix_list in best_fix_list]

    def _convert_f_e_2_d_sybmbol(self, target_variable):
        new_variable = []
        batch, colums = target_variable.size()
        for i in range(batch):
            tmp = []
            for j in range(colums):
                # idx = self.decode_classes_dict[self.vocab_list[target_variable[i][j].data[0]]]
                idx = self.decode_classes_dict[self.vocab_list[target_variable[i][j].item()]]
                tmp.append(idx)
            new_variable.append(tmp)
        return Variable(torch.LongTensor(np.array(new_variable)))

    def _train_batch(self, input_variables, input_lengths, target_variables, target_lengths, model: Seq2seq, \
                     teacher_forcing_ratio, mode, post_flag, num_list, solutions, ids, mask_const):

        # EXPLORE
        with torch.no_grad():
            # decoder_outputs: expr_len (list) * batch_size * classes
            # symbols_list: expr_len (list) * batch_size * 1
            decoder_outputs, decoder_hidden, symbols_list = \
                model(input_variable=input_variables,
                      input_lengths=input_lengths,
                      target_variable=target_variables,
                      teacher_forcing_ratio=teacher_forcing_ratio,
                      mode=mode,
                      use_rule=self.use_rule,
                      use_cuda=self.cuda_use,
                      vocab_dict=self.vocab_dict,
                      vocab_list=self.vocab_list,
                      class_dict=self.class_dict,
                      class_list=self.class_list,
                      num_list=num_list,
                      fix_rng=self.fix_rng,
                      use_rule_old=False,
                      target_lengths=target_lengths,
                      mask_const=mask_const,
                      noise=True)
            # cuda
            target_variables_converted = self._convert_f_e_2_d_sybmbol(target_variables)
            if self.cuda_use:
                target_variables_converted = target_variables_converted.cuda()

            pad_in_classes_idx = self.decode_classes_dict['PAD_token']
            batch_size = len(input_lengths)

            match = 0
            total = 0

            seq = symbols_list
            seq_var = torch.cat(seq, 1)

            probs = torch.stack(decoder_outputs, dim=1)  # batch_size * expr_len * classes
            preds = torch.stack(symbols_list, dim=1).squeeze(2)  # batch_size * expr_len
            #preds_print = [[self.class_list[j] for j in preds[i]] for i in range(batch_size)]
            res = solutions

            fix_list = self.find_fix(
                preds.data.cpu().numpy(),
                res,
                probs.data.cpu().numpy(),
                num_list,
                ids,
                self.n_step)

            learn_queue = []
            for i, new_fix in enumerate(fix_list):
                total_fix_buffer = self.fix_buffer[ids[i]]
                if len(new_fix) > 0:
                    if new_fix not in total_fix_buffer:
                        total_fix_buffer.append(new_fix)

        for i in range(batch_size):
            total_fix_buffer = self.fix_buffer[ids[i]]
            if len(total_fix_buffer) > 0:
                print(f"{num_list[i]} => {solutions[i]}")
                #min_len = min([len(x) for x in total_fix_buffer])
                for fix in total_fix_buffer:
                    #if len(fix) > 2 + min_len: # too long
                    #    continue

                    learn_queue.append((i, fix))

                    new_ids = fix
                    new_temp = [self.class_list[id] for id in new_ids]
                    new_str = [str(x) for x in [inverse_temp_to_num(temp, num_list[i]) for temp in new_temp]]

                    print(f"  {' '.join(new_str)}")

        # UPDATE
        random.shuffle(learn_queue)
        learn_queue_idx, learn_queue_fixes = map(list, zip(*learn_queue)) #unzip

        mapo_batch_size = 64
        mapo_last_batch_size = len(learn_queue_idx) % mapo_batch_size
        mapo_batch_sizes = [mapo_batch_size]*(len(learn_queue_idx)//mapo_batch_size) \
                           + ([] if mapo_last_batch_size==0 else [mapo_last_batch_size])
        pos = 0
        total_avg_loss = 0
        for mapo_batch_size in mapo_batch_sizes:
            learn_queue_idx_batch = learn_queue_idx[pos:(pos + mapo_batch_size)]
            learn_queue_fixes_batch = learn_queue_fixes[pos:(pos + mapo_batch_size)]

            mapo_decoder_outputs, mapo_decoder_hidden, mapo_symbols_list = \
                model(input_variable=input_variables[learn_queue_idx_batch],
                      input_lengths=input_lengths[learn_queue_idx_batch],
                      target_variable=target_variables[learn_queue_idx_batch],
                      teacher_forcing_ratio=teacher_forcing_ratio,
                      mode=mode,
                      use_rule=False,
                      use_cuda=self.cuda_use,
                      vocab_dict=self.vocab_dict,
                      vocab_list=self.vocab_list,
                      class_dict=self.class_dict,
                      class_list=self.class_list,
                      num_list=num_list[learn_queue_idx_batch],
                      fix_rng=self.fix_rng,
                      use_rule_old=False,
                      target_lengths=None,
                      mask_const=mask_const,
                      noise=False)

            self.loss.reset()
            for step, step_output in enumerate(mapo_decoder_outputs):
                fixed_step = torch.full((mapo_batch_size,), -1, dtype=torch.long)  # -1 ignored in NLLLoss
                for i in range(mapo_batch_size):
                    if step < len(learn_queue_fixes_batch[i]):
                        fixed_step[i] = torch.tensor(learn_queue_fixes_batch[i][step])
                    elif step == len(learn_queue_fixes_batch[i]):
                        fixed_step[i] = self.class_dict['END_token']
                    else:
                        fixed_step[i] = self.class_dict['PAD_token']

                if self.cuda_use:
                    step_output = step_output.cuda()
                    fixed_step = fixed_step.cuda()

                # loss wrt fixed output
                self.loss.eval_batch(step_output.contiguous().view(mapo_batch_size, -1), fixed_step)

            pos += mapo_batch_size

            model.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            total_avg_loss += self.loss.get_loss()

        return total_avg_loss/len(mapo_batch_sizes), [0, match, total]

    def _train_epoches(self, data_loader, model, batch_size, start_epoch, start_step, n_epoch, \
                       mode, teacher_forcing_ratio, post_flag):
        print_loss_total = 0

        train_list = data_loader.math23k_train_list
        test_list = data_loader.math23k_test_list
        valid_list = data_loader.math23k_valid_list
        train_list = train_list + valid_list  # = data_loader.math23k_valid_list
        steps_per_epoch = len(train_list) / batch_size
        total_steps = steps_per_epoch * n_epoch

        step = start_step
        step_elapsed = 0

        threshold = [0] + [1] * 9

        for epoch in range(start_epoch, n_epoch + 1):
            epoch_loss_total = 0

            # marker if self.teacher_schedule:

            batch_generator = data_loader.get_batch(train_list, batch_size, True)

            right_count = 0
            match = 0
            total_m = 0
            total_r = 0

            model.train(True)
            for batch_idx, batch_data_dict in enumerate(batch_generator):
                step += 1
                step_elapsed += 1
                ids = batch_data_dict['batch_index']
                input_variables = batch_data_dict['batch_encode_pad_idx']
                input_lengths = batch_data_dict['batch_encode_len']
                target_variables = batch_data_dict['batch_decode_pad_idx']
                target_lengths = batch_data_dict['batch_decode_len']
                num_list = batch_data_dict['batch_num_list']
                solutions = batch_data_dict['batch_solution']

                # cuda
                input_variables = Variable(torch.LongTensor(input_variables))
                target_variables = Variable(torch.LongTensor(target_variables))

                if self.cuda_use:
                    input_variables = input_variables.cuda()
                    target_variables = target_variables.cuda()

                mask_const = False
                if epoch == 1:
                    mask_const = True

                loss, com_list = self._train_batch(input_variables=input_variables,
                                                   input_lengths=input_lengths,
                                                   target_variables=target_variables,
                                                   target_lengths=target_lengths,
                                                   model=model,
                                                   teacher_forcing_ratio=teacher_forcing_ratio,
                                                   mode=mode,
                                                   post_flag=post_flag,
                                                   num_list=num_list,
                                                   solutions=solutions,
                                                   ids=ids,
                                                   mask_const=mask_const)

                right_count += com_list[0]
                total_r += batch_size

                match += com_list[1]
                total_m += com_list[2]

                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed >= self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    print('step: %d, Progress: %d%%, Train %s: %.4f, Teacher_r: %.2f' % (
                        step,
                        step * 1.0 / total_steps * 100,
                        self.loss.name,
                        print_loss_avg,
                        teacher_forcing_ratio))
                    if WANDB:
                        wandb.log({"epoch": epoch, "avg loss": print_loss_avg}, step=step)

            model.eval()
            with torch.no_grad():
                train_temp_acc, train_ans_acc = \
                    self.evaluator.evaluate(model=model,
                                            data_loader=data_loader,
                                            data_list=train_list,
                                            batch_size=batch_size,
                                            evaluate_type=0,
                                            use_rule=self.use_rule,
                                            buffer=self.fix_buffer,
                                            mode=mode,
                                            post_flag=post_flag,
                                            use_rule_old=False,
                                            fix_rng=self.fix_rng)
                # valid_temp_acc, valid_ans_acc =\
                #                            self.evaluator.evaluate(model = model,
                #                                                    data_loader = data_loader,
                #                                                    data_list = valid_list,
                #                                                    batch_size = batch_size,
                #                                                    evaluate_type = 0,
                #                                                    use_rule = self.use_rule,
                #                                                    mode = mode,
                #                                                    post_flag=post_flag,
                #                                                    use-rule_old=False,
                #                                                    fix_rng=self.fix_rng)
                test_temp_acc, test_ans_acc = \
                    self.evaluator.evaluate(model=model,
                                            data_loader=data_loader,
                                            data_list=test_list,
                                            batch_size=batch_size,
                                            evaluate_type=0,
                                            use_rule=self.use_rule,
                                            mode=mode,
                                            buffer=self.fix_buffer,
                                            post_flag=post_flag,
                                            use_rule_old=False,
                                            name_save="test",
                                            fix_rng=self.fix_rng)

                if len(self.test_acc_list) > 0:
                    max_test_acc = max([test_acc for _, _, test_acc in self.test_acc_list])
                else:
                    max_test_acc = 0

                self.train_acc_list.append((epoch, step, train_ans_acc))
                self.test_acc_list.append((epoch, step, test_ans_acc))
                self.loss_list.append((epoch, epoch_loss_total / steps_per_epoch))

                checkpoint = Checkpoint(model=model,
                                        optimizer=self.optimizer,
                                        epoch=epoch,
                                        step=step,
                                        train_acc_list=self.train_acc_list,
                                        test_acc_list=self.test_acc_list,
                                        loss_list=self.loss_list,
                                        buffer=self.fix_buffer)
                checkpoint.save_according_name("./experiment", "latest")

                if test_ans_acc > max_test_acc:
                    max_test_acc = test_ans_acc
                    checkpoint.save_according_name("./experiment", 'best')
                    print(f"Checkpoint best saved! max acc: {max_test_acc}")
                    if WANDB:
                        wandb.save(f"./experiment/{checkpoint.CHECKPOINT_DIR_NAME}/best/model.pt")
                        wandb.save(f"./experiment/{checkpoint.CHECKPOINT_DIR_NAME}/best/trainer_states.pt")
                if WANDB:
                    wandb.save(f"./experiment/{checkpoint.CHECKPOINT_DIR_NAME}/latest/pg_seq_norm_True_train.json")
                    wandb.save(f"./experiment/{checkpoint.CHECKPOINT_DIR_NAME}/latest/pg_seq_norm_True_test.json")

            # print ("Epoch: %d, Step: %d, train_acc: %.2f, %.2f, validate_acc: %.2f, %.2f, test_acc: %.2f, %.2f"\
            #      % (epoch, step, train_temp_acc, train_ans_acc, valid_temp_acc, valid_ans_acc, test_temp_acc, test_ans_acc))
            print("Epoch: %d, Step: %d, train_acc: %.2f, %.2f, test_acc: %.2f, %.2f, max_test_acc: %.2f" \
                  % (epoch, step, train_temp_acc, train_ans_acc, test_temp_acc, test_ans_acc, max_test_acc))

            if WANDB:
                wandb.log({"epoch": epoch,
                           "train temp accuracy": train_temp_acc,
                           "train ans accuracy": train_ans_acc,
                           "test temp accuracy": test_temp_acc,
                           "test ans accuracy": test_ans_acc}, step=step)

    def train(self, model, data_loader, batch_size, n_epoch, \
              resume=False, optimizer=None, mode=0, teacher_forcing_ratio=0, post_flag=False):
        self.evaluator = Evaluator(vocab_dict=self.vocab_dict,
                                   vocab_list=self.vocab_list,
                                   decode_classes_dict=self.decode_classes_dict,
                                   decode_classes_list=self.decode_classes_list,
                                   loss=NLLLoss(),
                                   cuda_use=self.cuda_use)
        if resume:
            checkpoint_path = Checkpoint.get_certain_checkpoint("./experiment", "latest")
            resume_checkpoint = Checkpoint.load(checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch + 1
            start_step = resume_checkpoint.step
            self.train_acc_list = resume_checkpoint.train_acc_list
            self.test_acc_list = resume_checkpoint.test_acc_list
            self.loss_list = resume_checkpoint.loss_list
            self.fix_buffer = resume_checkpoint.buffer
        else:
            start_epoch = 1
            start_step = 0
            self.train_acc_list = []
            self.test_acc_list = []
            self.loss_list = []

            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=0)
            self.optimizer = optimizer

        self._train_epoches(data_loader=data_loader,
                            model=model,
                            batch_size=batch_size,
                            start_epoch=start_epoch,
                            start_step=start_step,
                            n_epoch=n_epoch,
                            mode=mode,
                            teacher_forcing_ratio=teacher_forcing_ratio,
                            post_flag=post_flag)


