
from .utils_train import *

import queue as Q
import math
sym2priority = {'+': 0, '-': 0, '*': 1, '/': 1, "^": 1}
#sym2priority.update({str(x):2 if x.isdigit()})

plus = lambda x,y: x + y
minus = lambda x,y: x - y
times = lambda x,y: x * y
divide = lambda x,y: x / (y + 1e-7)
exp = lambda x,y: x ** y
symbol2semantic= {'+': plus, '-': minus, '*': times, '/': divide, '^': exp}
#symbol2semantic.update({x: eval(x) if x.isdigit()})
inverse_op = {plus: minus, minus: plus, times: divide, divide: times}

class LeafNode:
    def __init__(self, token, num_list, new_list, new_dict):
        self.symbol = token[0]
        self.all_prob = token[1] - np.log(np.sum(np.exp(token[1])))
        self.num_list = num_list
        self.new_list = new_list
        self.new_dict = new_dict
        self.initialize()



    def initialize(self):

        temp = self.symbol
        if self.symbol[0].isdigit() and self.symbol != '3.14':
            i = 0
            for num in self.num_list:
                if eval(self.symbol) == num:
                    temp = 'temp_' + chr(ord('a')+i)
                i += 1
        if self.symbol == '3.14':
            temp = 'PI'
        self.symbol_id = self.new_dict[temp]

        if not self.symbol[0].isdigit():
            self.priority = sym2priority[self.symbol]
        else:
            self.priority = 2
        self.prob = self.all_prob[self.symbol_id]
        self.max_prob = self.all_prob.max()
        self.parent = None

        if not self.symbol[0].isdigit():
            self._res= symbol2semantic[self.symbol]
        else:
            self._res = eval(self.symbol)


    def res(self):
        return [self._res, self.prob, self.max_prob]


    def entropy(self):
        return -1 * np.sum(np.exp(self.all_prob) * self.all_prob)
    
    def sample(self):
        # self.all_prob[self.symbol_id] = np.log(1e-30)
        # self.all_prob = self.all_prob - np.log(np.sum(np.exp(self.all_prob)))
        all_prob = np.exp(self.all_prob)
        all_prob /= all_prob.sum()
        new_symbol = np.random.choice(sym_list, p = all_prob)
        self.prev_symbol = self.symbol
        self.symbol = new_symbol
        self.initialize()
        return self.symbol
    
    def resume(self):
        self.symbol = self.prev_symbol
        self.initialize()

class Node:
    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op
        self.parent = None
        self._res = None # (res, prob, max_prob)
        self.prob = None
        self.max_prob = None

    def res(self):
        if self._res != None:
            return self._res

        left_res = self.left.res()
        right_res = self.right.res()
        op_res = self.op.res()
        prob = left_res[1] + right_res[1] + op_res[1]
        max_prob = left_res[2] + right_res[2] + op_res[2]
        try:
            res = op_res[0](left_res[0], right_res[0])
        except:
            res = -10000
        self._res = [res, prob, max_prob]
        self.prob = prob
        self.max_prob = max_prob
        return self._res

from dataclasses import dataclass, field
from typing import Any
@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)
        
class ExprTree:
    def __init__(self, num_list, new_dict):
        self.num_list = num_list
        self.new_dict = new_dict
        self.tokens = None
        self.root = None

    # Shunting-yard algorithm. See Wikipedia for detailed explanations.
    # https://en.wikipedia.org/wiki/Shunting-yard_algorithm 
    # https://www.geeksforgeeks.org/expression-evaluation/ 

    def parse(self, tokens=None, num_list = None, new_list=None, new_dict=None):
        if tokens is not None:
            tokens = [LeafNode(tok, num_list, new_list, new_dict) for tok in tokens]
            self.tokens = tokens
        else:
            tokens = self.tokens
        values = []
        operators = []

        for token in tokens:
            if token.symbol[0].isdigit():
                values.append(token)
            else:
                while len(operators) > 0 and operators[-1].priority >= token.priority:
                    op = operators.pop()
                    right = values.pop()
                    left = values.pop()
                    new_node = Node(left, right, op)
                    op.parent = new_node
                    right.parent = new_node
                    left.parent = new_node
                    values.append(new_node)
                operators.append(token)

        while len(operators) > 0:
            op = operators.pop()
            right = values.pop()
            left = values.pop()
            new_node = Node(left, right, op)
            op.parent = new_node
            right.parent = new_node
            left.parent = new_node
            values.append(new_node)
        
        self.root = values.pop()
        self.root.res()
        return self.root
    
    def res(self):
        return self.root.res()

    def find_valid_change(self, node, target):
        if isinstance(node, LeafNode):
            find = False

            i = 0
            temp = ''
            for num in self.num_list:
                if np.abs(float(target) - num) < 1e-5:
                    target = str(num)
                    temp = 'temp_' + chr(ord('a')+i)
                    find = True
                i += 1

            if find:
                target_id = self.new_dict[temp]
                change = PrioritizedItem(node.prob - node.all_prob[target_id], (node, target))

            else:
                change = None
        else:
            change = PrioritizedItem(node.prob - node.max_prob, (node, target))
        return change


    def fix_1step(self, gt):
        queue = Q.PriorityQueue()
        change = PrioritizedItem(0., (self.root, gt))
        queue.put(change)
        find_fix = False
        j = 0
        while not queue.empty():
            change = queue.get()
            prob = change.priority
            node, target = change.item
            if isinstance(node, LeafNode):
                # print('find a fix, early stop.')
                if j == 0:
                    change = self.find_valid_change(node, target)
                    if change:
                        find_fix = True
                    break
                else:
                    find_fix = True
                    break

            left = node.left
            right = node.right
            op = node.op

            # change left
            if op.symbol == '^':
                sub_target = target ** (1/(right.res()[0]+1e-12))
            else:
                sub_target = inverse_op[op.res()[0]](target, right.res()[0])
            change = self.find_valid_change(left, sub_target)
            if change != None:
                queue.put(change)

            # change right
            if op.symbol in ['+', '*']:
                sub_target = inverse_op[op.res()[0]](target, left.res()[0])
            elif op.symbol == '^':
                sub_target = math.log(target+1e-12, left.res()[0]+1e-12)
            else:
                sub_target = op.res()[0](left.res()[0], target)
            change = self.find_valid_change(right, sub_target)
            if change != None:
                queue.put(change)

            # change op
            ori_op = op.symbol
            token_id = self.tokens.index(op)
            sub_target = None
            for new_op in ["+", "-", "*", "/", "^"]:
                if new_op == ori_op:
                    continue
                new_str = [tok.symbol for tok in self.tokens]
                if new_op == "^":
                    new_str[token_id] = "**"
                else:
                    new_str[token_id] = new_op
                try: 
                    new_res = eval(''.join(new_str))
                    if equal_res(new_res, gt):
                        sub_target = new_op
                        change = PrioritizedItem(op.prob - op.all_prob[self.new_dict[sub_target]], (op, sub_target))
                        queue.put(change)
                except:
                    pass
            j += 1

        if find_fix:
            token_id = self.tokens.index(node)
            new_str = [tok.symbol for tok in self.tokens]
            if not isinstance(target, str):
                target = str(target)
            new_str[token_id] = target
            return (new_str, self.root.res()[1] - prob)
        return None
    
    def fix(self, gt, n_step=1):
        fix = self.fix_1step(gt)
        return fix
        # entropy_list = np.array([x.entropy() for x in self.tokens])
        # entropy_list = entropy_list / entropy_list.sum()
        # print ("\n")
        # print ("entropy")
        # print([x.symbol for x in self.tokens])
        
        # for i in range(n_step):
            # if i > 0:
            #     self.parse()
            # fix = self.fix_1step(gt)
            # if fix is not None:
            #     return fix
            # else:
            #     accept = False
            #     while not accept:
            #         n_sym_change = int(np.abs(np.random.normal(0, 1, 1)))
            #         n_sym_change = np.maximum(n_sym_change, 1)
            #         n_sym_change = np.minimum(n_sym_change, len(self.tokens))


            #         prob_old_string = np.sum([x.prob for x in self.tokens])
            #         token_ids = np.random.choice(len(self.tokens), n_sym_change, replace=False)

            #         for tok_id in token_ids:
            #             self.tokens[tok_id].sample()
            #         prob_new_string = np.sum([x.prob for x in self.tokens])
            #         accept_ratio = np.exp(prob_new_string - prob_old_string)
            #         if np.random.random() < accept_ratio:
            #             accept = True
            #         else:
            #             for tok_id in token_ids:
            #                 self.tokens[tok_id].resume()
            #     print (i)
            #     print([x.symbol for x in self.tokens])

    def fix_bak(self, gt, n_step=1):
        entropy_list = np.array([x.entropy() for x in self.tokens])
        entropy_list = entropy_list / entropy_list.sum()
        print([x.symbol for x in self.tokens])
        for i in range(n_step):
            if i > 0:
                self.parse()
            fix = self.fix_1step(gt)
            if fix is not None:
                return fix
            else:
                token_id = np.random.choice(entropy_list.shape[0], p=entropy_list)
                new_symbol = self.tokens[token_id].sample()
                print([x.symbol for x in self.tokens])
        return None



            




if __name__ == "__main__":
    import numpy as np
    np.random.seed(777)
    expr = "1-3*4"
    all_prob = np.log(np.random.random(size=(len(expr), len(sym_list))))
    max_len = len(expr)
    digit_pos_list = np.arange(0, max_len, 2)
    op_pos_list = np.arange(1, max_len, 2)
    mask = np.zeros_like(all_prob)
    mask[digit_pos_list[:, None], digit_idx_list] = 1.
    if len(op_pos_list) > 0:
        mask[op_pos_list[:, None], op_idx_list] = 1. 
    all_prob = np.log(mask * np.exp(all_prob) + 1e-12)
    print(all_prob)

    tokens = list(zip(expr, all_prob))
    etree = ExprTree()
    etree.parse(tokens)
    print(etree.res())
    print(etree.fix(11, n_step=20))
            

