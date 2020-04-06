import sys

from utils import *
import numpy as np
import queue as Q
import math

DEBUG = False

sym2priority = {'+': 0, '-': 0, '*': 1, '/': 1, "^": 1}
#sym2priority.update({str(x):2 if x.isdigit()})

DIFF_THRESHOLD = 1e-5
NAN_THRESHOLD = 10e7
thres_nan = lambda x: x if (NAN_THRESHOLD > abs(x) > DIFF_THRESHOLD and not np.iscomplex(x)) else float('nan')
plus = lambda x,y: thres_nan(x + y)
minus = lambda x,y: thres_nan(x - y)
times = lambda x,y: thres_nan(x * y)
divide = lambda x,y: thres_nan(x / y if y != 0 else float('nan'))
exp = lambda x,y: thres_nan(x ** y if abs(x) < 1000 and abs(y) < 10 and x != 1 and y != 1 else float('nan'))
root = lambda x,y: thres_nan(exp(x, divide(1, y)) if x >= 0 else float('nan'))
log = lambda x,base: thres_nan(math.log(x, base) if base > 0 and base != 1 and x > 0 else float('nan'))
symbol2semantic= {'+': plus, '-': minus, '*': times, '/': divide, '^': exp}
#symbol2semantic.update({x: eval(x) if x.isdigit()})
inverse_op_left = {'+': minus, '-': plus, '*': divide, '/': times, '^': root}
inverse_op_right = {
    '+': minus,
    '-': lambda target, left: minus(left, target),
    '*': divide,
    '/': lambda target, left: divide(left, target),
    '^': log}

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


class LeafNode:
    def __init__(self, symbol_id, all_prob, class_list_expr, num_list_single):
        self.symbol_id = symbol_id
        self.all_prob = all_prob - np.log(np.sum(np.exp(all_prob)))
        self.class_list_expr = class_list_expr
        self.num_list_single = num_list_single
        self.initialize()

    @property
    def symbol(self):
        return self.class_list_expr[self.symbol_id]

    @property
    def priority(self):
        return sym2priority[self.symbol] if self.symbol in sym2priority else 2

    @property
    def prob(self):
        return self.all_prob[self.symbol_id]

    @property
    def max_prob(self):
        return self.all_prob.max()

    @property
    def _res(self):
        if self.symbol in symbol2semantic:
            return symbol2semantic[self.symbol]
        else:
            return inverse_temp_to_num(self.symbol, self.num_list_single)

    def initialize(self):
        self.parent = None

    def res(self):
        return [self._res, self.prob, self.max_prob]

    def entropy(self):
        return -1 * np.sum(np.exp(self.all_prob) * self.all_prob)

    def sample(self):
        all_prob = np.exp(self.all_prob)

        # zero out impossible vars
        for (idx, k) in enumerate(self.class_list_expr):
            if 'temp' in k:
                if (ord(k[5]) - ord('a') >= len(self.num_list_single)):
                    all_prob[idx] = 0

        if 'temp' in self.symbol or 'PI' == self.symbol or self.symbol.isdigit():
            for (idx, k) in enumerate(self.class_list_expr):
                if not ('temp' in k or 'PI' == k or k.isdigit()):
                    all_prob[idx] = 0
        else:
            for (idx, k) in enumerate(self.class_list_expr):
                if 'temp' in k or 'PI' == k or k.isdigit():
                    all_prob[idx] = 0

        if np.all(all_prob==0):
            print(f"sampling all 0 {all_prob}")
            sys.exit(1)

        # zero out self, if there's some other valid symbol
        if all_prob.sum() - all_prob[self.symbol_id] > 1e-5:
            all_prob[self.symbol_id] = 0

        all_prob /= all_prob.sum()
        new_symbol = np.random.choice(range(len(self.class_list_expr)), p=all_prob)
        self.prev_symbol_id = self.symbol_id
        self.symbol_id = new_symbol
        self.initialize()
        return self.symbol_id

    def resume(self):
        self.symbol_id = self.prev_symbol_id
        self.initialize()


class Node:
    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op
        self.parent = None
        self._res = None  # (res, prob, max_prob)
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
            res = float('inf')
        self._res = [res, prob, max_prob]
        self.prob = prob
        self.max_prob = max_prob
        return self._res


from dataclasses import dataclass, field
from typing import Any, List


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


class ExprTree:
    def __init__(self, num_list_single, class_list_expr):
        self.class_list_expr = class_list_expr
        self.num_list_single = num_list_single
        self.tokens: List[LeafNode] = None
        self.root = None

    # Shunting-yard algorithm. See Wikipedia for detailed explanations.
    # https://en.wikipedia.org/wiki/Shunting-yard_algorithm
    # https://www.geeksforgeeks.org/expression-evaluation/
    def parse(self, tokens=None):
        self.parse_postfix(tokens)
    #     if tokens is not None:
    #         tokens = [LeafNode(*tok, self.class_list_expr, self.num_list_single) for tok in tokens]
    #         self.tokens = tokens
    #     else:
    #         tokens = self.tokens
    #     values = []
    #     operators = []
    #     for token in tokens:
    #         if 'temp' in token.symbol or 'PI' == token.symbol or token.symbol.isdigit():
    #             values.append(token)
    #         elif '(' == token.symbol:
    #             operators.append(token)
    #         elif ')' == token.symbol:
    #             while operators[-1].symbol != '(':
    #                 op = operators.pop()
    #                 right = values.pop()
    #                 left = values.pop()
    #                 new_node = Node(left, right, op)
    #                 op.parent = new_node
    #                 right.parent = new_node
    #                 left.parent = new_node
    #                 values.append(new_node)
    #             operators.pop() # discard left parenthesis
    #         else:
    #             while len(operators) > 0 and operators[-1].priority >= token.priority:
    #                 op = operators.pop()
    #                 right = values.pop()
    #                 left = values.pop()
    #                 new_node = Node(left, right, op)
    #                 op.parent = new_node
    #                 right.parent = new_node
    #                 left.parent = new_node
    #                 values.append(new_node)
    #             operators.append(token)
    #
    #     while len(operators) > 0:
    #         op = operators.pop()
    #         right = values.pop()
    #         left = values.pop()
    #         new_node = Node(left, right, op)
    #         op.parent = new_node
    #         right.parent = new_node
    #         left.parent = new_node
    #         values.append(new_node)
    #
    #     self.root = values.pop()
    #     self.root.res()
    #     return self.root

    def parse_postfix(self, tokens=None):
        if tokens is not None:
            tokens = [LeafNode(*tok, self.class_list_expr, self.num_list_single) for tok in tokens]
            self.tokens = tokens
        else:
            tokens = self.tokens
        values = []
        for token in tokens:
            if 'temp' in token.symbol or 'PI' == token.symbol or token.symbol.isdigit():
                values.append(token)
            else:
                op = token
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
            temp_diff = np.array([abs(target - num) for num in self.num_list_single])
            if np.any(temp_diff < DIFF_THRESHOLD):
                target_idx = np.argmax(temp_diff < DIFF_THRESHOLD)
                target_id = self.class_list_expr.index('temp_' + (chr(ord('a') + target_idx)))
                change = PrioritizedItem(node.prob - node.all_prob[target_id], (node, target, target_id))
            elif abs(target - 3.14) < DIFF_THRESHOLD:
                target_id = self.class_list_expr.index('PI')
                change = PrioritizedItem(node.prob - node.all_prob[target_id], (node, target, target_id))
            else:
                change = None
        else:
            change = PrioritizedItem(node.prob - node.max_prob, (node, target))
        return change

    def fix_1step(self, gt):
        old_ids = [tok.symbol_id for tok in self.tokens]

        queue = Q.PriorityQueue()
        change = None
        if isinstance(self.root, LeafNode):
            change = self.find_valid_change(self.root, gt)
        if change is None:
            change = PrioritizedItem(0., (self.root, gt))
        queue.put(change)
        while not queue.empty():
            change = queue.get()
            prob = change.priority
            node, target, *rest = change.item
            if isinstance(node, LeafNode):
                # print('find a fix, early stop.')
                token_idx = self.tokens.index(node)

                if len(change.item) >= 3: # if target_id exists
                    target_id = change.item[2]
                    new_ids = old_ids.copy()
                    new_ids[token_idx] = target_id
                    return (new_ids, self.root.res()[1] - prob)
                else:
                    return None

            left = node.left
            right = node.right
            op = node.op

            # change left
            sub_target = inverse_op_left[op.symbol](target, right.res()[0])
            change = self.find_valid_change(left, sub_target)
            if change is not None:
                if DEBUG and len(change.item) >= 3:
                    changed_token_ids = old_ids.copy()
                    changed_idx = self.tokens.index(left)
                    changed_token_ids[changed_idx] = change.item[2]
                    print(f"    try change: {self.token_id_list_to_str(changed_token_ids)}")

                queue.put(change)

            # change right
            sub_target = inverse_op_right[op.symbol](target, left.res()[0])
            change = self.find_valid_change(right, sub_target)
            if change is not None:
                if DEBUG and len(change.item) >= 3:
                    changed_token_ids = old_ids.copy()
                    changed_idx = self.tokens.index(right)
                    changed_token_ids[changed_idx] = change.item[2]
                    print(f"    try change: {self.token_id_list_to_str(changed_token_ids)}")

                queue.put(change)

            # change op
            ori_op = op.symbol
            ori_op_id = op.symbol_id
            token_idx = self.tokens.index(op)
            for new_op in sym2priority.keys():
                if new_op == ori_op:
                    continue
                op.symbol_id = self.class_list_expr.index(new_op)
                try:
                    node._res = None
                    new_res = node.res()[0]
                    if abs(new_res - target) < DIFF_THRESHOLD:
                        target_id = self.class_list_expr.index(new_op)
                        change = PrioritizedItem(op.prob - op.all_prob[target_id], (op, target, target_id))

                        if DEBUG and len(change.item) >= 3:
                            changed_token_ids = old_ids.copy()
                            changed_token_ids[token_idx] = change.item[2]
                            print(f"    try change op: {self.token_id_list_to_str(changed_token_ids)}")

                        queue.put(change)
                except:
                    pass
            op.symbol_id = ori_op_id #restore
            node._res = None
            node.res()

        return None

    def fix(self, gt, n_step=1):
        if DEBUG:
            print(f"fixing: goal is {gt}, num_list: {self.num_list_single}")
        entropy_list = np.array([x.entropy() for x in self.tokens])
        entropy_list = entropy_list / entropy_list.sum()
        # print([x.symbol for x in self.tokens])

        for i in range(n_step):
            if i > 0:
                self.parse()

            if DEBUG:
                print(f"  fix step {i}: start with {self.token_list_to_str()[0]}")

            fix = self.fix_1step(gt)
            if fix is not None:
                return fix, i + 1
            else:
                accept = False
                while not accept:
                    n_sym_change = int(np.abs(np.random.normal(0, 1, 1)))
                    n_sym_change = np.maximum(n_sym_change, 1)
                    n_sym_change = np.minimum(n_sym_change, len(self.tokens))

                    prob_old_string = np.sum([x.prob for x in self.tokens])
                    token_ids = np.random.choice(len(self.tokens), n_sym_change, replace=False)

                    for tok_id in token_ids:
                        self.tokens[tok_id].sample()
                    prob_new_string = np.sum([x.prob for x in self.tokens])
                    accept_ratio = np.exp(prob_new_string - prob_old_string)
                    if np.random.random() < accept_ratio:
                        accept = True
                    else:
                        for tok_id in token_ids:
                            self.tokens[tok_id].resume()

                # print([x.symbol for x in self.tokens])
        return None

    def token_list_to_str(self, tokens=None):
        if tokens is None:
            tokens = self.tokens
        return self.token_id_list_to_str([tok.symbol_id for tok in tokens])

    def token_id_list_to_str(self, token_ids):
        cur_temp = [self.class_list_expr[tok_id] for tok_id in token_ids]
        cur_str = [str(x) for x in [inverse_temp_to_num(temp, self.num_list_single) for temp in cur_temp]]
        return (cur_temp, cur_str)

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


