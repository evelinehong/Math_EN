from utils import *
import numpy as np
import queue as Q
import math
sym2priority = {'+': 0, '-': 0, '*': 1, '/': 1, "^": 1}
#sym2priority.update({str(x):2 if x.isdigit()})

plus = lambda x,y: x + y
minus = lambda x,y: x - y
times = lambda x,y: x * y
divide = lambda x,y: x / (y + 1e-7)
exp = lambda x,y: x ** y
root = lambda x,y: x ** (1/(y+1e-7))
symbol2semantic= {'+': plus, '-': minus, '*': times, '/': divide, '^': exp}
#symbol2semantic.update({x: eval(x) if x.isdigit()})
inverse_op = {'+': minus, '-': plus, '*': divide, '/': times, '^': root}


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
        self.symbol = self.class_list_expr[self.symbol_id]
        self.num_list_single = num_list_single
        self.initialize()

    def initialize(self):
        self.priority = sym2priority[self.symbol] if self.symbol in sym2priority else 2
        self.prob = self.all_prob[self.symbol_id]
        self.max_prob = self.all_prob.max()
        self.parent = None
        if self.symbol in symbol2semantic:
            self._res = symbol2semantic[self.symbol]
        else:
            self._res = inverse_temp_to_num(self.symbol, self.num_list_single)

    def res(self):
        return [self._res, self.prob, self.max_prob]

    def entropy(self):
        return -1 * np.sum(np.exp(self.all_prob) * self.all_prob)

    def sample(self):
        # self.all_prob[self.symbol_id] = np.log(1e-30)
        # self.all_prob = self.all_prob - np.log(np.sum(np.exp(self.all_prob)))
        all_prob = np.exp(self.all_prob)
        all_prob /= all_prob.sum()
        new_symbol = np.random.choice(range(len(self.class_list_expr)), p=all_prob)
        self.prev_symbol = self.symbol_id
        self.symbol_id = new_symbol
        self.initialize()
        return self.symbol_id

    def resume(self):
        self.symbol_id = self.prev_symbol
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
from typing import Any


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


class ExprTree:
    def __init__(self, num_list_single, class_list_expr):
        self.class_list_expr = class_list_expr
        self.num_list_single = num_list_single
        self.tokens = None
        self.root = None

    # Shunting-yard algorithm. See Wikipedia for detailed explanations.
    # https://en.wikipedia.org/wiki/Shunting-yard_algorithm
    # https://www.geeksforgeeks.org/expression-evaluation/
    def parse(self, tokens=None):
        if tokens is not None:
            tokens = [LeafNode(*tok, self.class_list_expr, self.num_list_single) for tok in tokens]
            self.tokens = tokens
        else:
            tokens = self.tokens
        values = []
        operators = []
        for token in tokens:
            if 'temp' in token.symbol or 'PI' == token.symbol or token.symbol.isdigit():
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
            target = round(target, 3)
            temp_diff = np.array([abs(target - num) for num in self.num_list_single])
            if np.any(temp_diff < 1e-5):
                target_idx = np.where(temp_diff < 1e-5)[0]
                target_id = self.class_list_expr.index('temp_' + (chr(ord('a') + target_idx)))
                change = PrioritizedItem(node.prob - node.all_prob[target_id], (node, target))
            elif abs(target - 3.14) < 1e-5:
                target_id = self.class_list_expr.index('PI')
                change = PrioritizedItem(node.prob - node.all_prob[target_id], (node, target))
            else:
                change = None
        else:
            change = PrioritizedItem(node.prob - node.max_prob, (node, target))
        return change

    def fix_1step(self, gt):
        old_ids = [tok.symbol_id for tok in self.tokens]

        queue = Q.PriorityQueue()
        change = PrioritizedItem(0., (self.root, gt))
        queue.put(change)
        find_fix = False
        while not queue.empty():
            change = queue.get()
            prob = change.priority
            node, target = change.item
            if isinstance(node, LeafNode):
                # print('find a fix, early stop.')
                if str(target).isnumeric():
                    temp_diff = np.array([abs(target - num) for num in self.num_list_single])
                    if np.any(temp_diff < 1e-5) or abs(target - 3.14) < 1e-5:
                        find_fix = True
                        break
                else:
                    find_fix = True
                    break

            left = node.left
            right = node.right
            op = node.op

            # change left
            sub_target = inverse_op[op.symbol](target, right.res()[0])
            change = self.find_valid_change(left, sub_target)
            if change != None:
                queue.put(change)

            # change right
            if op.symbol in ['+', '*']:
                sub_target = inverse_op[op.symbol](target, left.res()[0])
            else:
                sub_target = op.res()[0](left.res()[0], target)
            change = self.find_valid_change(right, sub_target)
            if change != None:
                queue.put(change)

            # change op
            ori_op = op.symbol
            token_id = self.tokens.index(op)
            sub_target = None
            for new_op in sym2priority.keys():
                if new_op == ori_op:
                    continue
                new_str = [str(tok.res()[0]) for tok in self.tokens]
                new_str[token_id] = "**" if new_op=="^" else new_op
                try:
                    new_res = eval(''.join(new_str))
                    if abs(new_res - gt) < 1e-5:
                        sub_target = new_op
                        change = PrioritizedItem(op.prob - op.all_prob[self.class_list_expr.index(sub_target)], (op, sub_target))
                        queue.put(change)
                except:
                    pass

        if find_fix:
            token_id = self.tokens.index(node)

            new_ids = old_ids.copy()

            if not isinstance(target, str): #numeric
                target = round(target, 3)
                temp_diff = np.array([abs(target - num) for num in self.num_list_single])
                target_sym = new_ids[token_id]
                if np.any(temp_diff < 1e-5):
                    target_idx = np.where(temp_diff < 1e-5)[0]
                    target_sym = self.class_list_expr.index('temp_' + (chr(ord('a') + target_idx)))
                elif abs(target - 3.14) < 1e-5:
                    target_sym = self.class_list_expr.index('PI')

                new_ids[token_id] = target_sym
            else:
                new_ids[token_id] = self.class_list_expr.index(target)

            if new_ids != old_ids:
                return (new_ids, self.root.res()[1])
        return None

    def fix(self, gt, n_step=1):
        entropy_list = np.array([x.entropy() for x in self.tokens])
        entropy_list = entropy_list / entropy_list.sum()
        # print([x.symbol for x in self.tokens])

        for i in range(n_step):
            if i > 0:
                self.parse()
            fix = self.fix_1step(gt)
            if fix is not None:
                return fix
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


