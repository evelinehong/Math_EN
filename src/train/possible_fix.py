from utils import *
import itertools

max_op = 3
possible_fix = []
for n_op in range(max_op+1):
    res2fix = {}
    combinations = [digit_list] + [op_list, digit_list] * n_op
    combinations = itertools.product(*combinations)
    for eq in combinations:
        res = eval(''.join(eq))
        res = round(res, res_precision)
        if res not in res2fix:
            res2fix[res] = []
        res2fix[res].append(list(map(sym2id, eq)))
    possible_fix.append(res2fix)

def find_fix(preds, res_list, seq_lens, probs):
    best_fix_list = []
    for pred, res, l, prob in zip(preds, res_list, seq_lens, probs):
        pred = pred[:l]
        prob = prob[:l]
        
        fix_list = possible_fix[l//2][round(float(res), res_precision)]
        if list(pred) in fix_list:
            best_fix = list(pred)
        else:
            fix_with_p_list = [(prob[range(l), x].sum(), x) for x in fix_list]
            best_fix = max(fix_with_p_list)[1]
        
        best_fix_list.append(best_fix)
    return best_fix_list