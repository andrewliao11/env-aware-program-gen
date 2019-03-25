import sys
import numpy as np
from tqdm import tqdm


filename = sys.argv[1]
file = np.load(filename)
num_programs = len(file['prog_test'])


file_str = []
for prog_id in tqdm(range(num_programs)):

    pred_prog = file['prog_train'][prog_id].replace(
        '<', ' ').replace('>', ' ').split(',')
    gt_prog = file['prog_test'][prog_id].replace(
        '<', ' ').replace('>', ' ').split(',')
    npred, ngt = len(pred_prog), len(gt_prog)

    curr_score = file['scores'][prog_id]
    table_content = '<h5> LCS {:.3f}. F1-attribute {:.3f}. F1-relation {:.3f} </h5>'.format(
        curr_score['LCS'], curr_score['F1'][-2], curr_score['F1'][-1])
    table_content += '<br><table>'
    table_content += '<tr><th> GT </th><th> Pred </th></tr>'
    
    for it_insrt in range(max(npred, ngt)):
        gt_instr = '' if it_insrt >= ngt else gt_prog[it_insrt]
        pred_instr = '' if it_insrt >= npred else pred_prog[it_insrt]
        table_content += '<tr><td> {} </td><td> {} </td></tr>'.format(
            gt_instr, pred_instr)
    table_content += '</table><br><br>'
    file_str.append(table_content)


with open('viz_NN.html', 'w+') as f:
    f.writelines(file_str)
