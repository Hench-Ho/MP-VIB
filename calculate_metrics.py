import os
import numpy as np

from calculate_modules import *


def calculate_minDCF_EER_CLLR(cm_scores_file,
                              cm_key_file,
                            output_file,
                            printout=True):
    # Evaluation metrics for Phase 1
    # Primary metrics: min DCF,
    # Secondary metrics: EER, CLLR

    key_list=[]
    score_list=[]
    
    Pspoof = 0.05
    dcf_cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Cmiss': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa' : 10, # Cost of CM system falsely accepting nontarget speaker
    }

    # 读取 key
    with open(cm_key_file, 'r') as f:
        key_lines = f.readlines()
        for line in key_lines:
            parts = line.strip().split(' ')[5]
            key_list.append(parts)#spoof/real
            
    # 读取分数
    with open(cm_scores_file, 'r') as f:
        score_lines = f.readlines()
        for line in score_lines:
            parts = line.strip().split(' ')[-1]
            score_list.append(float(parts))
    

    # Load CM scores
    key_arr   = np.array(key_list[:100000])
    score_arr = np.array(score_list[:100000])

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = score_arr[key_arr == 'bonafide']
    spoof_cm = score_arr[key_arr == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_cm, frr, far, thresholds = compute_eer(bona_cm, spoof_cm)#[0]
    cllr_cm = calculate_CLLR(bona_cm, spoof_cm)
    minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])
    # actual DCF
    actDCF, _ = compute_actDCF(bona_cm, spoof_cm, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\n\tmin DCF\t\t= {} '
                        '(min DCF for countermeasure)\n'.format(
                            minDCF_cm))
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(EER for countermeasure)\n'.format(
                            eer_cm * 100))
            f_res.write('\tCLLR\t\t= {:8.9f} bits '
                        '(CLLR for countermeasure)\n'.format(
                            cllr_cm))
            f_res.write('\tactDCF\t\t= {:} '
                        '(actual DCF)\n\n'.format(
                            actDCF))
        os.system(f"cat {output_file}")

    return minDCF_cm, eer_cm*100, cllr_cm, actDCF

