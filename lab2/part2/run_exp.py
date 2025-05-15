import os
import sys
import run_deit
import run_sim


# run subprocess

nbits_list = [4, 2]
group_size_list = [64]
# (4, 64, 15.050880432128906, 1524.1336669921875, np.float64(552.3643442372212), 5)
# (2, 64, 215830.671875, 1292.1336669921875, np.float64(622.39901892391), 5)

results = []
for nbits in nbits_list:
    for group_size in group_size_list:
        # acc_after_quant, model_size, score = run_deit.main(nbits, group_size)
        # results.append((nbits, group_size, acc_after_quant, model_size, score))
        ppl, model_size, quant_tput, score = run_sim.main(nbits, group_size)
        results.append((nbits, group_size, ppl, model_size, quant_tput, score))
        
for result in results:
    print(result)

