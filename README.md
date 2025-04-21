Solving Texas Holdem using CFR algo:

Directly play with Slumbot using eval_slumbot files, where suffices mean num of buckets and additional action, e.g. eval_slumbot_15b_raise.py means using 15 buckets and allow raise action. The corresponding database used in the file can be seen in the loading process in each program.

Calculating solutions for the game using files with cfr prefix, e.g. cfr_turn_15b_raise.py means using 15 buckets and allow raise action, same as eval files, and if not mentioned specifically in the name, we use 10 buckets. Note that I have tried some solvers that combine streets like cfr_turn_river.py and cfr_preflop_flop.py.

prepare database like buckets using bucket files. E.g. bucket6.py means calculating bucketing for turn street where 6 is for 2 hole cards + 4 community cards. Also you can get db database like db7 for river street by db7s.py. Detail explanation is put in corresponding .py file.

all needed data are stored in 'backup' folder, where another README will explain the content of each file.