cat /home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/tables_1000.list | xargs -d '\n' -P 5 -n 1 python -u parallel_navigation.py
