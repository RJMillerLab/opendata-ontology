cat /home/fnargesian/FINDOPENDATA_DATASETS/10k/datasets-31k.list | xargs -d '\n' -P 20 -n 1 python json2emb.py
#cat /home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/small.list | xargs -d '\n' -P 20 -n 1 python json2emb.py
