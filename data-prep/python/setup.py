import os

LIST = os.environ['OPENDATA_LIST']
OUTPUT = os.environ['OUTPUT_DIR']

with open(LIST) as f:
    for line in f:
        dirpath = os.path.join(OUTPUT, "domains", line.strip())
        try:
            os.makedirs(dirpath)
            print("[x] %s" % dirpath)
        except FileExistsError:
            print("[o] %s" % dirpath)

os.makedirs(os.path.join(OUTPUT, "logs"))
