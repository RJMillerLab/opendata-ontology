from gensim.models import FastText
import os
import numpy as np
import random
import base64
import sys
import sqlite3
import time

def get_model(bin_file):
    return FastText.load_fasttext_format(bin_file)

def model2sqlite3(sqlite_db, model):
    db = sqlite3.connect(sqlite_db)
    cursor = db.cursor()

    words = model.wv.vocab.keys()
    n = len(words)
    print("Total words: %d" % n)
    start_time = time.time()
    def report(i, n):
        frac = (i+1) / n
        duration = time.time() - start_time
        eta = duration / frac
        if i+1 < n:
            print("... %d (%.2f %%) rows inserted. Estimated total in %.2f seconds" % (
                (i+1), frac * 100, eta))
        else:
            print("=== completed: %d rows inserted in %.2f seconds ===" % (n, duration))

    cursor.execute("drop table if exists wv;")
    cursor.execute("drop index if exists wv_idx")
    cursor.execute("""
        create table wv (
            word varchar(100),
            vec blob,
            idx int,
            count int
            )
    """)

    for i, w in enumerate(words):
        if i > 0 and i % (n//10) == 0: report(i, n)
        vec = model.wv.get_vector(w).tobytes()
        vocab = model.wv.vocab.get(w)
        cursor.execute("insert into wv values (?,?,?,?)",
                (w, vec, vocab.index, vocab.count))
        if i > 0 and i % 10000 == 0: db.commit()
    db.commit()
    report(i, n)

    print("Creating index...")
    start_time = time.time()
    cursor.execute("create index wv_idx on wv(word)")
    print("Done in %.2f seconds" % (time.time() - start_time))

    db.close()


def vectorize_words(words, model):
    return np.array([model[w] for w in words])

def load_words(filename):
    with open(filename, "r") as f:
        words = [line.split() for line in f]
    return words

def sample_words(dict_file, model, sample_size):
    words = []
    with open(dict_file, 'r') as f:
        vocab = [w.strip() for w in f]
    vocab = random.sample(vocab, sample_size * 10)
    n = 0
    for i, w in enumerate(vocab):
        if w in model:
            words.append(w)
            n += 1
            if n >= sample_size:
                break
    return words

def make_column(word, tags, model, size):
    # neg = [w for w in tags if not w == word]
    neg = []
    answers = model.wv.most_similar(positive=[word],
            negative=neg,
            topn=size)
    return [x[0].replace(",", "") for x in answers]

def make_table(tags, model, size, name=None):
    """ Generates len(tags) columns x size using similar
    words from model.
    """
    table = []
    for w in tags:
        table.append(make_column(w, tags, model, size))

    if name:
        # save the tags
        with open("%s.tags.vec" % name, "w") as f:
            for w in tags:
                vec = model[w]
                s = base64.b64encode(vec.tostring())
                print(w, s, file=f)
        # save the table
        with open("%s.csv" % name, "w") as f:
            for i in range(size):
                row = [col[i] for col in table]
                print(",".join(row), file=f)
    return table

def print_args(args):
    divider = "%s|%s" % ("-"*21, "-"*40)
    print("{:>20} | {}".format("num-tags", args.num_tags))
    print("{:>20} | {}".format("num-tables", args.num_tables))
    print(divider)
    print("{:>20} | {}".format("min-rows", args.min_rows))
    print("{:>20} | {}".format("max-rows", args.max_rows))
    print(divider)
    print("{:>20} | {}".format("min-attrs", args.min_attrs))
    print("{:>20} | {}".format("max-attrs", args.max_attrs))
    print(divider)
    print("{:>20} | {}".format("out-dir", args.out_dir))
    print("{:>20} | {}".format("dict-file", args.dict_file))
    print("{:>20} | {}".format("bin_file", args.bin_file))

def main():
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=1000)
    #parser.add_argument("--size-distribution", help="zipf, unif")
    parser.add_argument("--min-attrs", type=int, default=3)
    parser.add_argument("--max-attrs", type=int, default=20)
    parser.add_argument("--num-tags", type=int, default=50)
    parser.add_argument("--num-tables", type=int, default=50)
    parser.add_argument("--out-dir")
    parser.add_argument("--dict-file")
    parser.add_argument("--bin-file")
    #
    parser.add_argument("--num-table-per-tag", type=int, default=500)
    args = parser.parse_args()

    # check for parameters
    if not args.out_dir or not args.dict_file or not args.bin_file:
        parser.print_help()
        sys.exit()

    print_args(args)

    start_time = time.time()
    print("Loading model from %s" % args.bin_file)
    model = get_model(args.bin_file)
    print("Model is loaded in %.2f seconds." % (time.time() - start_time))

    words = sample_words(args.dict_file, model, args.num_tags)
    os.makedirs(args.out_dir)

    tags = random.sample(words, args.num_tags)
    k = 0
    for j in range(len(tags)):
        t = tags[j]
        for i in range(args.num_table_per_tag):
            num_attrs = 1
            num_rows = random.randint(args.min_rows, args.max_rows)
            table_name = "table_%d" % k
            k += 1
            table_file = os.path.join(args.out_dir, table_name)

            print("> %s (%d X %d)" % (table_name, num_attrs, num_rows))

            make_table([t], model, size=num_rows, name=table_file)

if __name__ == '__main__':
    main()
