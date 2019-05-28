import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

def plot():
    tps = json.load(open(TAX_RESULT_FILE))
    ops = json.load(open(ORG_RESULT_FILE))
    xs = [i for i in range(len(tps))]
    taxonomy_probs = list(tps.values())#[tps[x] for x in xs]
    org_probs = list(ops.values())#[ops[x] for x in xs]
    print('%d %d %d' % (len(xs), len(taxonomy_probs), len(org_probs)))
    taxonomy_probs.sort()
    org_probs.sort()
    plt.plot(xs, org_probs, color='crimson', label='organization', markevery=100, markersize=4)
    plt.plot(xs, taxonomy_probs, color='darkblue', label='yago taxonomy', marker='x', markevery=100, markersize=4)

    plt.legend(fontsize=18,loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Attribute', fontsize=21)
    plt.ylabel('Discovery Probability', fontsize=21)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('taxonomy_org.pdf')
    plt.clf()


TAX_RESULT_FILE = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yago_output/agri_taxonomy_100.json'
ORG_RESULT_FILE = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yago_output/final_sps.json'

plot()
