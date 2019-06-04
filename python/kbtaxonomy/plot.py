import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

def plot1():
    TAX_RESULT_FILE = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yago_output/agri_taxonomy_500.json'
    ORG_RESULT_FILE = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yago_output/final_sps_500.json'
    tps = json.load(open(TAX_RESULT_FILE))
    ops = json.load(open(ORG_RESULT_FILE))
    xs = [i for i in range(len(tps))]
    taxonomy_probs = list(tps.values())#[tps[x] for x in xs]
    org_probs = list(ops.values())#[ops[x] for x in xs]
    print('%d %d %d' % (len(xs), len(taxonomy_probs), len(org_probs)))
    taxonomy_probs.sort()
    org_probs.sort()
    plt.plot(xs, org_probs, color='crimson', label='organization', markevery=50, markersize=4)
    plt.plot(xs, taxonomy_probs, color='darkblue', label='yago taxonomy', marker='x', markevery=50, markersize=4)

    plt.legend(fontsize=18,loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Attribute', fontsize=16)
    plt.ylabel('Discovery Probability', fontsize=16)
    plt.ylim([0.0,1.0])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('taxonomy_org_'+str(len(org_probs))+'.pdf')
    print('saved to %s' % 'taxonomy_org_'+str(len(org_probs))+'.pdf')
    plt.clf()



def plot2():
    TAX_RESULT_FILE = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yagocloud_output/agri_taxonomy_2364.json'
    TRIM_TAX_RESULT_FILE = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yagocloud_output/agri_taxonomy_trim_2364.json'
    ORG_RESULT_FILE = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yagocloud_output/final_sps_2364.json'
    tps = json.load(open(TAX_RESULT_FILE))
    ops = json.load(open(ORG_RESULT_FILE))
    ttps = json.load(open(TRIM_TAX_RESULT_FILE))
    xs = [i for i in range(len(tps))]
    taxonomy_probs = list(tps.values())
    org_probs = list(ops.values())
    trim_taxonomy_probs = list(ttps.values())
    print('%d %d %d %d' % (len(trim_taxonomy_probs), len(xs), len(taxonomy_probs), len(org_probs)))
    taxonomy_probs.sort()
    org_probs.sort()
    trim_taxonomy_probs.sort()
    plt.plot(xs, org_probs, color='crimson', label='organization', markevery=50, markersize=4)
    plt.plot(xs, trim_taxonomy_probs, color='green', label='organization', markevery=50, markersize=4)
    plt.plot(xs, taxonomy_probs, color='darkblue', label='yago taxonomy', marker='x', markevery=50, markersize=4)

    plt.legend(fontsize=18,loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Attribute', fontsize=16)
    plt.ylabel('Success Probability', fontsize=16)
    plt.ylim([0.0,1.0])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('trim_taxonomy_org_'+str(len(org_probs))+'.pdf')
    print('saved to %s' % 'trim_taxonomy_org_'+str(len(org_probs))+'.pdf')
    plt.clf()



plot2()
