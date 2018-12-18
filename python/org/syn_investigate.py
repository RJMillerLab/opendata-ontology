import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-dark-palette')
#print(plt.style.available)


def plot():
    # loading files
    tableprobs_flat = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/flat_dists_2651_notor.json', 'r'))
    tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists2651fuzzy_g10t752opint.json'))
    tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists_2651_single_g10t752opint.json', 'r'))
    tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/multidim_dists_2651_2_g10rhap.json', 'r'))
    tableprobs_multidim_prune = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/multidim_dists_prune_rep_sim_threshold062_2651_2_g10rhap.json', 'r'))
    tableprobs_multidim_boost = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/multidim_dists_boosted_2651_2_g10rhap.json', 'r'))


    before = []
    after = []
    flat = []
    #flat100 = []
    multidim = []
    multidim_prune = []
    multidim_boost = []
    for t, p in tableprobs_before.items():
        before.append(p)
        after.append(tableprobs_after[t])
        flat.append(tableprobs_flat[t])
        #flat100.append(tableprobs_flat100[t])
        multidim.append(tableprobs_multidim[t])
        multidim_prune.append(tableprobs_multidim_prune[t])
        multidim_boost.append(tableprobs_multidim_boost[t])
    print('loaded')
    inx = np.argsort(np.array(after))
    after = list(np.array(after)[inx])
    print('s1')
    inx = np.argsort(np.array(before))
    before = list(np.array(before)[inx])
    print('s2')
    inx = np.argsort(np.array(flat))
    flat = list(np.array(flat)[inx])
    print('s3')
    inx = np.argsort(np.array(multidim))
    multidim = list(np.array(multidim)[inx])
    print('s4')
    inx = np.argsort(np.array(multidim_prune))
    multidim_prune = list(np.array(multidim_prune)[inx])
    print('s5')
    inx = np.argsort(np.array(multidim_boost))
    multidim_boost = list(np.array(multidim_boost)[inx])


    xs = [i for i in range(len(before))]
    #plt.plot(xs, before, color='r', label='initial org (avg:'+'{:.3f}'.format(sum(before)/len(before))+')', linestyle='--', marker='x', markevery=40, markersize=4)
    plt.plot(xs, before, color='r', label='initial org', linestyle='--', marker='x', markevery=40,markersize=4)
    #plt.plot(xs, after, color='b', label='fixed org (avg:'+'{:.3f}'.format(sum(after)/len(after))+')', marker='x', markevery=40, markersize=4)
    plt.plot(xs, after, color='b', label='fixed org', marker='x', markevery=40, markersize=4)
    #plt.plot(xs, flat, color='g', label='baseline (avg:'+'{:.3f}'.format(sum(flat)/len(flat))+')')
    plt.plot(xs, flat, color='g', label='baseline')
    #plt.plot(xs, multidim, color='black', label='2-dimensional (avg:'+'{:.3f}'.format(sum(multidim)/len(multidim))+')', marker='+', markevery=40, markersize=4)
    plt.plot(xs, multidim, color='black', label='2-dimensional', marker='+', markevery=40, markersize=4)
    #plt.plot(xs, multidim_prune, color='orange', label='2-dimensional prune (avg:'+'{:.3f}'.format(sum(multidim_prune)/len(multidim_prune))+')', marker='^', markevery=40, markersize=4)
    plt.plot(xs, multidim_prune, color='orange', label='2-dimensional prune', marker='^',         markevery=40, markersize=4)
    #plt.plot(xs, multidim_boost, color='purple', label='2-dimensional boosted\ntags by 2 (avg:'+'{:.3f}'.format(sum(multidim_boost)/len(multidim_boost))+')', marker='o', markevery=40, markersize=4)
    plt.plot(xs, multidim_boost, color='purple', label='2-dimensional boosted\ntags by 2', marker='o', markevery=40, markersize=4)
    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Tables', fontsize=16)
    plt.ylabel('Discovery Probability', fontsize=16)
    #plt.title('Table Discovery in Benchmark')
    plt.savefig('bigbench_multidim_boosted_orgs.pdf')

    print('%f -> %f -> %f -> %f -> %f -> %f' % (sum(flat)/len(flat), sum(before)/len(before), sum(after)/len(after), sum(multidim)/len(multidim), sum(multidim_prune)/len(multidim_prune), sum(multidim_boost)/len(multidim_boost)))

    plt.clf()


def plot_domain_stats():
    syn_multidim_stats = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/fix_1773_prunning_stats.json', 'r'))

    stats = syn_multidim_stats
    domprune = []
    repprune = []
    for st in stats:
        domprune.append(st['active_domains'])
        repprune.append(st['active_reps'])
    xs = [i for i in range(len(domprune))]
    num_domains = max(domprune)
    noprune = [num_domains for i in xs]


    xs=[0.6,1.8,3.0]
    labels=['No Prunning','Effected Atts', 'Effected Reps']
    ys=[np.mean(np.array(noprune)), np.mean(np.array(domprune)), np.mean(np.array(repprune))]
    plt.bar(xs,ys, width=0.35, yerr = [float(np.std(np.array(noprune))), float(np.std(np.array(domprune))), float(np.std(np.array(repprune)))], align='center', color=('teal', 'mediumblue', 'red'), capsize=7)
    plt.xticks(xs, labels, fontsize=16, ha='center')#, rotation=10)
    plt.ylim([0,max(noprune)+50])
    plt.xlim([0,3.6])

    plt.grid(linestyle='dotted')
    plt.ylabel('#Evaluated Attributes', fontsize=18)
    #plt.title('Prunning Attributes in Synthetic Benchmark')
    plt.savefig('synthetic_prunning_dom_rep_bar.pdf')
    plt.clf()




def plot_domain_stats2():
    syn_multidim_stats = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/fix_1773_prunning_stats.json', 'r'))

    stats = syn_multidim_stats
    domprune = []
    repprune = []
    for st in stats:
        domprune.append(st['active_domains'])
        repprune.append(st['active_reps'])
    xs = [i for i in range(len(domprune))]
    #domprune.sort()
    #repprune.sort()
    num_domains = max(domprune)
    noprune = [num_domains for i in xs]
    #plt.plot(xs, domprune, color='mediumblue', label='Prunning Domains', marker='x', markevery=80, markersize=4)
    #plt.plot(xs, repprune, color='crimson', label='Prunning by Reps', marker='+', markevery=80, markersize=4)
    #plt.plot(xs, noprune, color='teal', linestyle='--', label='No Prunning')
    width = 1.5
    ns = 3*np.array(xs)
    plt.bar(ns+width, np.array(noprune), width, label='No Prunning', color='teal')
    plt.bar(ns+2*width, np.array(domprune), width, label='Prunning Domains', color='mediumblue')
    plt.bar(ns+3*width, np.array(repprune), width, label='Prunning by Reps', color='crimson')


    plt.legend(loc='best', fancybox=True)
    #plt.grid(linestyle='dotted')
    plt.xlabel('Exploration Iteration')
    plt.ylabel('Number of Considered Data Points')
    plt.title('Prunning Domains and States in Synthetic Benchmark')
    plt.xlim([0,len(xs)])
    plt.ylim([0,max(max(domprune), max(repprune))])
    plt.savefig('synthetic_prunning_dom_rep.pdf')
    plt.clf()

def plot_state_stats():
    syn_multidim_stats = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/fix_1773_prunning_stats.json', 'r'))

    stats = syn_multidim_stats
    stateprune = []
    for st in stats:
        stateprune.append(st['active_states'])
    num_states = max(stateprune)
    noprune = [num_states for i in stateprune]

    xs=[0.6,1.6]
    labels=['No Prunning','Effected States']
    ys=[np.mean(np.array(noprune)), np.mean(np.array(stateprune))]
    plt.bar(xs,ys, width=0.25, yerr = [float(np.std(np.array(noprune))), float(np.std(np.array(stateprune)))], align='center', color=('teal', 'mediumblue'), capsize=7)
    plt.xticks(xs, labels,  fontsize=16)#, rotation=10)


    plt.ylim([0,max(max(noprune),max(stateprune))+20])
    plt.xlim([0,2.2])
    plt.ylabel('#Visited States', fontsize=18)
    #plt.title('Prunning Unchanged States in Synthetic Benchmark')
    plt.grid(linestyle='dotted')
    plt.savefig('synthetic_prunning_state_bar.pdf')
    plt.clf()



def plot_state_stats2():
    syn_multidim_stats = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/fix_1773_prunning_stats.json', 'r'))

    stats = syn_multidim_stats
    stateprune = []
    for st in stats:
        stateprune.append(st['active_states'])
    xs = [i for i in range(len(stateprune))]
    #stateprune.sort()
    num_states = max(stateprune)
    noprune = [num_states for i in xs]

    width = 1.5
    ns = 3*np.array(xs)
    plt.bar(ns+2*width, np.array(noprune), width, label='No Prunning', color='teal')
    plt.bar(ns+width, np.array(stateprune), width, label='Prunning States', color='mediumblue')

    #plt.plot(xs, stateprune, color='mediumblue', label='Prunning States', marker='x', markevery=80, markersize=4)
    #plt.plot(xs, noprune, color='teal', linestyle='--', label='No Prunning')
    plt.legend(loc='best', fancybox=True)
    #plt.grid(linestyle='dotted')
    plt.xlim([0,len(stateprune)])
    plt.ylim([0,max(max(noprune),max(stateprune))])
    plt.xlabel('Exploration Iteration')
    plt.ylabel('Number of Visited States')
    plt.title('Prunning States in Synthetic Benchmark')
    plt.savefig('synthetic_prunning_state.pdf')
    plt.clf()




def plot_synthetic_label_dom_sims():
    tagdomsims = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/rep_domain_sims.json', 'r'))
    ys = []
    for t, dsims in tagdomsims.items():
        for d, s in dsims.items():
            ys.append(s)
    ys.sort()
    xs = [i for i in range(len(ys))]
    plt.plot(xs, ys, color='blue')
    plt.xlabel('label-domain pairs')
    plt.ylabel('similarity')
    plt.savefig('syntehtic_labeldomain_sims.pdf')
    plt.clf()



plot_state_stats()
#plot_domain_stats()
#plot()
