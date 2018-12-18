

def test(tagdomains):
    dmax = 0
    dmin = 10000000
    for t, ds in tagdomains.items():
        dmax = len(ds) if len(ds) > dmax else dmax
        dmin = len(ds) if len(ds) < dmin else dmin
    print('dmin: %d' % dmin)
    print('dmax: %d' % dmax)


def stratified_sample(tagdomains, sample_perc):
    # sample from tables of each tag and add all of their domains.
    tagtables = dict()
    tabledomains = dict()
    for tag, doms in tagdomains.items():
        tagtables[tag] = []
        for domain in doms:
            dom = domain['name']
            table = dom[:dom.rfind('_')]
            if table not in tagtables[tag]:
                tagtables[tag].append(table)
            if table not in tabledomains:
                tabledomains[table] = []
            tabledomains[table].append(domain)


    stagdoms = dict()
    sdoms = []

    for tag, tables in tagtables.items():
        stagdoms[tag] = []
        table_sample_inx = int(len(tables)*sample_perc)+1
        for tbl in tables[:table_sample_inx]:
            stagdoms[tag].extend(tabledomains[tbl])
            sdoms.extend(tabledomains[tbl])


    #tag_sample_inx = 0
    #for t, doms in tagdomains.items():
    #    tag_sample_inx = int(len(doms)*sample_perc)+1
    #    sdoms.extend(tagdomains[t][:tag_sample_inx])
    #    stagdoms[t] = list(tagdomains[t][:tag_sample_inx])
    return stagdoms, sdoms




