

def test(tagdomains):
    dmax = 0
    dmin = 10000000
    for t, ds in tagdomains.items():
        dmax = len(ds) if len(ds) > dmax else dmax
        dmin = len(ds) if len(ds) < dmin else dmin
    print('dmin: %d' % dmin)
    print('dmax: %d' % dmax)


def stratified_sample(tagdomains, sample_perc):
    if int(sample_perc*len(tagdomains)) < len(tagdomains):
        print('invalid sample num < num of tags')
    stagdoms = dict()
    sdoms = []

    tag_sample_inx = 0
    for t, doms in tagdomains.items():
        tag_sample_inx = int(len(doms)*sample_perc)+1
        sdoms.extend(tagdomains[t][:tag_sample_inx])
        stagdoms[t] = list(tagdomains[t][:tag_sample_inx])
    print('sdoms: %d' % len(sdoms))
    print('stagdoms: %d' % sum([len(doms) for t, doms in stagdoms.items()]))
    return stagdoms, sdoms




