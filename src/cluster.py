'''This module provides basic cluster processing for seismic events.'''

import collections
import operator
import math
import os
import pickle
import numpy as num
from pyrocko import moment_tensor, orthodrome, model

epsilon = 1e-6
km = 1000.


class DBEvent:
    '''
    Event condition with respect to the DBSCAN clustering.

    :param event_id: event index (sequential number)
    :param event_type: type of the event according to clustering
    :param directly_reachable: list of ids of events directly reachable
    '''

    def __init__(self, event_id, event_type, directly_reachables):
        self.event_id = event_id
        self.event_type = event_type
        self.directly_reachables = directly_reachables


class ClusterError(Exception):
    pass


def save_obj(obj, name):
    '''
    Save an object (clustering results)
    '''
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    '''
    Load an object (clustering results)
    '''
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def dbscan(simmat, nmin, eps, sw_clusterall):
    '''
    Apply DBSCAN algorithm, reading a similarity matrix and returning a list of
    events clusters

    :param simmat: similarity matrix (numpy matrix)
    :param nmin: minimum number of neighbours to define a cluster
    :param eps: minimum distance to search for neighbours
    '''

    nev = len(simmat)
    clusterevents = []
    for i in range(nev):
        event_id = i
        directly_reachables = []
        reachables = []
        for j in range(nev):
            if simmat[i, j] <= eps:
                if (i != j):
                    directly_reachables.append(j)
                    reachables.append(j)

        if len(directly_reachables) >= nmin:
            event_type = "core"
        else:
            event_type = "not-core"

        clusterevents.append(
            DBEvent(event_id, event_type, directly_reachables))

    for i in range(nev):
        for j in range(nev):
            if (i in clusterevents[j].directly_reachables) \
                    and (clusterevents[j].event_type == "core"):

                if clusterevents[i].event_type == "not-core":
                    clusterevents[i].event_type = "edge"

        if clusterevents[i].event_type == "not-core":
            clusterevents[i].event_type = "isle"

    eventtypes = []
    for clusterevent in clusterevents:
        if clusterevent.event_type not in eventtypes:
            eventtypes.append(clusterevent.event_type)
    eventsclusters = num.zeros((nev), dtype=int)
    actualcluster = -1
    for i in range(nev):
        if clusterevents[i].event_type == "isle":
            eventsclusters[i] = -1
        else:
            if clusterevents[i].event_type == "core":
                actualcluster = actualcluster + 1
                reachedevents = []
                pointingevents = []

                eventsclusters[i] = actualcluster
                reachedevents.append(i)
                pointingevents.append(i)

                while len(pointingevents) > 0:
                    newpointingevents = []
                    for j in pointingevents:
                        for k in clusterevents[j].directly_reachables:
                            if clusterevents[k].event_type == "core":
                                reachedevents.append(k)
                                newpointingevents.append(k)
                                eventsclusters[k] = actualcluster
                                clusterevents[k].event_type = "reached"
                            elif clusterevents[k].event_type == "edge":
                                reachedevents.append(k)
                                eventsclusters[k] = actualcluster
                                clusterevents[k].event_type = "reached"

                    pointingevents = []
                    for newpointingevent in newpointingevents:
                        pointingevents.append(newpointingevent)

    n_clusters = actualcluster + 1

# assigning all events to a cluster
    if sw_clusterall:
        for i in range(nev):
            if eventsclusters[i] == -1:
                ds_to_clusters = num.zeros(n_clusters-1)
                cluster_sizes = num.zeros(n_clusters-1)
                for j in range(nev):
                    d = simmat[i, j]
                    ds_to_clusters[eventsclusters[j]] += d
                    cluster_sizes[eventsclusters[j]] += 1.
                for k in range(n_clusters-1):
                    ds_to_clusters[k] = ds_to_clusters[k]/cluster_sizes[k]
                k, = num.where(ds_to_clusters == min(ds_to_clusters))
                eventsclusters[i] == k[0]

#   resorting data clusters (noise remains as -1)
    clustersizes = []
    for icl in range(n_clusters):
        lencl = len([ev for ev in eventsclusters if ev == icl])
        clustersizes.append([icl, lencl])
#    for icl in range(n_clusters):
#        clustersizes.append([icl, 0])
#    for evcl in eventsclusters:
#        clustersizes[evcl][1] += 1

    resorted_clustersizes = sorted(
        clustersizes, key=lambda tup: tup[1], reverse=True)
    resorting_dict = {}
    resorting_dict[-1] = -1

    for icl in range(n_clusters):
        resorting_dict[resorted_clustersizes[icl][0]] = icl

    for ievcl, evcl in enumerate(eventsclusters):
        eventsclusters[ievcl] = resorting_dict[evcl]

    return eventsclusters


def get_clusters(events, eventsclusters):
    '''
    Provided a list of events and a list of cluster labels of equal size
    returns a dictionary of clusters with all their events

    i.e.

         cluster -1 -> [event1, event4, eventx]
         cluster  0 -> [event2, event3, eventn]
    '''

    origclusters = {}
    if len(events) != len(eventsclusters):
        raise ClusterError(
            'number of events different from number of cluster labels: %s, %s'
            % len(events), len(eventsclusters))

    for iev, evcl in enumerate(eventsclusters):
        if evcl not in origclusters:
            origclusters[evcl] = [events[iev]]
        else:
            origclusters[evcl].append(events[iev])

    clusters = collections.OrderedDict(sorted(origclusters.items()))
    return clusters


def get_distance_mt_l2(eventi, eventj):
    '''
    L2 norm among two moment tensors, with 6 independet entries
    '''

    # ni = (eventi.mxx)**2 + (eventi.myy)**2 + (eventi.mzz)**2 + \
    #      (eventi.mxy)**2 + (eventi.mxz)**2 + (eventi.myz)**2
    # nj = (eventj.mxx)**2 + (eventj.myy)**2 + (eventj.mzz)**2 + \
    #      (eventj.mxy)**2 + (eventj.mxz)**2 + (eventj.myz)**2

    # mixx, miyy, mizz = eventi.mxx / ni, eventi.myy / ni, eventi.mzz / ni
    # mixy, mixz, miyz = eventi.mxy / ni, eventi.mxz / ni, eventi.myz / ni
    # mjxx, mjyy, mjzz = eventj.mxx / nj, eventj.myy / nj, eventj.mzz / nj
    # mjxy, mjxz, mjyz = eventj.mxy / nj, eventj.mxz / nj, eventj.myz / nj

    mti = eventi.moment_tensor
    mtj = eventj.moment_tensor

    d = (mti.mnn - mtj.mnn)**2 + (mti.mee - mtj.mee)**2 + \
        (mti.mdd - mtj.mdd)**2 + (mti.mne - mtj.mne)**2 + \
        (mti.mnd - mtj.mnd)**2 + (mti.med - mtj.med)**2

    d = 0.5 * math.sqrt(d)

    return d


def get_distance_mt_l1(eventi, eventj):
    '''
    L1 norm among two moment tensors, with 6 independet entries
    '''

    # ni = abs(eventi.mxx) + abs(eventi.myy) + abs(eventi.mzz) + \
    #      abs(eventi.mxy) + abs(eventi.mxz) + abs(eventi.myz)
    # nj = abs(eventj.mxx) + abs(eventj.myy) + abs(eventj.mzz) + \
    #      abs(eventj.mxy) + abs(eventj.mxz) + abs(eventj.myz)
    #
    # mixx, miyy, mizz = eventi.mxx / ni, eventi.myy / ni, eventi.mzz / ni
    # mixy, mixz, miyz = eventi.mxy / ni, eventi.mxz / ni, eventi.myz / ni
    # mjxx, mjyy, mjzz = eventj.mxx / nj, eventj.myy / nj, eventj.mzz / nj
    # mjxy, mjxz, mjyz = eventj.mxy / nj, eventj.mxz / nj, eventj.myz / nj

    mti = eventi.moment_tensor
    mtj = eventj.moment_tensor

    d = abs(mti.mnn - mtj.mnn) + abs(mti.mee - mtj.mee) + \
        abs(mti.mdd - mtj.mdd) + abs(mti.mne - mtj.mne) + \
        abs(mti.mnd - mtj.mnd) + abs(mti.med - mtj.med)

    d = 0.5 * math.sqrt(d)

    return d


def get_distance_mt_cos(eventi, eventj):
    '''
    Inner product among two moment tensors.

    According to Willemann 1993; and later to Tape & Tape, normalization in
    R^9 to ensure innerproduct between -1 and +1.
    '''

    mti = eventi.moment_tensor
    mtj = eventj.moment_tensor

    ni = math.sqrt(
        mti.mnn * mti.mnn +
        mti.mee * mti.mee +
        mti.mdd * mti.mdd +
        2. * mti.mne * mti.mne +
        2. * mti.mnd * mti.mnd +
        2. * mti.med * mti.med)

    nj = math.sqrt(
        mtj.mnn * mtj.mnn +
        mtj.mee * mtj.mee +
        mtj.mdd * mtj.mdd +
        2. * mtj.mne * mtj.mne +
        2. * mtj.mnd * mtj.mnd +
        2. * mtj.med * mtj.med)

    nc = ni * nj
    innerproduct = (
        mti.mnn * mtj.mnn +
        mti.mee * mtj.mee +
        mti.mdd * mtj.mdd +
        2. * mti.mne * mtj.mne +
        2. * mti.mnd * mtj.mnd +
        2. * mti.med * mtj.med) / nc

    if innerproduct >= 1.0:
        innerproduct = 1.0
    elif innerproduct <= -1.0:
        innerproduct = -1.0

    d = 0.5 * (1 - innerproduct)

    return d


def get_distance_mt_weighted_cos(eventi, eventj, ws):
    '''
    Weighted moment tensor distance.

    According to Cesca et al. 2014 GJI
    '''
    mti = eventi.moment_tensor
    mtj = eventj.moment_tensor

    ni = math.sqrt(
        (ws[0] * mti.mnn)**2 +
        (ws[1] * mti.mne)**2 +
        (ws[2] * mti.mee)**2 +
        (ws[3] * mti.mnd)**2 +
        (ws[4] * mti.med)**2 +
        (ws[5] * mti.mdd)**2)

    nj = math.sqrt(
        (ws[0] * mtj.mnn)**2 +
        (ws[1] * mtj.mne)**2 +
        (ws[2] * mtj.mee)**2 +
        (ws[3] * mtj.mnd)**2 +
        (ws[4] * mtj.med)**2 +
        (ws[5] * mtj.mdd)**2)

    nc = ni * nj
    innerproduct = (
        ws[0] * ws[0] * mti.mnn * mtj.mnn +
        ws[1] * ws[1] * mti.mne * mtj.mne +
        ws[2] * ws[2] * mti.mee * mtj.mee +
        ws[3] * ws[3] * mti.mnd * mtj.mnd +
        ws[4] * ws[4] * mti.med * mtj.med +
        ws[5] * ws[5] * mti.mdd * mtj.mdd) / nc

    if innerproduct >= 1.0:
        innerproduct = 1.0

    elif innerproduct <= -1.0:
        innerproduct = -1.0

    d = 0.5 * (1.0 - innerproduct)

    return d


def get_distance_dc(eventi, eventj):
    '''Normalized Kagan angle distance among DC components of moment tensors.
       Based on Kagan, Y. Y., 1991, GJI
    '''

    mti = eventi.moment_tensor
    mtj = eventj.moment_tensor

    d = moment_tensor.kagan_angle(mti, mtj) / 120.
    if d > 1.:
        d = 1.

    return d


def get_distance_hypo(eventi, eventj):
    '''
    Normalized Euclidean hypocentral distance, assuming flat earth to combine
    epicentral distance and depth difference.

    The normalization assumes largest considered distance is 1000 km.
    '''
    maxdist_km = 1000.
    a_lats, a_lons, b_lats, b_lons = \
        eventi.lat, eventi.lon, eventj.lat, eventj.lon

    a_dep, b_dep = eventi.depth, eventj.depth

    if (a_lats == b_lats) and (a_lons == b_lons) and (a_dep == b_dep):
        d = 0.
    else:
        distance_m = orthodrome.distance_accurate50m_numpy(
            a_lats, a_lons, b_lats, b_lons)

        distance_km = distance_m / 1000.
        ddepth_km = abs(eventi.depth - eventj.depth)/km
        hypo_distance_km = math.sqrt(
            distance_km * distance_km + ddepth_km * ddepth_km)

        # maxdist = float(inv_param['EUCLIDEAN_MAX'])

        d = hypo_distance_km / maxdist_km
        if d >= 1.:
            d = 1.

    return d


def get_distance_epi(eventi, eventj):
    '''Normalized Euclidean epicentral distance.
       The normalization assumes largest considered distance is 1000 km.
    '''
    maxdist_km = 1000.

    a_lats, a_lons, b_lats, b_lons = \
        eventi.lat, eventi.lon, eventj.lat, eventj.lon

    if (a_lats == b_lats) and (a_lons == b_lons):
        d = 0.
    else:
        distance_m = orthodrome.distance_accurate50m_numpy(
            a_lats, a_lons, b_lats, b_lons)

        distance_km = distance_m / 1000.

        d = distance_km / maxdist_km
        if d >= 1.:
            d = 1.

    return d


def get_distance_interevent_time(eventi, eventj):
    '''
    Normalized interevent time.

    The normalization assumes largest considered time difference is 365 days.
    '''

    ti = eventi.time
    tj = eventj.time

    d = abs(ti - tj) / 31536000.
    if d >= 1.:
        d = 1.

    return d


def get_distance_magnitude(eventi, eventj):
    '''
    Normalized magnitude difference.

    The normalization assumes largest considered difference 10.
    '''

    mi = eventi.magnitude
    mj = eventj.magnitude

    d = abs(mi - mj) / 10.
    if d >= 1.:
        d = 1.

    return d


def get_distance_triangle_diagram(eventi, eventj):
    '''
    Scalar product among principal axes (?).
    '''

    mti = eventi.moment_tensor
    mtj = eventj.moment_tensor

    ti, pi, bi = mti.t_axis(), mti.p_axis(), mti.b_axis()
    deltabi = math.acos(abs(bi[2]))
    deltati = math.acos(abs(ti[2]))
    deltapi = math.acos(abs(pi[2]))
    tj, pj, bj = mtj.t_axis(), mtj.p_axis(), mtj.b_axis()
    deltabj = math.acos(abs(bj[2]))
    deltatj = math.acos(abs(tj[2]))
    deltapj = math.acos(abs(pj[2]))
    dotprod = deltabi * deltabj + deltati * deltatj + deltapi * deltapj

    if dotprod >= 1.:
        dotprod == 1.

    d = 1. - dotprod
    return d


def get_distance(eventi, eventj, metric, **kwargs):
    '''
    Compute the normalized distance among two earthquakes, calling the function
    for the chosen metric definition.
    '''

    metric_funcs = {
        'mt_l2norm': get_distance_mt_l2,
        'mt_l1norm': get_distance_mt_l1,
        'mt_cos': get_distance_mt_cos,
        'mt_weighted_cos': get_distance_mt_weighted_cos,
        'kagan_angle': get_distance_dc,
        'hypocentral': get_distance_hypo,
        'epicentral': get_distance_epi,
        'temporal': get_distance_interevent_time,
        'magnitude': get_distance_magnitude,
        'principal_axis': get_distance_triangle_diagram}

    try:
        func = metric_funcs[metric]
    except KeyError:
        raise ClusterError('unknown metric: %s' % metric)

    return func(eventi, eventj)


def compute_similarity_matrix(events, metric):
    '''
    Compute and return a similarity matrix for all event pairs, according to
    the desired metric

    :param events: list of pyrocko events
    :param metric: metric type (string)

    :returns: similarity matrix as NumPy array
    '''

    nev = len(events)
    simmat = num.zeros((nev, nev), dtype=float)
    for i in range(len(events)):
        for j in range(i):
            d = get_distance(events[i], events[j], metric)
            simmat[i, j] = d
            simmat[j, i] = d

    return simmat


def get_simmat_clustered(events, eventsclusters, clusters, conf,
                         resdir, simmat_temp):
    '''
    Compute and return a similarity matrix after clustering
    '''
    nev = len(events)
    clevs = []
    for iev, ev in enumerate(events):
        clevs.append([eventsclusters[iev], iev])
    clevs.sort(key=operator.itemgetter(0))

    simmat2 = num.zeros((nev, nev))
    for i, clevi in enumerate(clevs):
        indi = clevi[1]
        for j, clevj in enumerate(clevs):
            indj = clevj[1]
            simmat2[i, j] = simmat_temp[indi, indj]
    return simmat2


def get_median(events, conf):
    '''
    Return the event, computed according to the used metric
    '''
    if conf.sw_simmat:
        dist_matrix = load_similarity_matrix(conf.sim_mat_fn)
    else:
        nev = len(events)
        dist_matrix = num.zeros((nev, nev))
        for i, evi in enumerate(events):
            for j, evj in enumerate(events):
                dist_matrix[i, j] = get_distance(evi, evj, conf.metric)
    for i, evi in enumerate(events):
        cumul_d = 0.
        for j, evj in enumerate(events):
            d = dist_matrix[i][j]
            cumul_d = cumul_d + d
        if i == 0:
            min_cumul_d = cumul_d
            median = evi
        else:
            if cumul_d <= min_cumul_d:
                min_cumul_d = cumul_d
                median = evi
    return median


def load_similarity_matrix(fname):
    '''
    Load a binary similarity matrix from file
    '''

    simmat = num.load(fname)
    return simmat


def save_similarity_matrix(simmat, fname):
    '''
    Save a binary similarity matrix from file
    '''

    num.save(fname, simmat)


def save_all(events, eventsclusters, clusters, conf, resdir):
    '''
    Save all results of the clustering analysis
    '''

    # save clustered events
    fn = os.path.join(resdir, 'clustered_events.pf')
    if os.path.isfile(fn):
        os.remove(fn)
    for iev, ev in enumerate(events):
        tag = 'cluster:'+str(eventsclusters[iev])
        if ev.tags is not None:
            ev.tags.append(tag)
        else:
            ev.tags = [tag]
    model.dump_events(events, fn)

    # save events of each cluster
    for cluster_id in clusters:
        wished_events = []
        fn = os.path.join(resdir, 'cluster.'+str(cluster_id)+'.events.pf')
        for iev, evcl in enumerate(eventsclusters):
            if evcl == cluster_id:
                wished_events.append(events[iev])
        model.dump_events(wished_events, fn)

    # save clusters
    fn = os.path.join(resdir, 'processed.clusters')
    save_obj(clusters, fn)

    # save eventsclusters
    fn = os.path.join(resdir, 'processed.eventsclusters')
    save_obj(eventsclusters, fn)
