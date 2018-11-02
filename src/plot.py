'''This module provides basic cluster processing for seismic events.'''

import sys
import os
import datetime
import math
import matplotlib.pyplot as plt
import numpy as num
from pyrocko import model, gmtpy
from pyrocko.automap import Map
from pyrocko import orthodrome as od
from pyrocko import moment_tensor as pmt
from matplotlib import dates, colors

epsilon = 1e-6
km = 1000.


def CartesianToLambert(e, n, u):
    '''
    Convert axis orientation to coordinate for axis plot
    (using Lambert azimuthal equal-area projection)
    '''
    x = (1/math.sqrt(2.))*e*math.sqrt(2/(1-u))
    y = (1/math.sqrt(2.))*n*math.sqrt(2/(1-u))
    return x, y


def get_axis_coords(events):
    '''
    Get axis plot coordinates for all events
    '''
    xs, ys = [], []

    for ev in events:
        pax = ev.moment_tensor.p_axis()
        tax = ev.moment_tensor.t_axis()
        bax = ev.moment_tensor.null_axis()
        p = [pax[0, 0], pax[0, 1], pax[0, 2]]
        t = [tax[0, 0], tax[0, 1], tax[0, 2]]
        b = [bax[0, 0], bax[0, 1], bax[0, 2]]

        if p[2] < 0:
            p = [-p[0], -p[1], -p[2]]
        if t[2] < 0:
            t = [-t[0], -t[1], -t[2]]
        if b[2] < 0:
            b = [-b[0], -b[1], -b[2]]

        px, py = CartesianToLambert(p[1], p[0], -p[2])
        tx, ty = CartesianToLambert(t[1], t[0], -t[2])
        bx, by = CartesianToLambert(b[1], b[0], -b[2])

        xs.append([px, tx, bx])
        ys.append([py, ty, by])

    return xs, ys


def getCoordinatesHudsonPlot(mts):
    '''
    Get Hudson plot coordinates for all events
    '''
    us, vs = [], []
    for mt in mts:
        mt = pmt.values_to_matrix(mt)
        eig_m = pmt.eigh_check(mt)[0]
        m3, m2, m1 = eig_m / num.max(num.abs(eig_m))
        u = -2./3. * (m1 + m3 - 2.0*m2)
        v = 1./3. * (m1 + m2 + m3)
        us.append(u)
        vs.append(v)
    return us, vs


def get_triangle_coords(events):
    '''
    Get triangle plot coordinates for all events
    '''
    xs, ys, cs = [], [], []

    for ev in events:
        pax = ev.moment_tensor.p_axis()
        tax = ev.moment_tensor.t_axis()
        bax = ev.moment_tensor.null_axis()
        p = [pax[0, 0], pax[0, 1], pax[0, 2]]
        t = [tax[0, 0], tax[0, 1], tax[0, 2]]
        b = [bax[0, 0], bax[0, 1], bax[0, 2]]

#        t,p,b=sds2tpb(ev.strike,ev.dip,ev.rake)
        a3526 = 35.26*math.pi/180.

        deltab = math.asin(abs(b[2]))
        deltat = math.asin(abs(t[2]))
        deltap = math.asin(abs(p[2]))

        psi = math.atan(math.sin(deltat)/math.sin(deltap))-0.25*math.pi
        den = math.sin(a3526)*math.sin(deltab) +\
            math.cos(a3526)*math.cos(deltab)*math.cos(psi)

        if abs(den) <= epsilon:
            den = epsilon

        x = math.cos(deltab)*math.sin(psi)/den
        y = (math.cos(a3526)*math.sin(deltab) -
             math.sin(a3526)*math.cos(deltab)*math.cos(psi))/den
        s2p = (math.sin(deltap))**2
        s2t = (math.sin(deltat))**2
        s2b = (math.sin(deltab))**2
        c = (s2t, s2b, s2p)

        xs.append(x)
        ys.append(y)
        cs.append(c)

    return xs, ys, cs


def view_and_savefig_similarity_matrix(simmat, figname, title):
    '''
    View and save a similarity matrix
    '''
    nev = len(simmat)

    f = plt.figure(1, figsize=(12, 10), facecolor='w', edgecolor='k')
    grid = plt.GridSpec(10, 10, wspace=0.2, hspace=0.2)

    plt.subplot(grid[0:9, 0:9])

    plt.imshow(simmat, interpolation='none', cmap='GnBu_r')
    plt.xlim(xmax=nev, xmin=0)
    plt.ylim(ymax=nev, ymin=0)

    plt.xlabel("Event number")
    plt.ylabel("Event number")
    plt.title(title)
    f.savefig(figname)
    plt.show()


def view_similarity_matrix(simmat, title):
    '''
    View (only) a similarity matrix
    '''
    nev = len(simmat)

    plt.figure(1, figsize=(12, 10), facecolor='w', edgecolor='k')
    grid = plt.GridSpec(10, 10, wspace=0.2, hspace=0.2)

    plt.subplot(grid[0:9, 0:9])

    plt.imshow(simmat, interpolation='none', cmap='GnBu_r')
    plt.xlim(xmax=nev, xmin=0)
    plt.ylim(ymax=nev, ymin=0)

    plt.xlabel("Event number")
    plt.ylabel("Event number")
    plt.title(title)
    plt.show()


def savefig_similarity_matrix(simmat, figname, title):
    '''
    Save (only) a similarity matrix
    '''
    nev = len(simmat)

    f = plt.figure(1, figsize=(12, 10), facecolor='w', edgecolor='k')
    grid = plt.GridSpec(10, 10, wspace=0.2, hspace=0.2)

    plt.subplot(grid[0:9, 0:9])

    plt.imshow(simmat, interpolation='none', cmap='GnBu_r')
    plt.xlim(xmax=nev, xmin=0)
    plt.ylim(ymax=nev, ymin=0)

    plt.xlabel("Event number")
    plt.ylabel("Event number")
    plt.title(title)
    f.savefig(figname)


def cluster_to_color(cluster_id):
    '''
    Given a cluster id provide the corresponding color
    '''
    mycolors = ['black', 'red', 'blue', 'green', 'darkviolet', 'gold',
                'darkorange', 'dodgerblue', 'brown', 'lightseagreen', 'plum',
                'lawngreen', 'palevioletred', 'royalblue', 'limegreen',
                'indigo']
    my_def_color = 'gainsboro'
    if cluster_id > (len(mycolors)-2):
        color = my_def_color
    else:
        color = mycolors[cluster_id+1]
    return color


def color2rgb(col):
    '''
    Return a red/green/blue string for GMT for a given color name
    '''
    colarray = colors.to_rgb(col)
    r, g, b = int(255*colarray[0]), int(255*colarray[1]), int(255*colarray[2])
    rgb = str(r)+'/'+str(g)+'/'+str(b)
    return rgb


def plot_spatial(events, eventsclusters, clusters, conf, plotdir):
    '''
    Plot map of seismicity clusters
    '''
    lats = [ev.lat for ev in events]
    lons = [ev.lon for ev in events]
#    deps = [ev.depth for ev in events]
#    colors = [cluster_to_color(clid) for clid in eventsclusters]

    if conf.sw_filterevent:
        latmin, latmax = conf.latmin, conf.latmax
        lonmin, lonmax = conf.lonmin, conf.lonmax
#        depmin, depmax = conf.depthmin, conf.depthmax
        if latmin > latmax:
            print('cases over lon +-180 still to be implemented')
            sys.exit()
        center = model.event(0.5*(latmin+latmax), 0.5*(lonmin+lonmax))
    else:
        latmin, latmax = min(lats)-0.1, max(lats)+0.1
        lonmin, lonmax = min(lons)-0.1, max(lons)+0.1
#        depmin = 0.*km
#        depmax = 1.05*max(deps)
        center = od.Loc(0.5*(latmin+latmax), 0.5*(lonmin+lonmax))

    # Map size
    if conf.map_radius is not None:
        safe_radius = conf.map_radius
    else:
        corners = [od.Loc(latmin, lonmin), od.Loc(latmin, lonmax)]
        dist1 = od.distance_accurate50m(center, corners[0])
        dist2 = od.distance_accurate50m(center, corners[1])
        safe_radius = max(dist1, dist2)

    # Generate the basic map
    m = Map(
        lat=center.lat,
        lon=center.lon,
        radius=safe_radius,
        width=30., height=30.,
        show_grid=False,
        show_topo=False,
        color_dry=(238, 236, 230),
        topo_cpt_wet='light_sea_uniform',
        topo_cpt_dry='light_land_uniform',
        illuminate=True,
        illuminate_factor_ocean=0.15,
        show_rivers=False,
        show_plates=False)

    if conf.sw_filterevent:
        rectlons = [lonmin, lonmin, lonmax, lonmax, lonmin]
        rectlats = [latmin, latmax, latmax, latmin, latmin]
        m.gmt.psxy(in_columns=(rectlons, rectlats),
                   W='thin,0/0/0,dashed', *m.jxyr)

    # Draw some larger cities covered by the map area
    m.draw_cities()

    # Events in clusters
    for id_cluster in clusters:
        col = cluster_to_color(id_cluster)
        mylats, mylons = [], []
        for iev, evcl in enumerate(eventsclusters):
            if evcl == id_cluster:
                mylats.append(events[iev].lat)
                mylons.append(events[iev].lon)
        m.gmt.psxy(in_columns=(mylons, mylats), S='c7p',
                   G=color2rgb(col), *m.jxyr)

    figname = os.path.join(plotdir, 'plot_map.'+conf.figure_format)
    m.save(figname)
#    m.show()


def plot_spatial_with_dcs(events, eventsclusters, clusters, conf, plotdir):
    '''
    Plot map of seismicity clusters, with focal mechanisms
    '''
    lats = [ev.lat for ev in events]
    lons = [ev.lon for ev in events]
#    deps = [ev.depth for ev in events]
#    colors = [cluster_to_color(clid) for clid in eventsclusters]

    if conf.sw_filterevent:
        latmin, latmax = conf.latmin, conf.latmax
        lonmin, lonmax = conf.lonmin, conf.lonmax
#        depmin, depmax = conf.depthmin, conf.depthmax
        if latmin > latmax:
            print('cases over lon +-180 still to be implemented')
            sys.exit()
        center = model.event(0.5*(latmin+latmax), 0.5*(lonmin+lonmax))
    else:
        latmin, latmax = min(lats)-0.1, max(lats)+0.1
        lonmin, lonmax = min(lons)-0.1, max(lons)+0.1
#        depmin = 0.*km
#        depmax = 1.05*max(deps)
        center = od.Loc(0.5*(latmin+latmax), 0.5*(lonmin+lonmax))

    # Map size
    if conf.map_radius is not None:
        safe_radius = conf.map_radius
    else:
        corners = [od.Loc(latmin, lonmin), od.Loc(latmin, lonmax)]
        dist1 = od.distance_accurate50m(center, corners[0])
        dist2 = od.distance_accurate50m(center, corners[1])
        safe_radius = max(dist1, dist2)

    # Generate the basic map
    m = Map(
        lat=center.lat,
        lon=center.lon,
        radius=safe_radius,
        width=30., height=30.,
        show_grid=False,
        show_topo=False,
        color_dry=(238, 236, 230),
        topo_cpt_wet='light_sea_uniform',
        topo_cpt_dry='light_land_uniform',
        illuminate=True,
        illuminate_factor_ocean=0.15,
        show_rivers=False,
        show_plates=False)

    if conf.sw_filterevent:
        rectlons = [lonmin, lonmin, lonmax, lonmax, lonmin]
        rectlats = [latmin, latmax, latmax, latmin, latmin]
        m.gmt.psxy(in_columns=(rectlons, rectlats),
                   W='thin,0/0/0,dashed', *m.jxyr)

    # Draw some larger cities covered by the map area
    m.draw_cities()

    # Events in clusters
    factor_symbl_size = 5.
    beachball_symbol = 'd'

    for id_cluster in clusters:
        col = cluster_to_color(id_cluster)
        g_col = color2rgb(col)
        for iev, evcl in enumerate(eventsclusters):
            if evcl == id_cluster:
                ev = events[iev]
                devi = ev.moment_tensor.deviatoric()
                beachball_size = 3.*factor_symbl_size
                mt = devi.m_up_south_east()
                mt = mt / ev.moment_tensor.scalar_moment() \
                    * pmt.magnitude_to_moment(5.0)
                m6 = pmt.to6(mt)

                if m.gmt.is_gmt5():
                    kwargs = dict(M=True, S='%s%g' %
                                  (beachball_symbol[0],
                                   (beachball_size)/gmtpy.cm))
                else:
                    kwargs = dict(S='%s%g' %
                                  (beachball_symbol[0],
                                   (beachball_size)*2 / gmtpy.cm))

                data = (ev.lon, ev.lat, 10) + tuple(m6) + (1, 0, 0)

                m.gmt.psmeca(
                    in_rows=[data],
                    G=g_col,
                    E='white',
                    W='1p,%s' % g_col,
                    *m.jxyr,
                    **kwargs)

    figname = os.path.join(plotdir, 'plot_map_with_dcs.'+conf.figure_format)
    m.save(figname)
#    m.show()


def plot_tm(events, eventsclusters, clusters, conf, plotdir):
    '''
    Plot magnitude vs time for the eismicity clusters
    '''
    times = [ev.time for ev in events]
    orig_dates = [datetime.datetime.fromtimestamp(ev.time) for ev in events]
    mpl_dates = dates.date2num(orig_dates)
    mags = [ev.magnitude for ev in events]
    colors = [cluster_to_color(clid) for clid in eventsclusters]

    dates_format = dates.DateFormatter('%Y-%m-%d')

    if conf.sw_filterevent:
        tmin, tmax = conf.tmin, conf.tmax
        magmin, magmax = conf.magmin, conf.magmax
    else:
        if (max(times)-min(times))/86400. > 720.:
            dt = 86400.
            dates_loc = dates.YearLocator()
        elif (max(times)-min(times))/86400. > 10.:
            dt = 86400.
            dates_loc = dates.MonthLocator()
        elif (max(times)-min(times))/3600. > 10.:
            dt = 3600.
            dates_loc = dates.DayLocator()
        else:
            dt = 1.
            dates_loc = dates.HourLocator()
            dates_format = dates.DateFormatter('%Y-%m-%d %h:%m:%s')
        tmin, tmax = min(times)-dt, max(times)+dt
        magmin, magmax = min(mags)-0.1, max(mags)+0.1
    dmin = dates.date2num(datetime.datetime.fromtimestamp(tmin))
    dmax = dates.date2num(datetime.datetime.fromtimestamp(tmax))

    f = plt.figure()
    f.suptitle('Temporal evolution of seismicity clusters', fontsize=14)

    ax = f.add_subplot(111)
    ax.scatter(mpl_dates, mags, s=15., c=colors, alpha=0.5)

    ax.xaxis.set_major_locator(dates_loc)
    ax.xaxis.set_major_formatter(dates_format)

    plt.xlim(xmax=dmax, xmin=dmin)
    plt.ylim(ymax=magmax, ymin=magmin)
    plt.xticks(rotation=45.)
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.subplots_adjust(bottom=.3)

    figname = os.path.join(plotdir, 'plot_tm.'+conf.figure_format)
    f.savefig(figname)
#    plt.show()


def plot_triangle(events, eventsclusters, clusters, conf, plotdir):
    '''
    Plot a triangular diagram for the seismicity clusters
    '''
    xs, ys, cs = get_triangle_coords(events)
    colors = [cluster_to_color(clid) for clid in eventsclusters]

    f = plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
    f.suptitle('Triangular diagram for seismicity clusters', fontsize=14)
    plt.xlim(xmin=-1.4, xmax=1.4)
    plt.ylim(ymin=-1.2, ymax=1.6)
    plt.plot([-1.2247, 1.2247, 0., -1.2247],
             [-0.7070, -0.7070, 1.4144, -0.7070], c='k', lw=1.)
    plt.scatter(xs, ys, c=colors, alpha=0.5)
    plt.text(0., 1.44, 'Strike-Slip', fontsize=12, ha='center')
    plt.text(-1.25, -0.78, 'Normal Fault', fontsize=12, ha='center')
    plt.text(1.25, -0.78, 'Thrust Fault', fontsize=12, ha='center')
    plt.axis('off')
#    plt.subplots_adjust(left=.3,right=.3,bottom=.3,top=.3)

    figname = os.path.join(plotdir, 'plot_triangle.'+conf.figure_format)
    f.savefig(figname)
#    plt.show()


def plot_axis(events, eventsclusters, clusters, conf, plotdir):
    '''
    Plot axis orientations for the seismicity clusters
    '''
    c1, c2, c3 = [2., 1.8], [6., 1.8], [10., 1.8]
    csize = 1.5
    xs, ys = get_axis_coords(events)
    pxs, pys = [x[0]*csize+c1[0] for x in xs], [y[0]*csize+c1[1] for y in ys]
    txs, tys = [x[1]*csize+c2[0] for x in xs], [y[1]*csize+c2[1] for y in ys]
    bxs, bys = [x[2]*csize+c3[0] for x in xs], [y[2]*csize+c3[1] for y in ys]
    colors = [cluster_to_color(clid) for clid in eventsclusters]

    f = plt.figure(figsize=(12, 4), facecolor='w', edgecolor='k')
    f.suptitle('Pressure (P), tension (T) and null (B) axis' +
               ' for seismicity clusters', fontsize=14)

    plt.xlim(xmin=0., xmax=12.)
    plt.ylim(ymin=0., ymax=4.)
    plt.text(c1[0], 3.4, 'P axis', fontsize=12, ha='center')
    plt.text(c2[0], 3.4, 'T axis', fontsize=12, ha='center')
    plt.text(c3[0], 3.4, 'B axis', fontsize=12, ha='center')
    plt.axis('off')

    plt.scatter(pxs, pys, c=colors, alpha=0.5)
    plt.scatter(txs, tys, c=colors, alpha=0.5)
    plt.scatter(bxs, bys, c=colors, alpha=0.5)

    ax = plt.gca()
    ax.add_artist(plt.Circle((c1[0], c1[1]), csize, color='black', fill=False))
    ax.add_artist(plt.Circle((c2[0], c2[1]), csize, color='black', fill=False))
    ax.add_artist(plt.Circle((c3[0], c3[1]), csize, color='black', fill=False))

#    plt.scatter(xs,ys,c=colors,alpha=0.5)

    figname = os.path.join(plotdir, 'plot_axis.'+conf.figure_format)
    f.savefig(figname)
#    plt.show()


def plot_hudson(events, eventsclusters, clusters, conf, plotdir):
    '''
    Plot a Hudson diagram for the seismicity clusters
    '''
    colors = [cluster_to_color(clid) for clid in eventsclusters]
    mts = [ev.moment_tensor for ev in events]
    us, vs = getCoordinatesHudsonPlot(mts)

    f = plt.figure(figsize=(12, 10), facecolor='w', edgecolor='k')
    f.suptitle('Hudson plot for seismicity clusters', fontsize=14)

    plt.plot([-1.33, 0.], [-0.33, 1.], 'k')
    plt.plot([0., 1.33], [1., 0.33], 'k')
    plt.plot([1.33, 0.], [0.33, -1.], 'k')
    plt.plot([0., -1.33], [-1., -0.33], 'k')
    plt.plot([-1., 1.], [0., 0.], 'k')
    plt.plot([0., 0.], [-1., 1.], 'k')

    plt.scatter(0.667, -0.333, 100, 'k', '+')
    plt.text(0.69, -0.37, '-Dipole', fontsize=12, ha='left')
    plt.scatter(1., 0., 100, 'k', '+')
    plt.text(1.03, -0.03, '-CLVD', fontsize=12, ha='left')
    plt.scatter(0.444, -0.556, 100, 'k', '+')
    plt.text(0.47, -0.59, '-Crack', fontsize=12, ha='left')
    plt.scatter(0., -1., 100, 'k', '+')
    plt.text(0.0, -1.075, '-ISO', fontsize=12, ha='center')

    plt.scatter(-0.667, 0.333, 100, 'k', '+')
    plt.text(-0.69, 0.37, '+Dipole', fontsize=12, ha='right')
    plt.scatter(-1., 0., 100, 'k', '+')
    plt.text(-1.03, 0.03, '+CLVD', fontsize=12, ha='right')
    plt.scatter(-0.444, +0.556, 100, 'k', '+')
    plt.text(-0.47, +0.59, '+Crack', fontsize=12, ha='right')
    plt.scatter(0., 1., 100, 'k', '+')
    plt.text(0.0, 1.025, '+ISO', fontsize=12, ha='center')

    plt.axis('off')

    plt.scatter(us, vs, c=colors, alpha=0.5)

    figname = os.path.join(plotdir, 'plot_hudson.'+conf.figure_format)
    f.savefig(figname)
#    plt.show()


def plot_similarity_matrices(events, eventsclusters, clusters, conf, plotdir):

    fmatname1 = os.path.join(conf.project_dir, 'simmat_temporal.npy')
    fmatname2 = os.path.join(conf.project_dir, 'simmat_clustered.npy')
    simmat1 = num.load(fmatname1)
    simmat2 = num.load(fmatname2)
    nev = len(simmat1)

    nclusters = len(clusters)
    cl_sizes = [len(clusters[i-1]) for i in range(nclusters)]
    cl_cumul_sizes = [sum(cl_sizes[:i+1]) for i in range(len(cl_sizes))]

#    plt.figure(1)
    f = plt.figure(figsize=(12, 6), facecolor='w', edgecolor='k')
    f.suptitle('Similarity matrices', fontsize=14)

    plt.subplot(121)
    plt.imshow(simmat1, interpolation='none', cmap='GnBu_r')
#    plt.imshow(simmat1,interpolation='none',cmap='coolwarm_r')
    plt.xlim(xmax=nev, xmin=0)
    plt.ylim(ymax=nev, ymin=0)
    plt.xlabel("Event number")
    plt.ylabel("Event number")
    plt.title("Sorted chronologically")

    plt.subplot(122)
    plt.imshow(simmat2, interpolation='none', cmap='GnBu_r')
#    plt.imshow(simmat2,interpolation='none',cmap='coolwarm_r')
    plt.xlim(xmax=nev, xmin=0)
    plt.ylim(ymax=nev, ymin=0)

    for ccs in cl_cumul_sizes:
        x = [0, nev]
        y = [ccs, ccs]
        plt.plot(x, y, 'red', ls='--', lw=1)
#        plt.plot(x,y,'midnightblue',ls='--',lw=1)
        y = [0, nev]
        x = [ccs, ccs]
        plt.plot(x, y, 'red', ls='--', lw=1)
#        plt.plot(x,y,'midnightblue',ls='--',lw=1)

    plt.xlabel("Event number")
    plt.ylabel("Event number")
    plt.title("Sorted after clustering")
#    plt.show()

    figname = os.path.join(plotdir,
                           'plot_similarity_matrices.'+conf.figure_format)
    f.savefig(figname)


def plot_all(events, eventsclusters, clusters, conf, plotdir):
    plot_similarity_matrices(events, eventsclusters, clusters, conf, plotdir)
    plot_tm(events, eventsclusters, clusters, conf, plotdir)
    plot_triangle(events, eventsclusters, clusters, conf, plotdir)
    plot_axis(events, eventsclusters, clusters, conf, plotdir)
    plot_hudson(events, eventsclusters, clusters, conf, plotdir)
    plot_spatial(events, eventsclusters, clusters, conf, plotdir)
    plot_spatial_with_dcs(events, eventsclusters, clusters, conf, plotdir)
