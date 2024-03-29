#!/usr/local/bin/python3


import sys
import os
import shutil
from optparse import OptionParser

from pyrocko import model, util
from pyrocko.client import catalog
from pyrocko.guts import load

from seiscloud import config
from seiscloud import cluster as sccluster
from seiscloud import plot as scplot

'''
Seismicity clustering
0. example - Create example configuration file
1. init    - Initialize a new project
2. matrix  - Compute or read similarity matrix
3. cluster - Run DBSCAN clustering algorithm
4. plot    - Plot clustering results
'''

km = 1000.


def d2u(d):
    if isinstance(d, dict):
        return dict((k.replace('-', '_'), v) for (k, v) in d.items())
    else:
        return d.replace('-', '_')


subcommand_descriptions = {
    'example': 'create example configuration file',
    'init': 'initialize a project',
    'matrix': 'compute or read similarity matrix',
    'cluster': 'run DBSCAN clustering',
    'plot': 'plot results',
}

subcommand_usages = {
    'example': 'example [options]',
    'init': 'init <configfile> [options]',
    'matrix': 'matrix <configfile> [options]',
    'cluster': 'cluster <configfile> [options]',
    'plot': 'plot <configfile> [options]',
}

subcommands = subcommand_descriptions.keys()

program_name = 'seiscloud'

usage_tdata = d2u(subcommand_descriptions)
usage_tdata['program_name'] = program_name

usage = '''%(program_name)s <subcommand> [options]

Subcommands:

    example         %(example)s
    init            %(init)s
    matrix          %(matrix)s
    cluster         %(cluster)s
    plot            %(plot)s

To get further help and a list of available options for any subcommand run:

    %(program_name)s <subcommand> --help

''' % usage_tdata


def die(message, err=''):
    if err:
        sys.exit('%s: error: %s \n %s' % (program_name, message, err))
    else:
        sys.exit('%s: error: %s' % (program_name, message))


def help_and_die(parser, message):
    parser.print_help(sys.stderr)
    sys.stderr.write('\n')
    die(message)


def cl_parse(command, args, setup=None, details=None):
    '''
    Parsing seiscloud call
    '''
    usage = subcommand_usages[command]
    descr = subcommand_descriptions[command]

    if isinstance(usage, str):
        usage = [usage]

    susage = '%s %s' % (program_name, usage[0])
    for s in usage[1:]:
        susage += '\n%s%s %s' % (' '*7, program_name, s)

    description = descr[0].upper() + descr[1:] + '.'

    if details:
        description = description + '\n\n%s' % details

    parser = OptionParser(usage=susage, description=description)

    if setup:
        setup(parser)

    (options, args) = parser.parse_args(args)

    return parser, options, args


def command_example(args):
    '''
    Execution of command example
    '''

    def setup(parser):

        parser.add_option(
            '--force', dest='force', action='store_true',
            help='overwrite existing configuration file')

    parser, options, args = cl_parse('example', args, setup)

    fn_config = 'seiscloud.example.config'

    if ((not options.force) and (os.path.exists('seiscloud.example.config'))):
        die('file exists: %s; use force option' % fn_config)
    else:
        conf = config.generate_default_config()
        conf.dump(filename=fn_config)
        print('Created a fresh config file "%s"' % fn_config)


def command_init(args):
    '''
    Execution of command init
    '''

    def setup(parser):

        parser.add_option(
            '--force', dest='force', action='store_true',
            help='overwrite existing project directory')

    parser, options, args = cl_parse('init', args, setup)

    if len(args) != 1:
        help_and_die(parser, 'missing argument')
    else:
        fn_config = args[0]

    if not os.path.isfile(fn_config):
        die('config file missing: %s' % fn_config)

    conf = load(filename=fn_config)
    config.check(conf)

    if ((not options.force) and (os.path.isdir(conf.project_dir))):
        die('project dir exists: %s; use force option' % conf.project_dir)
    else:
        if os.path.isdir(conf.project_dir):
            shutil.rmtree(conf.project_dir)
        os.mkdir(conf.project_dir)
        conf.dump(filename=os.path.join(conf.project_dir, 'seiscloud.config'))

        if conf.catalog_origin == 'file':
            src = conf.catalog_fn
            try:
                events = model.load_events(conf.catalog_fn)
            except:
                die('catalog file not readable: %s' % conf.catalog_fn)
        else:
            if conf.catalog_origin == 'globalcmt':
                orig_catalog = catalog.GlobalCMT()
            elif conf.catalog_origin == 'geofon':
                orig_catalog = catalog.Geofon()
            elif conf.catalog_origin == 'usgs':
                orig_catalog = catalog.USGS()
            else:
                die('unknown remote catalog: %s' % conf.catalog_origin)
            if conf.latmin > conf.latmax:
                die('bad latitude range: %s %s' % (conf.latmin, conf.latmax))
            if conf.depthmin > conf.depthmax:
                die('bad depth range: %s %s' % (conf.depthmin, conf.depthmax))
            if conf.magmin >= conf.magmax:
                die('bad magnitude range: %s %s' % (conf.magmin, conf.magmax))
            events = orig_catalog.get_events(
                        time_range=(util.str_to_time(conf.tmin),
                                    util.str_to_time(conf.tmax)),
                        magmin=conf.magmin,
                        latmin=conf.latmin,
                        latmax=conf.latmax,
                        lonmin=conf.lonmin,
                        lonmax=conf.lonmax)

        dst = os.path.join(conf.project_dir, 'catalog.pf')
        if conf.sw_filter_events:
            sel1ev = [ev for ev in events if
                      ev.lat >= conf.latmin and
                      ev.lat <= conf.latmax]
            if conf.lonmin <= conf.lonmax:
                sel2ev = [ev for ev in sel1ev if
                          ev.lon >= conf.lonmin and
                          ev.lon <= conf.lonmax]
            else:
                sel2ev = [ev for ev in sel1ev if
                          ev.longitude >= conf.lonmin or
                          ev.longitude <= conf.lonmax]
            sel3ev = [ev for ev in sel2ev if
                      ev.depth >= conf.depthmin and
                      ev.depth <= conf.depthmax]
            sel4ev = [ev for ev in sel3ev if ev.magnitude is not None]
            sel5ev = [ev for ev in sel4ev if
                      ev.magnitude >= conf.magmin and
                      ev.magnitude <= conf.magmax]
            selevents = sel5ev
        else:
            selevents = events
        model.dump_events(selevents, dst)

        print('Project directory prepared "%s"' % conf.project_dir)
        print('Catalog prepared with %s events' % str(len(selevents)))


def command_matrix(args):
    '''
    Execution of command matrix
    '''

    def setup(parser):

        parser.add_option(
            '--force', dest='force', action='store_true',
            help='overwrite existing project directory')

        parser.add_option(
            '--view', dest='view', action='store_true',
            help='view similarity matrix')

        parser.add_option(
            '--savefig', dest='savefig', action='store_true',
            help='save figure of similarity matrix')

    parser, options, args = cl_parse('matrix', args, setup)

    if len(args) != 1:
        help_and_die(parser, 'missing argument')
    else:
        fn_config = args[0]

    if not os.path.isfile(fn_config):
        die('config file missing: %s' % fn_config)

    conf = load(filename=fn_config)
    config.check(conf)

    if not os.path.isdir(conf.project_dir):
        die('project directory missing: %s' % conf.project_dir)

    simmat_temporal_fn = os.path.join(conf.project_dir, 'simmat_temporal.npy')

    if ((not options.force) and (os.path.isfile(simmat_temporal_fn))):
        die('similarity matrix exists: %s; use force option'
            % simmat_temporal_fn)

    catalog_ref_fn = os.path.join(conf.project_dir, 'catalog.pf')
    if os.path.isfile(catalog_ref_fn):
        allevents = model.load_events(catalog_ref_fn)
    else:
        die('catalog missing: %s' % catalog_ref_fn)

    if conf.sw_simmat:
        if not os.path.isfile(conf.sim_mat_fn):
            die('similarity matrix missing: %s' % conf.sim_mat_fn)
        if conf.sim_mat_type == 'binary':
            if os.path.isfile(conf.sim_mat_fn):
                simmat = sccluster.load_similarity_matrix(conf.sim_mat_fn)
            else:
                die('cannot read similarity matrix: %s' % conf.sim_mat_fn)
        else:
            die('ascii format for similarity matrix not yet implemented')

        if len(allevents) != len(simmat):
            print(len(allevents), len(simmat))
            die('clustering stopped, number of events ' +
                'differs from matrix size')

        new_catalog_fn = os.path.join(conf.project_dir,
                                      'events_to_be_clustered.pf')
        model.dump_events(allevents, new_catalog_fn)

    else:
        if conf.metric in config.acceptable_mt_based_metrics:
            events = [ev for ev in allevents if ev.moment_tensor is not None]
        else:
            events = [ev for ev in allevents]
        new_catalog_fn = os.path.join(conf.project_dir,
                                      'events_to_be_clustered.pf')
        model.dump_events(events, new_catalog_fn)

        simmat = sccluster.compute_similarity_matrix(events, conf)

    sccluster.save_similarity_matrix(simmat, simmat_temporal_fn)

    simmat_fig_fn = os.path.join(conf.project_dir,
                                 'simmat_temporal.'+conf.figure_format)
    if options.view and options.savefig:
        scplot.view_and_savefig_similarity_matrix(simmat, simmat_fig_fn,
                                                  'Sorted chronologically')
    else:
        if options.view:
            scplot.view_similarity_matrix(simmat, 'Sorted chronologically')
        if options.savefig:
            scplot.savefig_similarity_matrix(simmat, simmat_fig_fn,
                                             'Sorted chronologically')

    print('Similarity matrix computed and stored as "%s"' % simmat_temporal_fn)
    if options.savefig:
        print('Similarity matrix figure saved as "%s"' % simmat_fig_fn)


def command_cluster(args):
    '''
    Execution of command cluster
    '''

    def setup(parser):

        parser.add_option(
            '--force', dest='force', action='store_true',
            help='overwrite existing project directory')

        parser.add_option(
            '--view', dest='view', action='store_true',
            help='view similarity matrix after clustering')

        parser.add_option(
            '--savefig', dest='savefig', action='store_true',
            help='save figure of similarity matrix after clustering')

    parser, options, args = cl_parse('cluster', args, setup)

    if len(args) != 1:
        help_and_die(parser, 'missing argument')
    else:
        fn_config = args[0]

    if not os.path.isfile(fn_config):
        die('config file missing: %s' % fn_config)

    conf = load(filename=fn_config)
    config.check(conf)

    if not os.path.isdir(conf.project_dir):
        die('project directory missing: %s' % conf.project_dir)

    resdir = os.path.join(conf.project_dir, 'clustering_results')
    if not(options.force):
        if (os.path.isdir(resdir)):
            die('clustering result directory exists; use force option')
    if options.force:
        if (os.path.isdir(resdir)):
            shutil.rmtree(resdir)
    os.mkdir(resdir)

    simmat_temporal_fn = os.path.join(conf.project_dir, 'simmat_temporal.npy')
    if not os.path.isfile(simmat_temporal_fn):
        die('similarity matrix does not exists: %s; ' +
            'use seiscloud matrix first' % simmat_temporal_fn)

    new_catalog_fn = os.path.join(conf.project_dir,
                                  'events_to_be_clustered.pf')
    if not os.path.isfile(new_catalog_fn):
        die('catalog of selected events does not exists: %s; ' +
            'use seiscloud matrix first' % new_catalog_fn)

    simmat_temp = sccluster.load_similarity_matrix(simmat_temporal_fn)
    events = model.load_events(new_catalog_fn)
    for iev, ev in enumerate(events):
        similar_ones = [i for i in simmat_temp[iev] if i < conf.dbscan_eps]
        nsimilar = len(similar_ones)-1
        if ev.tags is None:
            ev.tags = []
        if nsimilar > conf.dbscan_nmin:
            ev.tags.append('dbtype:core')

    eventsclusters = sccluster.dbscan(simmat_temp,
                                      conf.dbscan_nmin, conf.dbscan_eps,
                                      conf.sw_force_cluster_all)
    clusters = sccluster.get_clusters(events, eventsclusters)
    if min(eventsclusters) == -1:
        noise_cluster_empty = False
    else:
        noise_cluster_empty = True

    sccluster.save_all(events, eventsclusters, clusters, conf, resdir)
    simmat_clus = sccluster.get_simmat_clustered(events, eventsclusters,
                                                 clusters, conf, resdir,
                                                 simmat_temp)

    simmat_clustered_fn = os.path.join(conf.project_dir,
                                       'simmat_clustered.npy')
    sccluster.save_similarity_matrix(simmat_clus, simmat_clustered_fn)

    print('I run seiscloud for the project in "%s"' % conf.project_dir)
    n_clusters = len(clusters)
    if not noise_cluster_empty:
        n_clusters = n_clusters - 1
    print(str(n_clusters)+' cluster(s) found')
    if noise_cluster_empty:
        n_unclustered = 0
    else:
        n_unclustered = len(clusters[-1])
    print(str(n_unclustered)+' unclustered events')

    simmat_fig_fn = os.path.join(conf.project_dir,
                                 'simmat_clustered.'+conf.figure_format)
    if options.view and options.savefig:
        scplot.view_and_savefig_similarity_matrix(simmat_clus, simmat_fig_fn,
                                                  'Sorted after clustering')
    else:
        if options.view:
            scplot.view_similarity_matrix(simmat_clus,
                                          'Sorted after clustering')
        if options.savefig:
            scplot.savefig_similarity_matrix(simmat_clus, simmat_fig_fn,
                                             'Sorted after clustering')

    for cluster in clusters:
        selevents = clusters[cluster]
        median = sccluster.get_median(selevents, conf)
        fn_median = os.path.join(resdir,
                                 'median_cluster' + str(cluster) + '.pf')
        model.dump_events([median], fn_median)
#        print(cluster,len(selevents))

    print('Similarity matrix after clustering computed and stored as "%s"'
          % simmat_clustered_fn)
    if options.savefig:
        print('Similarity matrix figure saved as "%s"' % simmat_fig_fn)


def command_plot(args):
    '''
    Execution of command plot
    '''

    def setup(parser):

        parser.add_option(
            '--force', dest='force', action='store_true',
            help='overwrite existing project directory')

    parser, options, args = cl_parse('plot', args, setup)

    if len(args) != 1:
        help_and_die(parser, 'missing argument')
    else:
        fn_config = args[0]

    if not os.path.isfile(fn_config):
        die('config file missing: %s' % fn_config)

    conf = load(filename=fn_config)
    config.check(conf)

    if not os.path.isdir(conf.project_dir):
        die('project directory missing: %s' % conf.project_dir)

    resdir = os.path.join(conf.project_dir, 'clustering_results')
    if not os.path.isdir(resdir):
        die('clustering results missing: %s' % resdir)

    plotdir = os.path.join(conf.project_dir, 'clustering_plots')
    resdir = os.path.join(conf.project_dir, 'clustering_results')
    if not(options.force):
        if (os.path.isdir(plotdir)):
            die('clustering plot directory exists; use force option')
    if options.force:
        if (os.path.isdir(plotdir)):
            shutil.rmtree(plotdir)
    os.mkdir(plotdir)

    simmat_temporal_fn = os.path.join(conf.project_dir, 'simmat_temporal.npy')
    simmat_clustered_fn = os.path.join(conf.project_dir,
                                       'simmat_clustered.npy')
    if not os.path.isfile(simmat_temporal_fn):
        die('similarity matrix does not exists: %s; ' +
            'use seiscloud matrix first' % simmat_temporal_fn)
    if not os.path.isfile(simmat_clustered_fn):
        die('similarity matrix does not exists: %s; ' +
            'use seiscloud matrix first' % simmat_clustered_fn)

    new_catalog_fn = os.path.join(conf.project_dir,
                                  'clustering_results',
                                  'clustered_events.pf')
#                                  'events_to_be_clustered.pf')
    if not os.path.isfile(new_catalog_fn):
        die('catalog of selected events does not exists: %s; ' +
            'use seiscloud matrix and seiscloud cluster first'
            % new_catalog_fn)

    events = model.load_events(new_catalog_fn)
    eventsclusters = sccluster.load_obj(
                     os.path.join(resdir, 'processed.eventsclusters'))
    clusters = sccluster.load_obj(
                     os.path.join(resdir, 'processed.clusters'))

    scplot.plot_all(events, eventsclusters, clusters, conf, resdir, plotdir)

    print('Seiscloud plots prepared in "%s"' % plotdir)


def main():
    if len(sys.argv) < 2:
        die('Usage: %s' % usage)

    args = list(sys.argv)
    args.pop(0)
    command = args.pop(0)

    if command in subcommands:
        globals()['command_' + d2u(command)](args)

    elif command in ('--help', '-h', 'help'):
        if command == 'help' and args:
            acommand = args[0]
            if acommand in subcommands:
                globals()['command_' + acommand](['--help'])

        die('Usage: %s' % usage)

    else:
        die('no such subcommand: %s' % command)
