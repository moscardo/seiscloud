import sys
from pyrocko.guts import Object, Float, Int, String, Bool


acceptable_metrics = ['kagan_angle', 'mt_l1norm', 'mt_l2norm', 'mt_cos',
                      'mt_weighted_cos', 'principal_axis',
                      'hypocentral', 'epicentral', 'temporal',
                      'magnitude']

acceptable_mt_based_metrics = ['kagan_angle', 'mt_l1norm', 'mt_l2norm',
                               'mt_cos', 'mt_weighted_cos', 'principal_axis']

acceptable_sim_mat_types = ['ascii', 'binary']

acceptable_figure_formats = ['png', 'pdf']

acceptable_catalog_origins = ['globalcmt', 'geofon', 'file']


class SeiscloudConfig(Object):

    project_dir = String.T(
                    help='Project directory',
                    default='./seiscloud_example_project')
    catalog_origin = String.T(
                    help='Origin of seismic catalog (globalcmt, geofon, file)',
                    default='globalcmt')
    catalog_fn = String.T(
                    help='Event file name',
                    default='none.pf')
    dbscan_eps = Float.T(
                    help='Epsilon parameter of DBSCAN', default=0.10)
    dbscan_nmin = Int.T(
                    help='Nmin parameter of DBSCAN', default=20)
    sw_simmat = Bool.T(
                    help='Switch to use a precalculated similarity matrix',
                    default=False)
    metric = String.T(
                    help='Metric definition', default='kagan_angle')
    sim_mat_fn = String.T(
                    help='Name of the similarity file name',
                    default='simmat.npy')
    sim_mat_type = String.T(
                    help='Name of the similarity file name', default='binary')
    weight_mnn = Float.T(
                    help='Weight of Mnn component in the distance computation',
                    default=1.0)
    weight_mee = Float.T(
                    help='Weight of Mee component in the distance computation',
                    default=1.0)
    weight_mdd = Float.T(
                    help='Weight of Mdd component in the distance computation',
                    default=1.0)
    weight_mne = Float.T(
                    help='Weight of Mne component in the distance computation',
                    default=1.0)
    weight_mnd = Float.T(
                    help='Weight of Mnd component in the distance computation',
                    default=1.0)
    weight_med = Float.T(
                    help='Weight of Med component in the distance computation',
                    default=1.0)
    tmin = String.T(
                    help='Minimum time', default='2000-01-01 00:00:00.000')
    tmax = String.T(
                    help='Maximum time', default='2018-01-01 00:00:00.000')
    time_window = Float.T(
                    help='Window length (days) for time evolution of clusters',
                    default=10.)
    time_window = String.T(
                    help='Window length (days) for time evolution of clusters',
                    default=10.)
    latmin = Float.T(
                    help='Minimum longitude (deg)', default=-25.)
    latmax = Float.T(
                    help='Maximum longitude (deg)', default=-18.)
    lonmin = Float.T(
                    help='Minimum longitude (deg)', default=-72.)
    lonmax = Float.T(
                    help='Maximum longitude (deg)', default=-67.)
    depthmin = Float.T(
                    help='Minimum depth (m)', default=0.)
    depthmax = Float.T(
                    help='Maximum depth (m)', default=700000.)
    magmin = Float.T(
                    help='Minimum magnitude', default=0.)
    magmax = Float.T(
                    help='Maximum magnitude', default=10.)
    euclidean_max = Float.T(
                    help='Maximum considered spatial distance',
                    default=500000.)
    intertime_max = Float.T(
                    help='Maximum considered time based distance (s)',
                    default=31536000.)
    magnitude_max = Float.T(
                    help='Maximum considered magnitude based distance',
                    default=10.)
    figure_format = String.T(
                    help='Format of output figures (pdf|png)',
                    default='png')
    sw_manual_radius = Bool.T(
                    help='Choose manually the map plot radius', default=False)
    map_radius = Float.T(
                    help='Radius of map plots (m); if None, it is calculated',
                    default=500000.)
    sw_filterevent = Bool.T(
                    help='Filter events for plotting', default=False)
#    sw_findcenters = Bool.T(
#                    help='Maximum considered spatial distance', default=False)
#    sw_cumulus = Bool.T(
#                    help='Maximum considered spatial distance', default=False)
#    sw_include_edge = Bool.T(
#                    help='Include edges of the clusters', default=False)


def generate_default_config():
    config = SeiscloudConfig()
    return config


def check(conf):
    if conf.metric not in acceptable_metrics:
        print('metric not acceptable: %s' % conf.metric)
        sys.exit()
    if conf.sim_mat_type not in acceptable_sim_mat_types:
        print('similarity matrix type not acceptable: %s' % conf.sim_mat_type)
        sys.exit()
    if conf.figure_format not in acceptable_figure_formats:
        print('figure format not acceptable: %s' % conf.figure_format)
        sys.exit()
    if conf.catalog_origin not in acceptable_catalog_origins:
        print('catalog origin not acceptable: %s' % conf.catalog_origin)
        sys.exit()
