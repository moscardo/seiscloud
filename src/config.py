from pyrocko.guts import Object, Float, Int, String, Bool, List, Tuple, Dict
import os


acceptable_metrics=['kagan_angle','mt_l1norm','mt_l2norm','mt_cos',\
                    'mt_weighted_cos','principal_axis',\
                    'hypocentral','epicentral','temporal']

acceptable_mt_based_metrics=['kagan_angle','mt_l1norm','mt_l2norm','mt_cos',\
                    'mt_weighted_cos','principal_axis']

#acceptable_sortings=['size','north','east','down','dc','clvd','iso']

acceptable_sim_mat_types=['ascii','binary']

acceptable_figure_formats=['png','pdf']


class SeiscloudConfig(Object):

    project_dir     = String.T(
                    help='Project directory',default='./seiscloud_example_project')
    catalog_fn      = String.T(
                    help='Event file name',default='../examples/globalcmt.nchile.pf')
    dbscan_eps      = Float.T(
                    help='Epsilon parameter of DBSCAN',default=0.10)
    dbscan_nmin     = Int.T(
                    help='Nmin parameter of DBSCAN',default=20)
    sw_simmat       = Bool.T(
                    help='Switch to use a precalculated similarity matrix',default=False)
    metric          = String.T(
                    help='Metric definition',default='kagan_angle')
    sim_mat_fn      = String.T(
                    help='Name of the similarity file name',default='simmat.npy')
    sim_mat_type    = String.T(
                    help='Name of the similarity file name',default='binary')
    weight_mnn      = Float.T(
                    help='Weight of the Mnn component in the distance computation',default=1.0)
    weight_mee      = Float.T(
                    help='Weight of the Mee component in the distance computation',default=1.0)
    weight_mdd      = Float.T(
                    help='Weight of the Mdd component in the distance computation',default=1.0)
    weight_mne      = Float.T(
                    help='Weight of the Mne component in the distance computation',default=1.0)
    weight_mnd      = Float.T(
                    help='Weight of the Mnd component in the distance computation',default=1.0)
    weight_med      = Float.T(
                    help='Weight of the Med component in the distance computation',default=1.0)
    tmin            = String.T(
                    help='Minimum time',default='2000-01-01 00:00:00.000')
    tmax            = String.T(
                    help='Maximum time',default='2018-01-01 00:00:00.000')
    time_window     = Float.T(
                    help='Window length (in days) for temporal evolution of clusters',default=10.)
    time_window     = String.T(
                    help='Window length (in days) for temporal evolution of clusters',default=10.)
    latmin          = Float.T(
                    help='Minimum longitude (deg)',default=0.)
    latmax          = Float.T(
                    help='Maximum longitude (deg)',default=1.)
    lonmin          = Float.T(
                    help='Minimum longitude (deg)',default=0.)
    lonmax          = Float.T(
                    help='Maximum longitude (deg)',default=1.)
    depthmin        = Float.T(
                    help='Minimum depth (m)',default=0.)
    depthmax        = Float.T(
                    help='Maximum depth (m)',default=700000.)
    magmin          = Float.T(
                    help='Minimum magnitude',default=0.)
    magmax          = Float.T(
                    help='Maximum magnitude',default=10.)
    euclidean_max   = Float.T(
                    help='Maximum considered spatial distance',default=1000000.)
    intertime_max   = Float.T(
                    help='Maximum considered time based distance (s)',default=86400.)
    figure_format   = String.T(
                    help='Format of output figures (pdf|png)',default='png')
    radius_map_plot = Float.T(
                    help='Radius of map plots (m); if None, it is internally calculated' ,default=None)
#    cluster_sorting = String.T(
#                    help='Clustering sorting parameter',default='size')
    sw_findcenters  = Bool.T(
                    help='Maximum considered spatial distance',default=False)
    sw_filterevent  = Bool.T(
                    help='Maximum considered spatial distance',default=False)
    sw_cumulus      = Bool.T(
                    help='Maximum considered spatial distance',default=False)
    sw_include_edge = Bool.T(
                    help='Maximum considered spatial distance',default=False)



def generate_default_config():
    config = SeiscloudConfig()
    return config



def check(conf):
#    if conf.cluster_sorting not in acceptable_sortings:
#        print('cluster sorting not acceptable: %s' % conf.cluster_sorting)
#        sys.exit()
    if conf.metric not in acceptable_metrics:
        print('metric not acceptable: %s' % conf.metric)
        sys.exit()
    if conf.sim_mat_type not in acceptable_sim_mat_types:
        print('similarity matrix type not acceptable: %s' % conf.sim_mat_type)
        sys.exit()
    if conf.figure_format not in acceptable_figure_formats:
        print('figure format not acceptable: %s' % conf.figure_format)
        sys.exit()
