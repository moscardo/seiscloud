## Seiscloud
Software for seismicity clustering.

* Computation of similarity matrices upon different metrics
* Clustering using DBSCAN algorithm
* Graphical output of cluster features

![seiscloud logo](https://github.com/moscardo/seiscloud/tree/master/src/seiscloud_logo.png)

Please contact me for further description and help: simone.cesca@gfz-potsdam.de

## Prerequisites

* [pyrocko](https://pyrocko.org)
* [GMT, version 5](https://www.soest.hawaii.edu/gmt/) 

## Download and Installation

    git clone https://github.com/moscardo/seiscloud.git
    cd seiscloud
    sudo python3 setup.py install

## Processing
If you need help add a ``--help`` to the command call in order to get additional information.

If clustering steps need to be repeated use the ``--force`` option to overwrite previous results.

At first you need a configuration file for seiscloud.
To create an example configuration file:

    seiscloud example

Adapt the configuration file to your needs or build your own configuration file.
The next step is to initialize your project:

    seiscloud init <configuration_file>

The previous command will create a project directory and store there some important information (e.g. the seismic catalog).
If a similarity matrix is already available this will also be stored in the project directory.
Otherwise the similarity matrix can be computed according to the metric chosen in the configuration time (e.g. similarity in location, origin time, focal mechanism, moment tensor, ...):

    seiscloud matrix <configuration_file>

Now, run the clustering:

    seiscloud cluster <configuration_file>

Results, in the form of subcatalogs for each cluster, are stored as ascii file (pyrocko format) in the subdirectory clustering_results of the project directory.

And finally produce plots:

    seiscloud plot <configuration_file>

Figures illustrative of the clustering results are stored as png or pdf files in the subdirectory clustering_plots of the project directory.
