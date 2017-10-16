import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from ..misc.plot import simpleaxis


def plot_spike_histogram(trials, color='b', ax=None, binsize=None, bins=None,
                         output='counts', edgecolor='k', alpha=1., ylabel=None,
                         nbins=None):
    """
    Raster plot of trials

    Parameters
    ----------
    trials : list of neo.SpikeTrains
    color : str
        Color of histogram.
    edgecolor : str
        Color of histogram edges.
    ax : matplotlib axes
    output : str
        Accepts 'counts', 'rate' or 'mean'.
    binsize :
        Binsize of spike rate histogram, default None, if not None then
        bins are overridden.
    nbins : int
        Number of bins, defaults to 100 if binsize is None.
    ylabel : str
        The ylabel of the plot, if None defaults to output type.

    Examples
    --------
    >>> import neo
    >>> from numpy.random import rand
    >>> from exana.stimulus import make_spiketrain_trials
    >>> spike_train = neo.SpikeTrain(rand(1000) * 10, t_stop=10, units='s')
    >>> epoch = neo.Epoch(times=np.arange(0, 10, 1) * pq.s,
    ...                   durations=[.5] * 10 * pq.s)
    >>> trials = make_spiketrain_trials(spike_train, epoch)
    >>> ax = plot_spike_histogram(trials, color='r', edgecolor='b',
    ...                           binsize=1 * pq.ms, output='rate', alpha=.5)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import quantities as pq
        import neo
        from numpy.random import rand
        from exana.stimulus import make_spiketrain_trials
        from exana.statistics import plot_spike_histogram
        spike_train = neo.SpikeTrain(rand(1000) * 10, t_stop=10, units='s')
        epoch = neo.Epoch(times=np.arange(0, 10, 1) * pq.s, durations=[.5] * 10 * pq.s)
        trials = make_spiketrain_trials(spike_train, epoch)
        ax = plot_spike_histogram(trials, color='r', edgecolor='b', binsize=1 * pq.ms, output='rate', alpha=.5)
        plt.show()

    Returns
    -------
    out : axes
    """
    ### TODO
    if bins is not None:
        assert isinstance(bins, int)
        warnins.warn('The variable "bins" is deprecated, use nbins in stead.')
        nbins = bins
    ###
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    from elephant.statistics import time_histogram
    dim = trials[0].times.dimensionality
    t_start = trials[0].t_start.rescale(dim)
    t_stop = trials[0].t_stop.rescale(dim)
    if binsize is None:
        if nbins is None:
            nbins = 100
        binsize = (abs(t_start)+abs(t_stop))/float(nbins)
    else:
        binsize = binsize.rescale(dim)
    time_hist = time_histogram(trials, binsize, t_start=t_start,
                               t_stop=t_stop, output=output, binary=False)
    bs = np.arange(t_start.magnitude, t_stop.magnitude, binsize.magnitude)
    if ylabel is None:
        if output == 'counts':
            ax.set_ylabel('count')
        elif output == 'rate':
            time_hist = time_hist.rescale('Hz')
            if ylabel:
                ax.set_ylabel('rate [%s]' % time_hist.dimensionality)
        elif output == 'mean':
            ax.set_ylabel('mean count')
    elif isinstance(ylabel, str):
        ax.set_ylabel(ylabel)
    else:
        raise TypeError('ylabel must be str not "' + str(type(ylabel)) + '"')
    ax.bar(bs[0:len(time_hist)], time_hist.magnitude, width=bs[1]-bs[0],
           edgecolor=edgecolor, facecolor=color, alpha=alpha)
    return ax


def plot_isi_hist(sptr, alpha=1, ax=None, binsize=2*pq.ms,
                  time_limit=100*pq.ms, color='b'):
    """
    Bar plot of interspike interval (ISI) histogram

    Parameters
    ----------
    sptr : neo.SpikeTrain
    color : color of histogram
    ax : matplotlib axes
    alpha : opacity
    binsize : binsize of spike rate histogram, default 30 ms
    time_limit : end time of histogram x limit

    Returns
    -------
    out : axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    spk_isi = np.diff(sptr)*sptr.units
    binsize.units = 's'
    time_limit.units = 's'
    ax.hist(spk_isi, bins=np.arange(0., time_limit.magnitude,
            binsize.magnitude), normed=True, alpha=alpha, color=color)
    ax.set_xlabel('$Interspike\, interval\, \Delta t \,[ms]$')
    binsize.units = 'ms'
    ax.set_ylabel('$Proportion\, of\, intervals\, (%.f ms\, bins)$' % binsize)
    return ax


def plot_autocorr(sptr, title='', color='k', edgecolor='k', ax=None, **kwargs):
    par = {'corr_bin_width': 0.01*pq.s,
           'corr_limit': 1.*pq.s}
    if kwargs:
        par.update(kwargs)
    from .tools import correlogram
    if ax is None:
        fig, ax = plt.subplots()
    bin_width = par['corr_bin_width'].rescale('s').magnitude
    limit = par['corr_limit'].rescale('s').magnitude
    count, bins = correlogram(t1=sptr.times.magnitude, t2=None,
                              bin_width=bin_width, limit=limit,  auto=True)
    ax.bar(bins[:-1] + bin_width / 2., count, width=bin_width, color=color,
            edgecolor=edgecolor)
    ax.set_xlim([-limit, limit])
    ax.set_title(title)


def hist_spike_rate(sptr, ax, sigma):
    '''
    deprecated
    calculates spike rate histogram and plots to given axis
    '''
    nbins = sptr.max() / sigma
    ns, bs = np.histogram(sptr, nbins)
    ax.bar(bs[0:-1], ns/sigma, width=bs[1]-bs[0])
    ax.set_ylabel('spikes/s')
