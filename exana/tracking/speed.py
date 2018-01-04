import numpy as np
import quantities as pq
import neo
import elephant
# import exana.tracking as tr

def speed_correlation(speed,t,sptr, stddev = 90*pq.ms):
    """
    correlates instantaneous spike rate and rat velocity, using a method
    described in [1]

    [1]: 
    """

    import matplotlib.pyplot as plt
    from astropy.convolution import Gaussian1DKernel, convolve_fft 


    min_speed = 0.02*pq.m*pq.Hz
    max_speed = np.percentile(speed, 95 )

    mask = np.logical_and(speed < max_speed,speed > min_speed)
    speed = speed[mask]
    t_ = t[:-1][mask]

    dt = np.average(np.diff(t_))

    t_binsize = dt
    stddev = t_binsize / dt
    binned_spikes, _ = np.histogram(sptr, t_)
    kernel = Gaussian1DKernel(stddev = stddev) # ms
    instant_spike_rate = convolve_fft(binned_spikes, kernel)

    corr = np.corrcoef(speed[:-1], instant_spike_rate)
    return corr[0,1]

def speed_correlation2(speed, t, sptr,plot=False):
    sp_units = speed.units
    min_speed = 0.02*sp_units
    max_speed = np.percentile(speed, 95 )*pq.m*pq.Hz

    mask = np.logical_and(speed < max_speed,speed > min_speed)
    speed = speed[mask]
    t_ = t[:-1][mask]
    # speed = speed
    # t_ = t[:-1]

    binsize=0.03*sp_units
    max_speed = binsize * int(max_speed/binsize+ 1)
    #max_speed = 9*sp_units # *pq.m*pq.Hz # binsize*( + 1)*speed.units
    kernel_size = 0.04 * pq.m/pq.s
    smoothing = kernel_size / max_speed
    sp_spike_rate, bins = speed_rate_map_1d(speed, t_, sptr, return_bins=True, max_speed = max_speed,
            smoothing = smoothing, binsize = binsize)
    mask = np.isnan(sp_spike_rate)
    bins = bins[:-1][~mask]
    rate = sp_spike_rate[~mask]
    if plot:
        import matplotlib.pyplot as plt
        ax = plt.gca()
        ax.plot(bins,rate)
    corr2 = np.corrcoef(bins, rate)
    return corr2[0,1]

def speed_rate_map_1d(vel_x, t, sptr,
                        binsize=0.03*pq.m/pq.s,
                        max_speed=9*pq.m/pq.s,
                        mask_unvisited=True,
                        only_relevant_bins = True,
                        min_bin_coverage = 0.005,
                        convolve=True,
                        return_bins=False,
                        smoothing=None):
    """Take vel_x coordinates of linear track data, divide in bins of binsize,
    count the number of spikes  in each bin and  divide by the time spent in
    respective bins. The map can then be convolved with a gaussian kernel of
    size csize determined by the smoothing factor, binsize and box_xlen.

    Parameters
    ----------
    sptr : neo.SpikeTrain
    vel_x : quantities.Quantity array in m/s
        1d vector of x velocities
    t : quantities.Quantity array in s
        1d vector of times at vel_x, y positions
    binsize : float
        spatial binsize
    box_xlen : quantities scalar in m/s
        side length of quadratic box
    mask_unvisited: bool
        mask bins which has not been visited by nans
    convolve : bool
        convolve the rate map with a 2D Gaussian kernel
    only_relevant_bins : bool
        keep only bins accounting for min_bin_coverage*len(sptr) spikes
    min_bin_coverage : float
        minimum number of spikes in bin compared to total number of spikes
        for bin to be valid, if only_relevant_bins = True

    Returns
    -------
    out : rate map
    if return_bins = True
    out : rate map, xbins
    """
    from exana.misc.tools import is_quantities
    if not all([len(var) == len(var2) for var in [vel_x, t] for var2 in [vel_x, t]]):
        raise ValueError('vel_x, t must have same number of elements')
    if max_speed < vel_x.max():
        raise ValueError('track length must be\
        larger or equal to max path length')
    if (smoothing is None) and (convolve is True):
        smoothing = 0.04 / max_speed
    from decimal import Decimal as dec
    decimals = 1e10
    remainderx = dec(float(max_speed)*decimals) % dec(float(binsize)*decimals)
    if remainderx != 0:
        raise ValueError('the remainder should be zero i.e. the ' +
                         'box length should be an exact multiple ' +
                         'of the binsize')
    is_quantities([vel_x, t], 'vector')
    is_quantities(binsize, 'scalar')
    t = t.rescale('s')
    max_speed = max_speed.rescale('m/s').magnitude
    binsize = binsize.rescale('m/s').magnitude
    vel_x = vel_x.rescale('m/s').magnitude
    # interpolate one extra timepoint
    t_ = np.array(t.tolist() + [t.max() + np.median(np.diff(t))]) * pq.s
    spikes_in_bin, _ = np.histogram(sptr.times, t_)
    time_in_bin = np.diff(t_.magnitude)
    xbins = np.arange(0, max_speed + binsize, binsize)
    ix = np.digitize(vel_x, xbins, right=True)
    spike_pos = np.zeros(xbins.size)
    time_pos = np.zeros(xbins.size)
    for n in range(len(vel_x)):
        spike_pos[ix[n]] += spikes_in_bin[n]
        time_pos[ix[n]] += time_in_bin[n]
    # correct for shifting of map since digitize returns values at right edges
    spike_pos = spike_pos[1:]
    time_pos = time_pos[1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.divide(spike_pos, time_pos)
    if convolve:
        rate[np.isnan(rate)] = 0.  # for convolution
        from astropy.convolution import Gaussian1DKernel, convolve_fft
        csize = (max_speed / binsize) * smoothing
        kernel = Gaussian1DKernel(csize)
        rate = convolve_fft(rate, kernel)  # TODO edge correction
    if only_relevant_bins:
        valid_bin = np.asarray(spike_pos > min_bin_coverage*sptr.size, dtype=bool)
        rate[np.invert(valid_bin)] = np.nan
    if mask_unvisited:
        was_in_bin = np.asarray(time_pos, dtype=bool)
        rate[np.invert(was_in_bin)] = np.nan
    if return_bins:
        return rate.T, xbins
    else:
        return rate.T


def speed_rate_map(velx, vely, t, sptr, binsize=0.02*pq.m/pq.s, 
                max_xspeed=5*pq.m/pq.s, max_yspeed=1*pq.m/pq.s,
                mask_unvisited=True, convolve=True,
                return_bins=False, smoothing=0.03):
    """Divide a 2D space in bins of size binsize**2, count the number of spikes
    in each bin and divide by the time spent in respective bins. The map can
    then be convolved with a gaussian kernel of size csize determined by the
    smoothing factor, binsize and max_xspeed.

    Parameters
    ----------
    sptr : neo.SpikeTrain
    velx : quantities.Quantity array in m
        1d vector of x velocities
    vely : quantities.Quantity array in m
        1d vector of y velocities
    t : quantities.Quantity array in s
        1d vector of times at x, y velocities
    binsize : float
        spatial binsize
    max_xspeed : quantities scalar in m
        side length of quadratic box
    mask_unvisited: bool
        mask bins which has not been visited by nans
    convolve : bool
        convolve the rate map with a 2D Gaussian kernel

    Returns
    ------n-
    out : rate map
    if return_bins = True
    out : rate map, xbins, ybins
    """
    from exana.misc.tools import is_quantities
    if not all([len(var) == len(var2) for var in [velx,vely,t] for var2 in [velx,vely,t]]):
        raise ValueError('velx, vely, t must have same number of elements')
    if max_xspeed < velx.max() or max_yspeed < vely.max():
        raise ValueError('box length must be larger or equal to max path length')
    from decimal import Decimal as dec
    decimals = 1e10
    remainderx = dec(float(max_xspeed)*decimals) % dec(float(binsize)*decimals)
    remaindery = dec(float(max_yspeed)*decimals) % dec(float(binsize)*decimals)
    if remainderx != 0 or remaindery != 0:
        raise ValueError('the remainder should be zero i.e. the ' +
                                     'box length should be an exact multiple ' +
                                     'of the binsize')
    is_quantities([velx, vely, t], 'vector')
    is_quantities(binsize, 'scalar')
    t = t.rescale('s')
    max_xspeed = max_xspeed.rescale('m/s').magnitude
    max_yspeed = max_yspeed.rescale('m/s').magnitude
    binsize = binsize.rescale('m/s').magnitude
    velx = velx.rescale('m/s').magnitude
    vely = vely.rescale('m/s').magnitude

    # interpolate one extra timepoint
    t_ = np.array(t.tolist() + [t.max() + np.median(np.diff(t))]) * pq.s
    spikes_in_bin, _ = np.histogram(sptr.times, t_)
    time_in_bin = np.diff(t_.magnitude)
    xbins = np.arange(0, max_xspeed + binsize, binsize)
    ybins = np.arange(0, max_yspeed + binsize, binsize)
    ix = np.digitize(velx, xbins, right=True)
    iy = np.digitize(vely, ybins, right=True)
    spike_pos = np.zeros((xbins.size, ybins.size))
    time_pos  = np.zeros((xbins.size, ybins.size))
    for n in range(len(velx)):
        spike_pos[ix[n], iy[n]] += spikes_in_bin[n]
        time_pos[ix[n], iy[n]] += time_in_bin[n]
    # correct for shifting of map since digitize returns values at right edges
    spike_pos = spike_pos[1:, 1:]
    time_pos = time_pos[1:, 1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.divide(spike_pos, time_pos)
    if convolve:
        rate[np.isnan(rate)] = 0.  # for convolution
        from astropy.convolution import Gaussian2DKernel, convolve_fft
        csize = (max_xspeed / binsize) * smoothing
        kernel = Gaussian2DKernel(csize)
        rate = convolve_fft(rate, kernel)  # TODO edge correction
    if mask_unvisited:
        was_in_bin = np.asarray(time_pos, dtype=bool)
        rate[np.invert(was_in_bin)] = np.nan
    if return_bins:
        return rate.T, xbins, ybins
    else:
        return rate.T
