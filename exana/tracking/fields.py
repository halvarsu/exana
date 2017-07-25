import numpy as np
import quantities as pq

def spatial_rate_map(x, y, t, sptr, binsize=0.01*pq.m, box_xlen=1*pq.m,
                     box_ylen=1*pq.m, mask_unvisited=True, convolve=True,
                     return_bins=False, smoothing=0.02):
    """Divide a 2D space in bins of size binsize**2, count the number of spikes
    in each bin and divide by the time spent in respective bins. The map can
    then be convolved with a gaussian kernel of size csize determined by the
    smoothing factor, binsize and box_xlen.

    Parameters
    ----------
    sptr : neo.SpikeTrain
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    t : quantities.Quantity array in s
        1d vector of times at x, y positions
    binsize : float
        spatial binsize
    box_xlen : quantities scalar in m
        side length of quadratic box
    mask_unvisited: bool
        mask bins which has not been visited by nans
    convolve : bool
        convolve the rate map with a 2D Gaussian kernel

    Returns
    -------
    out : rate map
    if return_bins = True
    out : rate map, xbins, ybins
    """
    from exana.misc.tools import is_quantities
    if not all([len(var) == len(var2) for var in [x,y,t] for var2 in [x,y,t]]):
        raise ValueError('x, y, t must have same number of elements')
    if box_xlen < x.max() or box_ylen < y.max():
        raise ValueError('box length must be larger or equal to max path length')
    from decimal import Decimal as dec
    decimals = 1e10
    remainderx = dec(float(box_xlen)*decimals) % dec(float(binsize)*decimals)
    remaindery = dec(float(box_ylen)*decimals) % dec(float(binsize)*decimals)
    if remainderx != 0 or remaindery != 0:
        raise ValueError('the remainder should be zero i.e. the ' +
                                     'box length should be an exact multiple ' +
                                     'of the binsize')
    is_quantities([x, y, t], 'vector')
    is_quantities(binsize, 'scalar')
    t = t.rescale('s')
    box_xlen = box_xlen.rescale('m').magnitude
    box_ylen = box_ylen.rescale('m').magnitude
    binsize = binsize.rescale('m').magnitude
    x = x.rescale('m').magnitude
    y = y.rescale('m').magnitude

    # interpolate one extra timepoint
    t_ = np.array(t.tolist() + [t.max() + np.median(np.diff(t))]) * pq.s
    spikes_in_bin, _ = np.histogram(sptr.times, t_)
    time_in_bin = np.diff(t_.magnitude)
    xbins = np.arange(0, box_xlen + binsize, binsize)
    ybins = np.arange(0, box_ylen + binsize, binsize)
    ix = np.digitize(x, xbins, right=True)
    iy = np.digitize(y, ybins, right=True)
    spike_pos = np.zeros((xbins.size, ybins.size))
    time_pos = np.zeros((xbins.size, ybins.size))
    for n in range(len(x)):
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
        csize = (box_xlen / binsize) * smoothing
        kernel = Gaussian2DKernel(csize)
        rate = convolve_fft(rate, kernel)  # TODO edge correction
    if mask_unvisited:
        was_in_bin = np.asarray(time_pos, dtype=bool)
        rate[np.invert(was_in_bin)] = np.nan
    if return_bins:
        return rate.T, xbins, ybins
    else:
        return rate.T



def gridness(rate_map, box_xlen, box_ylen, return_acorr=False,
             step_size=0.1*pq.m):
    '''Calculates gridness of a rate map. Calculates the normalized
    autocorrelation (A) of a rate map B where A is given as
    A = 1/n\Sum_{x,y}(B - \bar{B})^{2}/\sigma_{B}^{2}. Further, the Pearsson's
    product-moment correlation coefficients is calculated between A and A_{rot}
    rotated 30 and 60 degrees. Finally the gridness is calculated as the
    difference between the minimum of coefficients at 60 degrees and the
    maximum of coefficients at 30 degrees i.e. gridness = min(r60) - max(r30).
    In order to focus the analysis on symmetry of A the the central and the
    outer part of the gridness is maximized by increasingly mask A at steps of
    ``step_size``. This function is inspired by Lukas Solankas gridcells
    package from Matt Nolans lab.

    Parameters
    ----------
    rate_map : numpy.ndarray
    box_xlen : quantities scalar in m
        side length of quadratic box
    step_size : quantities scalar in m
        step size in masking
    return_acorr : bool
        return autocorrelation map or not

    Returns
    -------
    out : gridness, (autocorrelation map)
    '''
    from scipy.ndimage.interpolation import rotate
    import numpy.ma as ma
    from exana.misc.tools import (is_quantities, fftcorrelate2d,
                                            masked_corrcoef2d)
    is_quantities([box_xlen, box_ylen, step_size], 'scalar')
    box_xlen = box_xlen.rescale('m').magnitude
    box_ylen = box_ylen.rescale('m').magnitude
    step_size = step_size.rescale('m').magnitude
    tmp_map = rate_map.copy()
    tmp_map[~np.isfinite(tmp_map)] = 0
    acorr = fftcorrelate2d(tmp_map, tmp_map, mode='full', normalize=True)
    rows, cols = acorr.shape
    b_x = np.linspace(-box_xlen/2., box_xlen/2., rows)
    b_y = np.linspace(-box_ylen/2., box_ylen/2., cols)
    B_x, B_y = np.meshgrid(b_x, b_y)
    grids = []
    acorrs = []
    # TODO find size of middle gaussian and exclude
    for outer in np.arange(box_xlen/4, box_xlen/2, step_size):
        m_acorr = ma.masked_array(acorr, mask=np.sqrt(B_x**2 + B_y**2) > outer)
        for inner in np.arange(0, box_xlen/4, step_size):
            m_acorr = \
                ma.masked_array(m_acorr, mask=np.sqrt(B_x**2 + B_y**2) < inner)
            angles = range(30, 180+30, 30)
            corr = []
            # Rotate and compute correlation coefficient
            for angle in angles:
                rot_acorr = rotate(m_acorr, angle, reshape=False)
                corr.append(masked_corrcoef2d(rot_acorr, m_acorr)[0, 1])
            r60 = corr[1::2]
            r30 = corr[::2]
            grids.append(np.min(r60) - np.max(r30))
            acorrs.append(m_acorr)
    if return_acorr:
        return max(grids), acorr,  # acorrs[grids.index(max(grids))]
    else:
        return max(grids)


def occupancy_map(x, y, t,
                  binsize=0.01*pq.m,
                  box_xlen=1*pq.m,
                  box_ylen=1*pq.m,
                  convolve=True,
                  return_bins=False,
                  smoothing=0.02):
    '''Divide a 2D space in bins of size binsize**2, count the time spent
    in each bin. The map can  be convolved with a gaussian kernel of size
    csize determined by the smoothing factor, binsize and box_xlen.

    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    t : quantities.Quantity array in s
        1d vector of times at x, y positions
    binsize : float
        spatial binsize
    box_xlen : quantities scalar in m
        side length of quadratic box
    mask_unvisited: bool
        mask bins which has not been visited by nans
    convolve : bool
        convolve the rate map with a 2D Gaussian kernel


    Returns
    -------
    occupancy_map : numpy.ndarray
    if return_bins = True
    out : occupancy_map, xbins, ybins
    '''

    from exana.misc.tools import is_quantities
    if not all([len(var) == len(var2) for var in [
            x, y, t] for var2 in [x, y, t]]):
        raise ValueError('x, y, t must have same number of elements')
    if box_xlen < x.max() or box_ylen < y.max():
        raise ValueError(
            'box length must be larger or equal to max path length')
    from decimal import Decimal as dec
    decimals = 1e10
    remainderx = dec(float(box_xlen)*decimals) % dec(float(binsize)*decimals)
    remaindery = dec(float(box_ylen)*decimals) % dec(float(binsize)*decimals)
    if remainderx != 0 or remaindery != 0:
        raise ValueError('the remainder should be zero i.e. the ' +
                         'box length should be an exact multiple ' +
                         'of the binsize')
    is_quantities([x, y, t], 'vector')
    is_quantities(binsize, 'scalar')
    t = t.rescale('s')
    box_xlen = box_xlen.rescale('m').magnitude
    box_ylen = box_ylen.rescale('m').magnitude
    binsize = binsize.rescale('m').magnitude
    x = x.rescale('m').magnitude
    y = y.rescale('m').magnitude

    # interpolate one extra timepoint
    t_ = np.array(t.tolist() + [t.max() + np.median(np.diff(t))]) * pq.s
    time_in_bin = np.diff(t_.magnitude)
    xbins = np.arange(0, box_xlen + binsize, binsize)
    ybins = np.arange(0, box_ylen + binsize, binsize)
    ix = np.digitize(x, xbins, right=True)
    iy = np.digitize(y, ybins, right=True)
    time_pos = np.zeros((xbins.size, ybins.size))
    for n in range(len(x)):
        time_pos[ix[n], iy[n]] += time_in_bin[n]
    # correct for shifting of map since digitize returns values at right edges
    time_pos = time_pos[1:, 1:]
    if convolve:
        from astropy.convolution import Gaussian2DKernel, convolve_fft
        csize = (box_xlen / binsize) * smoothing
        kernel = Gaussian2DKernel(csize)
        time_pos = convolve_fft(time_pos, kernel)  # TODO edge correction
    if return_bins:
        return time_pos.T, xbins, ybins
    else:
        return time_pos.T


def nvisits_map(x, y, t,
                binsize=0.01*pq.m,
                box_xlen=1*pq.m,
                box_ylen=1*pq.m,
                return_bins=False):
    '''Divide a 2D space in bins of size binsize**2, count the
    number of visits in each bin. The map can  be convolved with
    a gaussian kernel of size  determined by the smoothing factor,
    binsize and box_xlen.

    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    t : quantities.Quantity array in s
        1d vector of times at x, y positions
    binsize : float
        spatial binsize
    box_xlen : quantities scalar in m
        side length of quadratic box


    Returns
    -------
    nvisits_map : numpy.ndarray
    if return_bins = True
    out : nvisits_map, xbins, ybins
    '''

    from exana.misc.tools import is_quantities
    if not all([len(var) == len(var2) for var in [
            x, y, t] for var2 in [x, y, t]]):
        raise ValueError('x, y, t must have same number of elements')
    if box_xlen < x.max() or box_ylen < y.max():
        raise ValueError(
            'box length must be larger or equal to max path length')
    from decimal import Decimal as dec
    decimals = 1e10
    remainderx = dec(float(box_xlen)*decimals) % dec(float(binsize)*decimals)
    remaindery = dec(float(box_ylen)*decimals) % dec(float(binsize)*decimals)
    if remainderx != 0 or remaindery != 0:
        raise ValueError('the remainder should be zero i.e. the ' +
                         'box length should be an exact multiple ' +
                         'of the binsize')
    is_quantities([x, y, t], 'vector')
    is_quantities(binsize, 'scalar')
    t = t.rescale('s')
    box_xlen = box_xlen.rescale('m').magnitude
    box_ylen = box_ylen.rescale('m').magnitude
    binsize = binsize.rescale('m').magnitude
    x = x.rescale('m').magnitude
    y = y.rescale('m').magnitude

    xbins = np.arange(0, box_xlen + binsize, binsize)
    ybins = np.arange(0, box_ylen + binsize, binsize)
    ix = np.digitize(x, xbins, right=True)
    iy = np.digitize(y, ybins, right=True)
    nvisits_map = np.zeros((xbins.size, ybins.size))
    for n in range(len(x)):
        if n == 0:
            nvisits_map[ix[n], iy[n]] = 1
        else:
            if ix[n-1] != ix[n] or iy[n-1] != iy[n]:
                nvisits_map[ix[n], iy[n]] += 1
    # correct for shifting of map since digitize returns values at right edges
    nvisits_map = nvisits_map[1:, 1:]
    if return_bins:
        return nvisits_map.T, xbins, ybins
    else:
        return nvisits_map.T



def spatial_rate_map_1d(x, t, sptr,
                        binsize=0.01*pq.m,
                        track_len=1*pq.m,
                        mask_unvisited=True,
                        convolve=True,
                        return_bins=False,
                        smoothing=0.02):
    """Take x coordinates of linear track data, divide in bins of binsize,
    count the number of spikes  in each bin and  divide by the time spent in
    respective bins. The map can then be convolved with a gaussian kernel of
    size csize determined by the smoothing factor, binsize and box_xlen.

    Parameters
    ----------
    sptr : neo.SpikeTrain
    x : quantities.Quantity array in m
        1d vector of x positions
    t : quantities.Quantity array in s
        1d vector of times at x, y positions
    binsize : float
        spatial binsize
    box_xlen : quantities scalar in m
        side length of quadratic box
    mask_unvisited: bool
        mask bins which has not been visited by nans
    convolve : bool
        convolve the rate map with a 2D Gaussian kernel

    Returns
    -------
    out : rate map
    if return_bins = True
    out : rate map, xbins
    """
    from exana.misc.tools import is_quantities
    if not all([len(var) == len(var2) for var in [x, t] for var2 in [x, t]]):
        raise ValueError('x, t must have same number of elements')
    if track_len < x.max():
        raise ValueError('track length must be\
        larger or equal to max path length')
    from decimal import Decimal as dec
    decimals = 1e10
    remainderx = dec(float(track_len)*decimals) % dec(float(binsize)*decimals)
    if remainderx != 0:
        raise ValueError('the remainder should be zero i.e. the ' +
                         'box length should be an exact multiple ' +
                         'of the binsize')
    is_quantities([x, t], 'vector')
    is_quantities(binsize, 'scalar')
    t = t.rescale('s')
    track_len = track_len.rescale('m').magnitude
    binsize = binsize.rescale('m').magnitude
    x = x.rescale('m').magnitude
    # interpolate one extra timepoint
    t_ = np.array(t.tolist() + [t.max() + np.median(np.diff(t))]) * pq.s
    spikes_in_bin, _ = np.histogram(sptr.times, t_)
    time_in_bin = np.diff(t_.magnitude)
    xbins = np.arange(0, track_len + binsize, binsize)
    ix = np.digitize(x, xbins, right=True)
    spike_pos = np.zeros(xbins.size)
    time_pos = np.zeros(xbins.size)
    for n in range(len(x)):
        spike_pos[ix[n]] += spikes_in_bin[n]
        time_pos[ix[n]] += time_in_bin[n]
    # correct for shifting of map since digitize returns values at right edges
    spike_pos = spike_pos[1:]
    time_pos = time_pos[1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.divide(spike_pos, time_pos)
    if convolve:
        rate[np.isnan(rate)] = 0.  # for convolution
        from astropy.convolution import Gaussian2DKernel, convolve_fft
        csize = (track_len / binsize) * smoothing
        kernel = Gaussian2DKernel(csize)
        rate = convolve_fft(rate, kernel)  # TODO edge correction
    if mask_unvisited:
        was_in_bin = np.asarray(time_pos, dtype=bool)
        rate[np.invert(was_in_bin)] = np.nan
    if return_bins:
        return rate.T, xbins
    else:
        return rate.T


def calculate_grid_geometry(rate_map, plot_fields=False):
    """Calculates quantitative information about grid field.
    Find bump centers, bump spacing, center diplacement and hexagon
    orientation

    Parameters
    ----------
    rate_map : np 2d array
               firing rate in each bin
    plot_fields : if True, plots the field labels with field centers to the
                  current matplotlib ax. Default False

    Returns
    -------
    bump_centers : 2d np.array
                   x,y positions of bump centers
    avg_dist : float
               average spacing between bumps, based on a hex-fit

    displacement : float
                   distance of bump closest to the center
    orientation : float
                  orientation of hexagon (in degrees)

    Examples
    --------

    """

    from scipy.ndimage import mean, center_of_mass

    # TODO: smooth data?
    # smooth_rate_map = lambda x:x
    # rate_map = smooth_rate_map(rate_map)


    fields, nfields = separate_fields(rate_map)
    # indices starts at 1
    indices = range(1,nfields+1)

    bump_centers = center_of_mass(rate_map, labels=fields, index=indices)

    if plot_fields:
        import matplotlib.pyplot as plt
        ax = plt.gca()
        plt.axis([0,1,0,1])
        plt.pcolormesh(fields)

    # normalize to (0,1)
    bump_centers = (np.array(bump_centers)/rate_map.shape)[:,::-1]

    avg_dist = find_avg_dist(rate_map)

    displacement, orientation = fit_hex(bump_centers, avg_dist, plot_bumps=plot_fields)
    return bump_centers, avg_dist, displacement, orientation


def separate_fields(rate_map, thrsh = 0):
    """Separates fields using the laplacian to identify fields separated by
    a negative second derivative.

    Parameters
    ----------
    rate_map

    thrsh : float
        upper cutoff of laplacian to separate fields by. Should be <= 0.
        (positive laplacian corresponds to dip in rate_map). Default 0.

    Returns
    -------
    fields : numpy array, shape like rate_map.
        contains areas all filled with same value, corresponding to fields
        in rate_map. The values are in range(1,nFields + 1), sorted by size of the
        field (sum of all field values). 0 elsewhere.

    n_field : int
        field count
    """
    from scipy.ndimage import laplace, label

    l = laplace(rate_map)
    l[l>thrsh] = 0

    # Labels areas of the laplacian not connected by values > 0.
    fields, n_fields = label(l)

    return fields, n_fields


def find_avg_dist(rate_map, thrsh = 0):
    """Uses autocorrelation and separate_fields to find average distance
    between bumps. """

    from scipy.ndimage import center_of_mass
    from exana.misc.tools import fftcorrelate2d

    # autocorrelate. Returns array (2x - 1) the size
    acorr = fftcorrelate2d(rate_map,rate_map)#, mode = 'full', normalize = True)

    #acorr[acorr<0] = 0
    f, nf = separate_fields(acorr,thrsh=thrsh) # TODO Find a valid value for thrsh, or 
                                               # remove. This is a big problem

    # index starts at 1!
    fpos = center_of_mass(acorr, labels=f, index=range(1,nf+1))
    fpos = np.array(fpos)

    # find dists from center in (rate_map-)relative units (from 0 to 2)
    distances = np.linalg.norm(fpos/rate_map.shape - (1,1), axis = 1)
    dist_sort = np.argsort(distances)
    distances = distances[dist_sort]

    # use maximum 6 closest values except center value
    avg_dist = np.average(distances[1:7])
    return avg_dist



def fit_hex(bump_centers, avg_dist=None, plot_bumps = False, method='best'):
    """Fits a hex grid to a given set of bumps. Uses the three bumps most


    Parameters
    ----------
    bump_centers : Nx2 np.array
                x,y positions of bump centers, x,y /in (0,1)

    avg_dist (optional): float
                average spacing between bumps

    plot_bumps (optional): bool
                if True, plots at the three bumps most likely to be in
                correct hex-position to the current matplotlib axes.

    method (optional): string, valid options: ['closest', 'best']
                method to find angle from neighboring bumps.
                'closest' uses six bumps nearest to center bump
                'best' uses the two bumps nearest to avg_dist

    Returns
    -------
    displacement : float
                   distance of bump closest to the center in meters
    orientation : float
                  orientation of hexagon (in degrees)
    """

    valid_methods = ['closest', 'best']
    if method not in valid_methods:
        msg = "Acceptable method flags are 'closest' or 'best'"
        raise(ValueError, msg)
    bump_centers = np.array(bump_centers)

    # sort by distance to center
    d = np.linalg.norm(bump_centers - (0.5,0.5), axis=1)
    d_sort = np.argsort(d)
    dist_sorted = bump_centers[d_sort]
    center_bump = dist_sorted[0]; others = dist_sorted[1:]

    displacement = d[d_sort][0]

    # others distances to center bumps
    relpos = others - center_bump
    reldist = np.linalg.norm(relpos, axis=1)

    if method == 'closest':
        # get 6 closest bumps
        rel_sort = np.argsort(reldist)
        closest = others[rel_sort][:6]
        relpos = relpos[rel_sort][:6]
    elif method == 'best':
        # get 2 bumps such that /sum_{i\neqj}(\abs{r_i-r_j}-avg_ist)^2 is minimized 
        squares = 1e32*np.ones((others.shape[0], others.shape[0]))
        
        for i in range(len(relpos)):
            for j in range(i,len(relpos)):
                rel1 = (reldist[i] - avg_dist)**2
                rel2 = (reldist[j] - avg_dist)**2
                rel3 = (np.linalg.norm(relpos[i]-relpos[j]) - avg_dist)**2
                squares[i,j] = rel1 + rel2 + rel3
        rel_slice = np.unravel_index(np.argmin(squares), squares.shape)
        rel_slice = np.array(rel_slice)
        #rel_sort = np.argsort(np.abs(reldist-avg_dist))
        closest = others[rel_slice]
        relpos = relpos[rel_slice]

    # sort by angle
    a = np.arctan2(relpos[:,1], relpos[:,0])%(2*np.pi)
    a_sort = np.argsort(a)
    print (relpos)

    # extract lowest angle and convert to degrees
    orientation = a[a_sort][0] *180/np.pi

    if plot_bumps:
        import matplotlib.pyplot as plt
        ax=plt.gca()
        i = 1
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        print(xmin,xmax,ymin,ymax)
        dx = xmax-xmin; dy = ymax - ymin

        closest = closest[a_sort]
        
        edges = [center_bump] if method == 'best' else []
        edges += [c for c in closest]
        edges = np.array(edges)*(dx,dy) + (xmin, ymin)
        poly = plt.Polygon(edges, alpha=0.5,color='r')
        ax.add_artist(poly)
    return displacement, orientation

