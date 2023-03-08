from header import *

def amax2(x, *args, **kwargs):
    if 'key' not in kwargs:
        return np.amax(x,*args,**kwargs)
    else: 
        key = kwargs.pop('key') # e.g. len, pop so no TypeError: unexpected keyword
        x_key = np.vectorize(key)(x) # apply key to x element-wise
        axis = kwargs.get('axis') # either None or axis is set in kwargs
        if len(args)>=2: # axis is set in args
            axis = args[1]
        # The following is kept verbose, but could be made more efficient/shorter    
        if axis is None: # max of flattened
            max_flat_index = np.argmax(x_key, axis=axis)
            max_tuple_index = np.unravel_index(max_flat_index, x.shape)
            return x[max_tuple_index]
        elif axis == 0: # max in each column
            max_indices = np.argmax(x_key, axis=axis)
            return np.array(
                 [ x[max_i, i] # reorder for col
                     for i, max_i in enumerate(max_indices) ], 
                 dtype=x.dtype)
        elif axis == 1: # max in each row
            max_indices = np.argmax(x_key, axis=axis)
            return np.array(
                 [ x[i, max_i]
                     for i, max_i in enumerate(max_indices) ],
                 dtype=x.dtype)


def normalize_channels_minmax( rs ): 
    chan_max = np.amax( rs, axis = 1 )
    chan_min = np.amin( rs, axis = 1 )
    gap = chan_max - chan_min
    gap[ np.where( gap == 0 ) ] = np.finfo(np.float32).eps
    rs = (rs - chan_min[ :, np.newaxis ])/gap[ :, np.newaxis ]
    return( rs )

def normalize_array_minmax( ss ): 
    arr_max = np.amax( ss )
    arr_min = np.amin( ss )
    gap = arr_max - arr_min
    gap = gap if( gap != 0 ) else np.finfo( np.float32 ).eps
    ss = (ss - arr_min)/gap
    return( ss )

def normalize_cols_minmax( df, cols = None ): 
    if( cols is None ):
        df = (df - df.min())/(df.max() - df.min() )
    elif( isinstance( cols[ 0 ], int ) ):
        df.iloc[ :, cols ] = (df.iloc[ :, cols ] - df.iloc[ :, cols ].min())/(df.iloc[ :, cols ].max() - df.iloc[ :, cols ].min() )
    elif( isinstance( cols[ 0 ], str ) ):
        df.loc[ :, cols ] = (df.loc[ :, cols ] - df.loc[ :, cols ].min())/(df.loc[ :, cols ].max() - df.loc[ :, cols ].min() )
    return( df )

def normalize_channels_standard( rs ): 
    chan_mean = np.mean( rs, axis = 1 )
    chan_std = np.std( rs, axis = 1 )
    rs = (rs - chan_mean)/chan_std
    return( rs )

def normalize_array_standard( ss ): 
    arr_mean = np.mean( ss )
    arr_std = np.std( ss )
    rs = (ss - arr_mean)/arr_std
    return( ss )

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def np_resample_filter( 
    rs, 
    down_quality = 0, 
    up_quality = 0, 
    low_count = 0, 
    high_count = 0
):
    ## Quality:
    ## 0: bypass
    ## 1: Linear
    ## 2: Fourier
    
    if low_count == 0:
        low_count = len( rs )
    if high_count == 0:
        high_count = len( rs )
    input_sample_count = len( rs )
    
    ## Downsampling.
    if( down_quality == 0 ): 
        ds = rs
    elif( down_quality == 1 ):
        ds = np.interp( 
            np.arange( low_count ), 
            np.linspace( 0, low_count, input_sample_count ), 
            rs
        )
    elif( down_quality == 2 ):
        ds = rs.resample( 
            rs, 
            low_count
        )
    
    ## Upsampling.
    if( up_quality == 0 ): 
        if( len( ds ) < 2 ):
            ds = np.append( ds, ds[ 0 ] )
        res = interpolate.interp1d( 
            np.arange( len( ds ) ), 
            ds, 
            kind = 'nearest' 
        )( 
            np.linspace( 0, len( ds ) - 1 , high_count )
        )
    elif( up_quality == 1 ):
        res = np.interp( 
            np.arange( high_count ), 
            np.linspace( 0, high_count, low_count ), 
            ds
        )
    elif( up_quality == 2 ):
        res = signal.resample( 
            ds, 
            high_count
        )
    return( res )

def mat_row_resample( 
    mat, 
    down_quality = 0, 
    up_quality = 0, 
    low_count = 0, 
    high_count = 0
): 
    if( type( mat[ 0 ] ) != np.ndarray ):
        raise ValueError( "Error: input must be a 2D numpy array!" ) 
    for ii in range( mat.shape[ 0 ] ):
        mat[ ii ] = np_resample_filter(
            mat[ ii ], 
            down_quality, 
            up_quality, 
            low_count, 
            high_count
        )
    return( mat )

def interpolate_all_channels( 
    rs, 
    up_quality = 0, 
    high_count = 0
): 
    row_len = max( [len( tt ) for tt in rs] )
    high_count = max( row_len, high_count )
    mat_res = np.zeros( high_count ).reshape( (1,high_count) )
    for ii in range( len( rs ) ):
        if( type( rs[ ii ] ) != np.ndarray ):
            raise ValueError( "Error: list element must be numpy ndarray!" ) 
        temp = np_resample_filter(
                rs[ ii ], 
                up_quality = up_quality, 
                high_count = high_count
            )
        mat_res = np.vstack( (mat_res, temp) )
    return( mat_res[ 1: ] )

def wavedec_to_matrix( dec ):
    max_level = len( dec ) - 1
    dec[ 0 ] = np.repeat( dec[ 0 ], 2**(max_level - 1) ) ## same repeatation for approx as last detail. 
    dec[ 1: ] = [
        np.repeat( wt, 2**((max_level - 1) - lvl) )
        for lvl, wt in enumerate( dec[ 1: ] )
    ]
    mat = np.asanyarray( dec )
    return( mat )

def check_power_of_2( tt ):
    return( (tt & (tt - 1) == 0) and tt != 0 )

def dwt_matrix( 
    signal, 
    wavelet ="haar" 
):
    if( not check_power_of_2( len( signal ) ) ):
        raise ValueError( "Error: Signal length must be power of 2! Current length: " + str( len( signal ) ) )
    dec = pywt.wavedec( 
        signal, 
        wavelet = wavelet, 
        mode = 'smooth'
    ) 
    mat = wavedec_to_matrix( dec )
    return( mat )

def matrix_to_wavedec( mat ):
    max_level = mat.shape[ 0 ]
    dec = []
    dec.append( np.asanyarray( mat[ 0, 0::2**(max_level - 2) ] ) )
    dec[ 1: ] = [
        row[ 0::2**(((max_level - 1) - lvl) - 1) ]
        for lvl, row in enumerate( mat[ 1:, : ] )
    ]
    return( dec )

def reconstruct_from_wavedec_matrix( mat ):
#     dec_rebuild = [np.asanyarray( [0] )]
#     dec_rebuild.extend( matrix_to_wavedec( mat ) )
    dec_rebuild = matrix_to_wavedec( mat )
    signal_o = pywt.waverec( dec_rebuild, wavelet ="haar" )
    return( signal_o )

def max_effective_level_cap( wavelet, input_level, original_signal_len ):
    capped_max_level = min( 
        input_level, 
        pywt.dwt_max_level( 
            original_signal_len, 
            filter_len = wavelet 
        ) 
    )
    return( capped_max_level )

def compute_rec_signals_lengths( 
    original_signal_len, 
    wavelet = "haar",  
    max_level_cap = 99999 
):
    max_effective_level = max_effective_level_cap( wavelet, max_level_cap, original_signal_len )
    
    rec_coef_lengths = pywt.wavedec( 
        np.zeros( original_signal_len ), 
        wavelet = wavelet, 
        level = max_effective_level, 
        mode = 'smooth'
    ) 
    rec_coef_lengths = [tt.shape[ 0 ] for tt in rec_coef_lengths]
    
    ## Append the orignal signal legnth to the end. 
    rec_coef_lengths.append( original_signal_len )
    
    return( rec_coef_lengths )

## Use wavelet decomposition to compress the signal by cutting coefficients. 
def waveletCompressionCut( 
    original_signal, 
    max_samples, 
    wavelet = "haar",  
    max_level_cap = 9999999, 
#     level_jump, 
    include_original_signal = False
):
    max_effective_level = max_effective_level_cap( wavelet, max_level_cap, len( original_signal ) )
#     print( "Wavelet compressing with", max_effective_level, "detail levels, each capping to", max_samples, "sample points.")
        
    wavelet_coef = pywt.wavedec( 
        original_signal, 
        wavelet = wavelet, 
        level = max_effective_level, 
        mode = 'smooth'
    ) 

#     ## Handling level jump.
#     ## Pop out the approx signal then add back for easier effective level selection. 
#     approximation_signal = wavelet_coef.pop( 0 )
#     effective_levels = np.arange( 
#         start = max_effective_level - 1, 
#         stop = -1,
#         step = -(level_jump + 1)
#     )
#     wavelet_coef = [ wavelet_coef[ ii ] for ii in sorted( effective_levels, reverse = True ) ]
#     ## Add back the poped approx signal. 
#     wavelet_coef.append( approximation_signal ) # Include the approximation coefficients. 
#     ## Include the original signal. 
#     wavelet_coef.insert( 0, original_signal ) 
    
    # Check if max_samples is enough to cover the entire original signal on the coarsest level. 
    if( max_samples < len( wavelet_coef[ 0 ] ) ):
        raise ValueError( 
            """Error: Max sample count {} not enough to cover the length {} of the coarsest detail coefficient. 
                Fix options:
                    1. Use a larger sample count. 
                    2. Allow more levels, current max level: {}. 
                    2. Input a shorter signal.""".format( 
                max_samples, 
                len( wavelet_coef[ 0 ] ), 
                max_level_cap, 
            ) 
        )
        
    # Cap all compression coef length. 
    wavelet_coef_cut = [coeff[ -max_samples:] for coeff in wavelet_coef]
    
    if( include_original_signal ):
        ## Include the original signal.
        wavelet_coef_cut.append( original_signal[ -max_samples:] ) 
    
    return( wavelet_coef_cut )


def wavelet_decompression( 
    rec_coef, 
    rec_coef_lengths, 
    wavelet = "haar", 
    alignment = "right"
):
    # Exclude the capped original signal which is the last signal in the list.
    rec_coef_proc = rec_coef.copy() 

    if( len( rec_coef_lengths ) > len( rec_coef_proc ) ):
        rec_coef_lengths = rec_coef_lengths[ -(len( rec_coef_proc ) - 1): ] # Remove the irrelevant levels. 
        rec_coef_lengths.insert( 0, rec_coef_lengths[ 0 ] ) # Duplicate the first legnth for the approx coeff. 
    elif( len( rec_coef_lengths ) < len( rec_coef_proc ) ):
        raise ValueError( "List of rec_coef_length should be longer than the length of the coefficient list." )

    ## Zero padding. 
    for ii in range( len( rec_coef_proc ) ):
        if( rec_coef_lengths[ ii ] > len( rec_coef_proc[ ii ] ) ):
            # Pad the capped wavelet levels. 
            rec_coef_proc[ ii ] = ( 
                np.hstack( [
                    np.zeros( rec_coef_lengths[ ii ] - len( rec_coef_proc[ ii ] ) ), 
                    rec_coef_proc[ ii ]
                ] )
            )
    signal_rec = pywt.waverec( rec_coef_proc, wavelet )
#     signal_rec = signal.resample( signal_rec, rec_coef_lengths[ -1 ] )
    
    ## Align the start (lowest resolution) of the reconstructed signal to the original signal legnth. 
    signal_rec = signal_rec[ (len( signal_rec ) - rec_coef_lengths[ -1 ]): ]
    
    # ## Fill in the few original signal sample points to the end. 
    # max_capping_len = max( [ss.shape[ 0 ] for ss in rec_coef] )
    # signal_rec[ -max_capping_len: ] = rec_coef[ -1 ][ -max_capping_len: ]

    return( signal_rec )

def genWindowedWaveletCompressionData(
    raw_signal, 
    win_size, 
    max_level_cap, 
    max_compressed_len, 
    shuffle = False
):
    signal_windows = rolling_window( raw_signal, win_size = win_size )[ :-1 ]
    
    compressed_windows = [
        waveletCompressionCut( 
            signal_window, 
            max_level_cap = max_level_cap, 
            max_samples = max_compressed_len
        ) for signal_window in signal_windows
    ]

    data_in = []
    for ii in range( len( compressed_windows[0] ) ):
        data_in.append( [] )

    for lvl in range( len( compressed_windows[0] ) ):
        for sample in compressed_windows:
#             ##WARNING_HARDCODED: normalization. 
#             min_val = np.amin( sample[ lvl ] )
#             max_val = np.amax( sample[ lvl ] )
#             denom = (max_val - min_val)
#             if( max_val == min_val ): 
#                 denom = 1
#             data_in[ lvl ].append( 
#                 (sample[ lvl ] - min_val)/denom
#             )
            data_in[ lvl ].append( 
                sample[ lvl ]
            )
    for ii, tt in enumerate( data_in ):
        data_in[ ii ] = np.asarray( tt, dtype = "float32" )
        
    data_in = list( normalize_channels_01( data_in ) )
    return( data_in )

