import numpy as np
import matplotlib.pyplot as plt

from mp_utility import *

## INPUT_SHAPE: 
def clip_df_perc_outliers( df_raw, outlier_perc = 0.05 ):
    keep_perc = 1 - outlier_perc
    for col_name in df_raw:
        ss = df_raw[ col_name ].values
        ss = ss[ ~np.isnan( ss ) ]
        freq, val = np.histogram( ss, bins = 128, weights = np.ones_like( ss )/len( ss ) )

        hist_pdf = np.cumsum( freq[ ::-1 ] ) 
        cap_idx = np.argmax( hist_pdf[ hist_pdf > keep_perc ] )
        cap_val_left = val[ cap_idx ] 

        hist_pdf = np.cumsum( freq ) 
        cap_idx = len( freq ) - np.argmax( hist_pdf[ hist_pdf > keep_perc ] )
        cap_val_right = val[ cap_idx ] 

        df_raw[ col_name ] = df_raw[ col_name ].clip( cap_val_left, cap_val_right )
    return( df_raw )



## INPUT_SHAPE: (TENSOR_SHAPE) 
## OUTPUT_SHAPE: (TOTAL_WINDOWS, WINDOW_SIZE, ...TENSOR_SHAPE) 
def gen_rolling_window( tensor_in, win_size ):
    if( tensor_in.shape[ 0 ] < win_size ):
        print( "ERROR: Window size too large | shape: ", tensor_in.shape, "| win_size: ", win_size )
        return()
    total_windows = tensor_in.shape[ 0 ] - win_size + 1
    shape =  (total_windows, win_size) + tensor_in.shape[ 1: ] 
    strides = (tensor_in.strides[ 0 ], ) + tensor_in.strides 
    windowed_tensor = np.lib.stride_tricks.as_strided( tensor_in, shape, strides )
    return windowed_tensor




## INPUT_SHAPE: (TOTAL_TIME_STEPS, TOTAL_TIME_SERIES) 
## OUTPUT_SHAPE: (TOTAL_WINDOWS, SIG_MATRIX_SHAPE = TOTAL_TIME_SERIES^2, TOTAL_SCALES) 
def generate_multiscale_signature_matrix_series( func_module_name, scales, ts_tensor ): 
    print( "Generating multiscale signature matrices..." ) 
    total_scales = len( scales ) 
    total_series = ts_tensor.shape[ -1 ] 

#     sig_tensor = np.like( [total_windows_padded, total_series, total_series, total_scales] )
    sig_tensor = []
    try:
        for scale_id, current_scale in enumerate( scales ): 
            print( "\t", "current_scale:", current_scale )
            data_tensor_windowed_current_scale = gen_rolling_window( ts_tensor, current_scale )
            total_windows = data_tensor_windowed_current_scale.shape[ 0 ]
            ## Invoke the parallel apply here:
            signature_data_per_scale = parallel_tensor_apply( 
                func_module_name = func_module_name, 
                data_tensor = data_tensor_windowed_current_scale, 
                index_set = range( total_windows ), 
                max_processes = 12, 
                current_scale = current_scale
            )
            signature_data_per_scale = np.concatenate( signature_data_per_scale, axis = 0 )
            ## Cut off data for each scale with more windows than others. 
            min_total_windows = ts_tensor.shape[ 0 ] - scales[ -1 ] + 1 
            signature_data_per_scale = signature_data_per_scale[ :min_total_windows, ... ]
            print( "\t"*2, signature_data_per_scale.shape )
            sig_tensor.append( signature_data_per_scale )
    except KeyboardInterrupt:
        print( "Break!")
    
    sig_tensor = np.concatenate( sig_tensor, axis = -1 )
    print( "\t", sig_tensor.shape )
    return( sig_tensor )
