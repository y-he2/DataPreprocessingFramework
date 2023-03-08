import numpy as np
import scipy as sp
import scipy.stats
import multiprocess as mp

# def proc( tt, idx ):
#     print( mp.current_process().name )
#     print( np.asanyarray( idx ), flush = True )
#     return( tt + 1 )

def proc( data_tensor, idx, **kwargs ):
    current_scale = kwargs.get( 'current_scale', 1 )
    
    total_series = data_tensor.shape[ -1 ]
    worker_res = np.zeros( [1, total_series, total_series, 1] )
    
    # if( idx % 100 == 0 ): 
        # print( "\tCurrent window:", idx )
        
    for ts1_id in range( total_series ):
        for ts2_id in range( total_series ):
            if( ts1_id <= ts2_id ):
                if( 
                    not all( data_tensor[ idx, :, ts1_id ] == data_tensor[ idx, 0, ts1_id ] ) and
                    not all( data_tensor[ idx, :, ts2_id ] == data_tensor[ idx, 0, ts2_id ] ) 
                ):
                    pearson_coeff = scipy.stats.pearsonr( 
                        data_tensor[ idx, :, ts1_id ], 
                        data_tensor[ idx, :, ts2_id ] 
                    )[ 0 ]
                    total_change = np.inner( 
                            data_tensor[ idx, :, ts1_id ], 
                            data_tensor[ idx, :, ts2_id ] 
                        )*np.inner( 
                            np.abs( np.diff( data_tensor[ idx, :, ts1_id ] ) ), 
                            np.abs( np.diff( data_tensor[ idx, :, ts2_id ] ) )
                        )
                        
                    if( pearson_coeff > 0 ):
                        worker_res[ 0, ts1_id, ts2_id, 0 ] = np.power( total_change*pearson_coeff, 1/3 )
                    else:
                        worker_res[ 0, ts2_id, ts1_id, 0 ] = np.power( -total_change*pearson_coeff, 1/3 )
                else: 
                    worker_res[ 0, ts1_id, ts2_id, 0 ] = 0
    return( worker_res/current_scale )