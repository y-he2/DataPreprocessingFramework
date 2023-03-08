from header import *

def list_files( dir_path, ext = ".csv" ):
    file_list = [
        file_name for file_name in os.listdir( dir_path ) 
        if os.path.isfile( os.path.join( dir_path, file_name ) ) and file_name.endswith( ext )
    ]
    return( file_list )
def load_csv_file( dir_path, file_name, col_name = None, **kwargs ):
    file_path = dir_path + file_name
    df_temp = pd.read_csv( file_path, **kwargs )
    if( col_name is not None ): 
        return( df_temp[ col_name ] )
    else:
        return( df_temp )
def load_csv_files( dir_path, **kwargs ):
    print( "Loading:" )
    file_data = []
    for ii, name in enumerate( list_files( dir_path ) ):
        print( "\tFile-", ii, ": ", name, sep = "" )
        file_data.append( load_csv_file( dir_path, name, **kwargs ) ) 
    return( file_data )

def normalize_channels( matrix, signal_axis = 0 ): 
    chan_max = np.amax( matrix, axis = signal_axis )
    chan_min = np.amin( matrix, axis = signal_axis )
    gap = chan_max - chan_min
    gap[ np.where( gap == 0 ) ] = np.finfo( np.float32 ).eps
    matrix = (matrix - chan_min)/gap
    return( matrix )

def normalize_df_mean( df ):
    return( (df - df.mean())/(df.std() + np.finfo(float).eps) )
    
def normalize_df_mm( df ):
    return( (df - df.min())/(df.max() - df.min() + np.finfo(float).eps) )
