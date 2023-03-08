try:
	__IPYTHON__
except NameError:
	import multiprocessing as mp
	import multiprocessing.shared_memory
else: 
	import multiprocess as mp
	import multiprocess.shared_memory
import numpy as np
import os

def init_worker( 
	func_module_name, 
	shared_memory_block_master, 
	data_shape_ref, 
	data_type_ref, 
	kwargs
): 
	print( "\t\tInit worker process: ", mp.current_process() )

	global func_module
	try:
		func_module = __import__( func_module_name )
	except ModuleNotFoundError:
		print( "The function module file must be in the same folder as the calling script!" )

	assert hasattr( func_module, 'proc' ), "The function module must include a function in form of proc( data_tensor, idx )!"

	global data_shape
	data_shape = data_shape_ref
	global data_type
	data_type = data_type_ref

	## suppose to define a global ref to the shared block on each worker
	global shared_memory_block_ref
	shared_memory_block_ref = mp.shared_memory.SharedMemory( name = shared_memory_block_master.name )

	global kwargs_proc
	kwargs_proc = kwargs
	
def worker_proc( idx ):
	# print( "Processing on worker:\n\t", mp.current_process() )

	# suppose on each worker try to access the global ref to create a buffered ndarray
	global data_shape
	global data_type
	shared_data_ref = np.ndarray( shape = data_shape, dtype = data_type, buffer = shared_memory_block_ref.buf )

	worker_res = func_module.proc( shared_data_ref, idx, **kwargs_proc )
	
	return( worker_res )

def parallel_tensor_apply( 
	func_module_name, 
	data_tensor, 
	index_set, 
	max_processes = 99, 
	**kwargs 
):
	if( mp.current_process().name == "MainProcess" ):
		shared_memory_block_master = mp.shared_memory.SharedMemory( 
			create = True, 
			size = data_tensor.nbytes 
		)
		shared_data_master = np.ndarray( 
			shape = data_tensor.shape, 
			dtype = data_tensor.dtype, 
			buffer = shared_memory_block_master.buf 
		)
		## Copy the data tensor to the shared memory block once, performed only on the master process. 
		shared_data_master[:] = data_tensor[:]
		
		print( "Pool initiated..." )
		with mp.Pool( 
			processes = min( max_processes, os.cpu_count() ), 
			initializer = init_worker, 
			initargs = (
				func_module_name, 
				shared_memory_block_master, 
				data_tensor.shape, 
				data_tensor.dtype, 
				kwargs
			)
		) as pool: 
			res = pool.map( # The result matrices will be concatenated along the first axis to a tensor. 
				worker_proc, 
				index_set
			)
		print( "Pool closed." )
		return( res )