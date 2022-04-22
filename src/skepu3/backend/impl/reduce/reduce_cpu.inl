/*! \file reduce_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the Reduce skeleton.
 */

namespace skepu
{
	namespace backend
	{
		/*!
		 *  Performs the Reduction on a whole Matrix either row-wise or column-wise. Returns a \em SkePU vector of reduction result.
		 *  Using the \em CPU as backend.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		void Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::CPU(Vector<T> &res, Matrix<T>& arg)
		{
			DEBUG_TEXT_LEVEL1("CPU Reduce (Matrix 1D): rows = " << arg.total_rows() << ", cols = " << arg.total_cols() << "\n");
			
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			
			// Make sure we are properly synched with device data
			arg.updateHost();
			
			T *data = arg.getAddress();
			for (size_t r = 0; r < rows; ++r, data += cols)
			{
				res(r) = data[0];
				for (size_t c = 1; c < cols; c++)
					res(r) = ReduceFunc::CPU(res(r), data[c]);
			}
			
		}
		
		
		/*!
		 *  Performs the Reduction on a range of elements. Returns a scalar result. Does the reduction on the \em CPU
		 *  by iterating over all elements in the range.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		typename ReduceFunc::Ret Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::CPU(size_t size, T &res, Iterator arg)
		{
			DEBUG_TEXT_LEVEL1("CPU Reduce (1D): size = " << size << "\n");
			
			// Make sure we are properly synched with device data
			arg.getParent().updateHost();
#ifdef SKEPU_MPI

			size_t begin = arg.getParent().part_begin();
			size_t end = arg.getParent().part_end();
			
			int rank = cluster::mpi_rank();
			int num_ranks = cluster::mpi_size();

			T _res = arg(begin++);

			for (size_t i = begin; i < end; i++)
				_res = ReduceFunc::CPU(_res, arg(i));
			
			std::vector<T> partsum(num_ranks);
			partsum[rank] = _res;

			size_t byte_size = sizeof(T);

			cluster::allgather(&_res,byte_size,&partsum[0],byte_size);

			_res = partsum[0];

			for (size_t i = 1; i < num_ranks; i++)
				_res = ReduceFunc::CPU(_res, partsum[i]);

			res = ReduceFunc::CPU(res,_res);

#else
			// Uses operator () to avoid unneccessary synchronization function calls
			for (size_t i = 0; i < size; ++i)
				res = ReduceFunc::CPU(res, arg(i));
#endif
			
			return res;
		}
		
		
		/*!
		 *  Performs the 2D Reduction (First row-wise then column-wise) on a
		 *  input Matrix. Returns a scalar result.
		 *  Using the \em CPU as backend.
		 */
		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		typename ReduceFuncRowWise::Ret Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>
		::CPU(T &res, Matrix<T>& arg)
		{
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			
			DEBUG_TEXT_LEVEL1("CPU Reduce (2D): rows = " << rows << ", cols = " << cols << "\n");
			
			// Make sure we are properly synched with device data
			arg.updateHost();
			
			T *data = arg.getAddress();
			for (size_t r = 0; r < rows; ++r, data += cols)
			{
				T tempResult = data[0];
				for (size_t c = 1; c < cols; c++)
					tempResult = ReduceFuncRowWise::CPU(tempResult, data[c]);
				
				res = ReduceFuncColWise::CPU(res, tempResult);
			}
			
			return res;
		}
		
	} // end namespace backend
} // end namespace skepu
