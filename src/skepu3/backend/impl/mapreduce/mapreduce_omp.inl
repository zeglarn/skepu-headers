/*! \file mapreduce_omp.inl
*  \brief Contains the definitions of OpenMP specific member functions for the MapReduce skeleton.
*/

#ifdef SKEPU_OPENMP

#include <omp.h>
#include <iostream>
#include <vector>

namespace skepu
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t startIdx, size_t localSize, size_t globalSize, int rank, int numRanks, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);

			const int max_threads = omp_get_max_threads();
			const int global_num_threads = numRanks * max_threads;

			std::vector<Ret> parsums(std::min<size_t>(localSize, max_threads));
			bool first = true;
			
			size_t threads = std::min<size_t>(localSize, max_threads);
			auto random = this->template prepareRandom<MapFunc::randomCount>(globalSize, global_num_threads);
			
			// Perform Map and partial Reduce with OpenMP
			#pragma omp parallel firstprivate(first)
			{
				const size_t myid = omp_get_thread_num();
				const size_t myGlobalId = rank * max_threads + myid;
				const size_t endIdx = startIdx + localSize;
				#pragma omp for schedule(runtime) 
				for (size_t i = startIdx; i < endIdx; ++i)
				{
					auto index = (get<0>(std::forward<CallArgs>(args)...) + i).getIndex();
					Ret tempMap = F::forward(MapFunc::OMP,
						index, random(myGlobalId),
						get<EI>(std::forward<CallArgs>(args)...)(i)...,
						get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-arity>(typename MapFunc::ProxyTags{}), index)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					if (first) 
					{
						parsums[myid] = tempMap;
						first = false;
					}
					else
						pack_expand((get_or_return<OI>(parsums[myid]) = ReduceFunc::OMP(get_or_return<OI>(parsums[myid]), get_or_return<OI>(tempMap)), 0)...);
				}
			}

			// Final Reduce sequentially
			size_t i{0};
			if (!rank)
				res = parsums[i++];

			for ( ; i < parsums.size(); i++)
				pack_expand((get_or_return<OI>(res) = ReduceFunc::OMP(get_or_return<OI>(res), get_or_return<OI>(parsums[i])), 0)...);
			
			
			return res;
		}
		
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... AI, size_t... CI, typename ...CallArgs>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t startIdx, size_t localSize, size_t globalSize, int rank, int numRanks, pack_indices<OI...>, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP MapReduce: localSize = " << localSize);
			// Sync with device data
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			const int max_threads = omp_get_max_threads();
			const int global_num_threads = numRanks * max_threads;

			std::vector<Ret> parsums(std::min<size_t>(localSize, max_threads));
			bool first = true;
			
			size_t threads = std::min<size_t>(localSize, max_threads);
			auto random = this->template prepareRandom<MapFunc::randomCount>(globalSize, global_num_threads);
			
			const size_t endIdx = startIdx + localSize;
			// Perform Map and partial Reduce with OpenMP
			#pragma omp parallel firstprivate(first)
			{
				const size_t myid = omp_get_thread_num();
				const size_t myGlobalId = rank * max_threads + myid;

				#pragma omp for schedule(runtime) 
				for (size_t i = startIdx; i < endIdx; ++i)
				{
					auto index = make_index(defaultDim{}, i, this->default_size_j, this->default_size_k, this->default_size_l);
					Ret tempMap = F::forward(MapFunc::OMP,
						index, random(myGlobalId),
						get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-arity>(typename MapFunc::ProxyTags{}), index)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					if (first) 
					{
						parsums[myid] = tempMap;
						first = false;
					}
					else
						pack_expand((get_or_return<OI>(parsums[myid]) = ReduceFunc::OMP(get_or_return<OI>(parsums[myid]), get_or_return<OI>(tempMap)), 0)...);
				}
			}
			
			// Final Reduce sequentially
			// for (Ret const& parsum : parsums)
			// 	pack_expand((get_or_return<OI>(res) = ReduceFunc::OMP(get_or_return<OI>(res), get_or_return<OI>(parsum)), 0)...);

			size_t i{0};
			if (!rank)
				res = parsums[i++];

			for ( ; i < parsums.size(); i++)
				pack_expand((get_or_return<OI>(res) = ReduceFunc::OMP(get_or_return<OI>(res), get_or_return<OI>(parsums[i])), 0)...);
			
			return res;
		}
	} // namespace backend
} // namespace skepu

#endif
