/*! \file mapreduce_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the MapReduce skeleton.
 */

namespace skepu
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::CPU(size_t size, size_t start, int rank, int numRanks, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			auto rank_random = this->template prepareRandom<MapFunc::randomCount>(size,numRanks);
			auto random = rank_random(rank);
			for (size_t i = 0; i < size; i++)
			{
				auto index = (get<0>(std::forward<CallArgs>(args)...) + i).getIndex();
				Ret temp = F::forward(MapFunc::CPU,
					index, random,
					get<EI>(std::forward<CallArgs>(args)...)(i)...,
					get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-arity>(typename MapFunc::ProxyTags{}), index)...,
					get<CI>(std::forward<CallArgs>(args)...)...
				);
				pack_expand((get_or_return<OI>(res) = ReduceFunc::CPU(get_or_return<OI>(res), get_or_return<OI>(temp)), 0)...);
			}
			return res;
		}
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... AI, size_t... CI, typename... CallArgs>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::CPU(size_t size, size_t start, int rank, int numRanks, pack_indices<OI...>, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Sync with device data
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);

			auto rank_random = this->template prepareRandom<MapFunc::randomCount>(size,numRanks);
			auto random = rank_random(rank);

			const size_t end = start + size;

			for (size_t i = start; i < end; i++)
			{
				auto index = make_index(defaultDim{}, i, this->default_size_j, this->default_size_k, this->default_size_l);
				Ret temp = F::forward(MapFunc::CPU,
					index, random,
					get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-arity>(typename MapFunc::ProxyTags{}), index)...,
					get<CI>(std::forward<CallArgs>(args)...)...
				);
				pack_expand((get_or_return<OI>(res) = ReduceFunc::CPU(get_or_return<OI>(res), get_or_return<OI>(temp)), 0)...);
			}
			return res;
		}
	}
}
