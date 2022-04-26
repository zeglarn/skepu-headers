/*! \file map_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the Map skeleton.
 */

namespace skepu
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel> 
		::CPU(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CPU Map: size = " << size);
			
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			

#ifdef SKEPU_MPI
			/*
			Owner computes rule for the result matrix decides where to do work.
			*/

			auto &out = get<0>(std::forward<CallArgs>(args)...).getParent();
			size_t begin = out.part_begin();
			size_t end = out.part_end();

			if (size > 0) out.dirty = true;

			const int rank = cluster::mpi_rank();
			const int num_ranks = cluster::mpi_size();

			auto rank_random = this->template prepareRandom<MapFunc::randomCount>(size,num_ranks);
			auto random = rank_random(rank);
			pack_expand(
				(
					skepu::cluster::handle_container_arg(
						get<AI>(std::forward<CallArgs>(args)...).getParent(),
						std::get<AI-arity-outArity>(typename MapFunc::ProxyTags{})), 0) ... );

			/*
			Run user function on the elements that belong to this rank.
			*/
			for (size_t i = begin; i < end; ++i)
#else
			auto random = this->template prepareRandom<MapFunc::randomCount>(size);
			for (size_t i = 0; i < size; ++i)
#endif			
			{
				auto index = (get<0>(std::forward<CallArgs>(args)...) + i).getIndex();
				auto res = F::forward(MapFunc::CPU, index, random,
					get<EI>(std::forward<CallArgs>(args)...)(i)..., 
					get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-arity-outArity>(typename MapFunc::ProxyTags{}), index)...,
					get<CI>(std::forward<CallArgs>(args)...)...
				);
				SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
			}
		}
	}
}

