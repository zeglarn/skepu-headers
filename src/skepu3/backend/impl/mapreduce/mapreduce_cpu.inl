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
		::CPU(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
#ifdef SKEPU_MPI
			/*
			Owner computes rule for the result matrix decides where to do work.
			*/

			// cluster::Partition<Ret> partition{};
			// partition.prepare(size);
			auto first = get<0>(std::forward<CallArgs>(args)...).getParent();
			size_t begin = first.part_begin();
			size_t end = first.part_end();

			const int rank = cluster::mpi_rank();
			const int num_ranks = cluster::mpi_size();

			auto rank_random = this->template prepareRandom<MapFunc::randomCount>(size);
			auto random = rank_random(rank);

			Ret _res;

			pack_expand(
				(
					skepu::cluster::handle_container_arg(
						get<AI>(std::forward<CallArgs>(args)...).getParent(),
						std::get<AI-arity>(typename MapFunc::ProxyTags{})), 0) ... );

			for (size_t i = begin; i < end; i++)
#else			
			auto random = this->template prepareRandom<MapFunc::randomCount>(size);
			for (size_t i = 0; i < size; i++)
#endif
			{
				auto index = (get<0>(std::forward<CallArgs>(args)...) + i).getIndex();
				Ret temp = F::forward(MapFunc::CPU,
					index, random,
					get<EI>(std::forward<CallArgs>(args)...)(i)...,
					get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-arity>(typename MapFunc::ProxyTags{}), index)...,
					get<CI>(std::forward<CallArgs>(args)...)...
				);
#ifdef SKEPU_MPI
				if (i != begin)
					pack_expand((get_or_return<OI>(_res) = ReduceFunc::CPU(get_or_return<OI>(_res), get_or_return<OI>(temp)), 0)...);
				else
					_res = temp;
#else
				pack_expand((get_or_return<OI>(res) = ReduceFunc::CPU(get_or_return<OI>(res), get_or_return<OI>(temp)), 0)...);
#endif
			}
#ifdef SKEPU_MPI
			std::vector<Ret> partsum(num_ranks);
			partsum[rank] = _res;

			size_t byte_size = sizeof(Ret);

			cluster::allgather(&_res,byte_size,&partsum[0],byte_size);

			_res = partsum[0];

			for (size_t i = 1; i < num_ranks; i++)
				pack_expand((get_or_return<OI>(_res) = ReduceFunc::CPU(get_or_return<OI>(_res), get_or_return<OI>(partsum[i])), 0)...);
			pack_expand((get_or_return<OI>(res) = ReduceFunc::CPU(get_or_return<OI>(_res), get_or_return<OI>(res)), 0)...);
#endif
			return res;
		}
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... AI, size_t... CI, typename... CallArgs>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::CPU(size_t size, pack_indices<OI...>, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Sync with device data
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			
#ifdef SKEPU_MPI
			/*
			Owner computes rule for the result matrix decides where to do work.
			*/

			cluster::Partition<Ret> partition{};
			partition.prepare(size);
			size_t begin = partition.part_begin();
			size_t end = partition.part_end();

			const int rank = cluster::mpi_rank();
			const int num_ranks = cluster::mpi_size();

			auto ranks_random = this->template prepareRandom<MapFunc::randomCount>(size, num_ranks);
			auto random = ranks_random(rank);

			Ret _res;

			pack_expand(
				(
					skepu::cluster::handle_container_arg(
						get<AI>(std::forward<CallArgs>(args)...).getParent(),
						std::get<AI-arity>(typename MapFunc::ProxyTags{})), 0) ... );

			for (size_t i = begin; i < end; i++)
#else		
			auto random = this->template prepareRandom<MapFunc::randomCount>(size);
			for (size_t i = 0; i < size; i++)
#endif
			{
				auto index = make_index(defaultDim{}, i, this->default_size_j, this->default_size_k, this->default_size_l);
				Ret temp = F::forward(MapFunc::CPU,
					index, random,
					get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-arity>(typename MapFunc::ProxyTags{}), index)...,
					get<CI>(std::forward<CallArgs>(args)...)...
				);
#ifdef SKEPU_MPI
				if (i != begin)
					pack_expand((get_or_return<OI>(_res) = ReduceFunc::CPU(get_or_return<OI>(_res), get_or_return<OI>(temp)), 0)...);
				else
					_res = temp;
#else
				pack_expand((get_or_return<OI>(res) = ReduceFunc::CPU(get_or_return<OI>(res), get_or_return<OI>(temp)), 0)...);
#endif
			}
#ifdef SKEPU_MPI
			std::vector<Ret> partsum(num_ranks);
			partsum[rank] = _res;

			size_t byte_size = sizeof(Ret);

			cluster::allgather(&_res,byte_size,&partsum[0],byte_size);

			_res = partsum[0];

			for (size_t i = 1; i < num_ranks; i++)
				pack_expand((get_or_return<OI>(_res) = ReduceFunc::CPU(get_or_return<OI>(_res), get_or_return<OI>(partsum[i])), 0)...);
				// _res = ReduceFunc::CPU(_res,partsum[i]);
			
			pack_expand((get_or_return<OI>(res) = ReduceFunc::CPU(get_or_return<OI>(_res), get_or_return<OI>(res)), 0)...);
#endif
			return res;
		}
	}
}
