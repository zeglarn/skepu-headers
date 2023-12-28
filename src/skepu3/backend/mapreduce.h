/*! \file mapreduce.h
 *  \brief Contains a class declaration for the MapReduce skeleton.
 */

#ifndef MAPREDUCE_H
#define MAPREDUCE_H

namespace skepu
{
	namespace backend
	{
		/*!
		*  \ingroup skeletons
		*/
		/*!
		*  \class MapReduce
		*
		*  \brief A class representing the MapReduce skeleton.
		*
		*  This class defines the MapReduce skeleton which is a combination of the Map and Reduce operations. It produces
		*  the same result as if one would first Map one or more vectors (matrices) to a result vector (matrix), then do a reduction on that result.
		*  It is provided since it combines the mapping and reduction in the same computation kernel and therefore avoids some
		*  synchronization, which speeds up the calculation. Once instantiated, it is meant to be used as a function and therefore overloading
		*  \p operator(). There are several overloaded versions of this operator that can be used depending on how many elements
		*  the mapping function uses (one, two or three). There are also variants which takes iterators as inputs and those that
		*  takes whole containers (vectors, matrices).
		*/
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		class MapReduce : public SkeletonBase
		{
		public:
			MapReduce(CUDAKernel mapreduce, CUDAReduceKernel reduce)
			: m_cuda_kernel(mapreduce), m_cuda_reduce_kernel(reduce)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}

			static constexpr auto skeletonType = SkeletonType::MapReduce;
			using ResultArg = std::tuple<>;
			using ElwiseArgs = typename MapFunc::ElwiseArgs;
			using ContainerArgs = typename MapFunc::ContainerArgs;
			using UniformArgs = typename MapFunc::UniformArgs;
			static constexpr bool prefers_matrix = MapFunc::prefersMatrix;

		private:
			CUDAKernel m_cuda_kernel;
			CUDAReduceKernel m_cuda_reduce_kernel;
			size_t default_size_i = 0;
			size_t default_size_j = 0;
			size_t default_size_k = 0;
			size_t default_size_l = 0;
			StrideList<arity> m_strides{};

			static constexpr size_t outArity = MapFunc::outArity;
			static constexpr size_t numArgs = MapFunc::totalArity - (MapFunc::indexed ? 1 : 0)  - (MapFunc::usesPRNG ? 1 : 0);
			static constexpr size_t anyArity = std::tuple_size<typename MapFunc::ContainerArgs>::value;
			static constexpr typename make_pack_indices<outArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<arity, 0>::type elwise_indices{};
			static constexpr typename make_pack_indices<arity + anyArity, arity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, arity + anyArity>::type const_indices{};

			using defaultDim = index_dimension<typename std::conditional<MapFunc::indexed, typename MapFunc::IndexType, skepu::Index1D>::type>;

			using F = ConditionalIndexForwarder<MapFunc::indexed, MapFunc::usesPRNG, decltype(&MapFunc::CPU)>;
			using Ret = typename MapFunc::Ret;
			using RedType = typename ReduceFunc::Ret;

			Ret m_start{};

		public:
			void setStartValue(Ret val)
			{
				this->m_start = val;
			}

			void setDefaultSize(size_t i, size_t j = 0, size_t k = 0, size_t l = 0)
			{
				this->default_size_i = i;
				this->default_size_j = j;
				this->default_size_k = k;
				this->default_size_l = l;
			}
			
			template<typename... S>
			void setStride(S... strides)
			{
				this->m_strides = StrideList<arity>(strides...);
			}

			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}

			// For first elwise argument as container
			template<typename First, template<class> class Container, typename... CallArgs, REQUIRES(arity > 0 && is_skepu_container<Container<First>>::value)>
			Ret operator()(Container<First> &arg1, CallArgs&&... args)
			{
				return backendDispatch(
					out_indices, elwise_indices, any_indices, const_indices,
					arg1.size() / this->m_strides[0], arg1,
					std::forward<CallArgs>(args)...
				);
			}

			// For first elwise argument as iterator
			template<typename Iterator, typename... CallArgs, REQUIRES(sizeof...(CallArgs) + 1 == numArgs)>
			Ret operator()(Iterator arg1, Iterator arg1_end, CallArgs&&... args)
			{
				return backendDispatch(
					out_indices, elwise_indices, any_indices, const_indices,
					(arg1_end - arg1) / this->m_strides[0], arg1,
					std::forward<CallArgs>(args)...
				);
			}

			// For no elwise arguments
			template<typename... CallArgs, REQUIRES(sizeof...(CallArgs) == numArgs)>
			Ret operator()(CallArgs&&... args)
			{
				size_t size = this->default_size_i;
				if (defaultDim::value >= 2) size *= this->default_size_j;
				if (defaultDim::value >= 3) size *= this->default_size_k;
				if (defaultDim::value >= 4) size *= this->default_size_l;
				
				if (size == 0)
					SKEPU_WARNING("MapReduce<0> default size not set or set to 0.")

				return this->backendDispatch(
					out_indices, elwise_indices, any_indices, const_indices,
					size,
					std::forward<CallArgs>(args)...
				);
			}

		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			Ret backendDispatch(pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, size_t size, CallArgs&&... args)
			{
			//	assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());

				Ret res = this->m_start;
				Ret ret{};

				size_t globalSize{size};
				size_t localSize{size};
				size_t startIdx{0};
				int rank{0};
				int numRanks{1};

				if (disjunction((get<EI>(std::forward<CallArgs>(args)...).size() < size)...))
					SKEPU_ERROR("Non-matching container sizes");

				this->selectBackend(size);

#ifdef SKEPU_MPI
				rank = cluster::mpi_rank();
				numRanks = cluster::mpi_size();
				int *counts;
				cluster::Partition<Ret> p{};
				
				pack_expand((cluster::handle_container_arg(get<AI>(std::forward<CallArgs>(args)...).getParent(),std::get<AI-arity-outArity>(typename MapFunc::ProxyTags{})), 0)...);
				pack_expand((cluster::handle_read_write_access(get<AI>(std::forward<CallArgs>(args)...).getParent(),MapFunc::anyAccessMode[AI-arity-outArity]), 0)...);
				
				if (sizeof...(CallArgs) == 0)
				{
					p.prepare(globalSize);
					startIdx = p.part_begin();
					localSize = p.part_size();
					counts = p.counts;
				}
				else
				{
					pack_expand((localSize = get<EI>(std::forward<CallArgs>(args)...).getParent().part_size(), 0)...);
					pack_expand((startIdx = get<EI>(std::forward<CallArgs>(args)...).getParent().part_begin(), 0)...);
					pack_expand((counts = get<EI>(std::forward<CallArgs>(args)...).getParent().partition.counts, 0)...);	
				}
				if (!localSize) goto afterSwitch;
#endif

				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					ret = this->Hybrid(size, oi, ei, ai, ci, res,
						get<EI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					ret = this->CUDA(startIdx, localSize, globalSize, rank, numRanks, oi, ei, ai, ci, res,
						get<EI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					ret = this->CL(0, size, ei, ai, ci, res,
						get<EI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					ret = this->OMP(startIdx, localSize, globalSize, rank, numRanks, oi, ei, ai, ci, res,
						get<EI>(std::forward<CallArgs>(args)...).stridedBegin(size, this->m_strides[EI])...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				default:
					ret = this->CPU(startIdx, localSize, globalSize, rank, numRanks, oi, ei, ai, ci, res,
						get<EI>(std::forward<CallArgs>(args)...).stridedBegin(size, this->m_strides[EI])...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
				}

#ifdef SKEPU_MPI
afterSwitch:
				size_t byteSize = sizeof(Ret);
				std::vector<Ret> partsum(numRanks);
				cluster::allgather(&ret, byteSize,&partsum[0],byteSize);

				ret = partsum[0];
				for (size_t i = 1; i < numRanks; i++)
					if (counts[i])
						ret = ReduceFunc::CPU(ret, partsum[i]);
#endif

				return ret;
			}


			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			Ret CPU(size_t startIdx, size_t localSize, size_t globalSize, int rank, int numRanks, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

			template<size_t... OI, size_t... AI, size_t... CI, typename... CallArgs>
			Ret CPU(size_t startIdx, size_t localSize, size_t globalSize, int rank, int numRanks, pack_indices<OI...>, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);


#ifdef SKEPU_OPENMP

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
			Ret OMP(size_t startIdx, size_t localSize, size_t globalSize, int rank, int numRanks, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

			template<size_t... OI, size_t... AI, size_t... CI, typename ...CallArgs>
			Ret OMP(size_t startIdx, size_t localSize, size_t globalSize, int rank, int numRanks, pack_indices<OI...>, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

#endif // SKEPU_OPENMP

#ifdef SKEPU_CUDA
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			Ret CUDA(size_t startIdx, size_t localSize, size_t globalSize, int rank, int numRanks, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
			Ret mapReduceSingleThread_CU(size_t deviceID, size_t startIdx, size_t localSize, size_t globalSize, int rank, int numRanks, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
			Ret mapReduceMultiStream_CU(size_t deviceID, size_t startIdx, size_t localSize, size_t globalSize, int rank, int numRanks, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			Ret mapReduceSingleThreadMultiGPU_CU(size_t useNumGPU, size_t startIdx, size_t localSize, size_t globalSize, int rank, int numRanks, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
			Ret mapReduceMultiStreamMultiGPU_CU(size_t useNumGPU, size_t startIdx, size_t localSize, size_t globalSize, int rank, int numRanks, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
#endif // SKEPU_CUDA

#ifdef SKEPU_OPENCL

			template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			Ret CL(size_t startIdx, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

			template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			Ret mapReduceNumDevices_CL(size_t numDevices, size_t startIdx, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

			template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			Ret mapReduceSingle_CL(size_t deviceID, size_t startIdx, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

#endif // SKEPU_OPENCL


#ifdef SKEPU_HYBRID

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			Ret Hybrid(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

			template<size_t... OI, size_t... AI, size_t... CI, typename... CallArgs>
			Ret Hybrid(size_t size, pack_indices<OI...>, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

#endif // SKEPU_HYBRID

		}; // class MapReduce

	} // end namespace backend

#ifdef SKEPU_MERCURIUM

template<int Arity, int GivenArity, typename Ret, typename ... Args>
class MapReduceImpl: public SeqSkeletonBase
{
	static constexpr bool indexed = is_indexed<Args...>::value;
	using First = typename pack_element<indexed ? 1 : 0, Args...>::type;

public:
	void setStartValue(Ret val);
	void setDefaultSize(size_t i, size_t j = 0, size_t k = 0, size_t l = 0);

	template<
		template<class> class Container,
		typename... CallArgs,
		REQUIRES_VALUE(is_skepu_container<Container<First>>)>
	Ret operator()(Container<First>& arg1, CallArgs&&... args);

	template<
		template<class> class Container,
		typename... CallArgs,
		REQUIRES_VALUE(is_skepu_container<Container<First>>)>
	Ret operator()(const Container<First>& arg1, CallArgs&&... args);

	template<
		typename Iterator,
		typename... CallArgs,
		REQUIRES_VALUE(is_skepu_iterator<Iterator, First>)>
	Ret operator()(Iterator arg1, Iterator arg1_end, CallArgs&&... args);

	template< typename... CallArgs>
	Ret operator()(CallArgs&&... args);
};

template<int arity = SKEPU_UNSET_ARITY, typename Ret, typename ... Args>
auto
MapReduce(Ret (*)(Args...), Ret (*)(Ret, Ret))
-> MapReduceImpl<
		resolve_map_arity<arity, Args...>::value,
		arity,
		Ret,
		Args...>;

template<int arity = SKEPU_UNSET_ARITY, typename T, typename U>
auto
MapReduce(T map_op, U red_op)
-> decltype(MapReduce<arity>(lambda_cast(map_op), lambda_cast(red_op)));

#endif // SKEPU_MERCURIUM

} // end namespace skepu


#include "impl/mapreduce/mapreduce_cpu.inl"
#include "impl/mapreduce/mapreduce_omp.inl"
#include "impl/mapreduce/mapreduce_cl.inl"
#include "impl/mapreduce/mapreduce_cu.inl"
#include "impl/mapreduce/mapreduce_hy.inl"

#endif
