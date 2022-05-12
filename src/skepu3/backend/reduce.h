/*! \file reduce.h
 *  \brief Contains a class declaration for the Reduce skeleton.
 */

#ifndef REDUCE_H
#define REDUCE_H

#include "reduce_helpers.h"

namespace skepu
{	
	namespace backend
	{
		
		/*!
		 *  \ingroup skeletons
		 */
		/*!
		 *
		 * \brief A specilalization of above class, used for 1D Reduce operation.
		 * Please note that the class name is same. The only difference is
		 * how you instantiate it either by passing 1 user function (i.e. 1D reduction)
		 * or 2 user function (i.e. 2D reduction). See code examples for more information.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		class Reduce1D : public SkeletonBase
		{
			
		public:
			using T = typename ReduceFunc::Ret;
			
			static constexpr auto skeletonType = SkeletonType::Reduce1D;
			using ResultArg = std::tuple<>;
			using ElwiseArgs = std::tuple<T>;
			using ContainerArgs = std::tuple<>;
			using UniformArgs = std::tuple<>;
			static constexpr bool prefers_matrix = false;
			
		public:
			Reduce1D(CUDAKernel kernel) : m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			void setReduceMode(ReduceMode mode)
			{
				this->m_mode = mode;
			}
			
			void setStartValue(T val)
			{
				this->m_start = val;
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
		protected:
			CUDAKernel m_cuda_kernel;
			
			ReduceMode m_mode = ReduceMode::RowWise;
			T m_start{};
			
			
			void CPU(VectorIterator<T> &res, MatrixIterator<T>& arg, size_t size);
			
			template<typename Iterator>
			T CPU(size_t size, T &res, Iterator arg);
			
#ifdef SKEPU_OPENMP
			
			void OMP(VectorIterator<T> &res, MatrixIterator<T>& arg, size_t size);
			
			template<typename Iterator>
			T OMP(size_t size, T &res, Iterator arg);
			
#endif
			
#ifdef SKEPU_CUDA
			
			void reduceSingleThreadOneDim_CU(size_t deviceID, VectorIterator<T> &res, const MatrixIterator<T> &arg, size_t numRows);
			
			void reduceMultipleOneDim_CU(size_t numDevices, VectorIterator<T> &res, const MatrixIterator<T> &arg, size_t numRows);
			
			void CU(VectorIterator<T> &res, const MatrixIterator<T>& arg, size_t numRows);
		
			template<typename Iterator>
			T reduceSingleThread_CU(size_t deviceID, size_t size, T &res, Iterator arg);
			
			template<typename Iterator>
			T reduceMultiple_CU(size_t numDevices, size_t size, T &res, Iterator arg);
		
			template<typename Iterator>
			T CU(size_t size, T &res, Iterator arg);
#endif
			
#ifdef SKEPU_OPENCL
			
			void reduceSingleThreadOneDim_CL(size_t deviceID, VectorIterator<T> &res, const MatrixIterator<T> &arg, size_t numRows);
			
			void reduceMultipleOneDim_CL(size_t numDevices, VectorIterator<T> &res, const MatrixIterator<T> &arg, size_t numRows);
			
			void CL(VectorIterator<T> &res, const MatrixIterator<T>& arg, size_t numRows);
		
			template<typename Iterator>
			T reduceSingle_CL(size_t deviceID, size_t size, T &res, Iterator arg);
			
			template<typename Iterator>
			T reduceMultiple_CL(size_t numDevices, size_t size, T &res, Iterator arg);
		
			template<typename Iterator>
			T CL(size_t size, T &res, Iterator arg);
			
#endif
			
					
#ifdef SKEPU_HYBRID
			
			void Hybrid(Vector<T> &res, Matrix<T>& arg);
			
			template<typename Iterator>
			T Hybrid(size_t size, T &res, Iterator arg);
			
#endif
			
			
		public:
			template<template<class> class Container>
			T operator()(Container<T> &arg)
			{
				// printf("T operator()(Container<T> &arg)\n");
#ifdef SKEPU_MPI
				arg.set_skeleton_iterator(true);
				T ret = this->backendDispatch(arg.part_size(), arg.begin());
				arg.set_skeleton_iterator(false);
				return ret;
#else
				return this->backendDispatch(arg.size(), arg.begin());
#endif
			}
		
			template<typename Iterator>
			T operator()(Iterator arg, Iterator arg_end)
			{
				return this->backendDispatch(arg_end - arg, arg.begin());
			}
			
			Vector<T> &operator()(Vector<T> &res, Matrix<T>& arg)
			{
			//	assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());
				
				const size_t m_size = arg.size();
				
				// TODO: check size
				
				this->selectBackend(m_size);

#ifdef SKEPU_MPI
				res.set_skeleton_iterator(true);
				res.mark_dirty();
				arg.set_skeleton_iterator(true);

				const size_t size = res.part_size();
#else
				const size_t size = res.size();
#endif
				
				Matrix<T> &arg_tr = (this->m_mode == ReduceMode::ColWise) ? arg.transpose(*this->m_selected_spec) : arg;
				MatrixIterator<T> arg_it = arg_tr.begin();
				VectorIterator<T> it = res.begin();

				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					this->Hybrid(res, arg_tr);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->CU(it, arg_it, size);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->CL(it, arg_it, size);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->OMP(it, arg_it, size);
					break;
#endif
				default:
					this->CPU(it, arg_it, size);
					// this->CPU(res, arg_tr);
				}

#ifdef SKEPU_MPI
				res.set_skeleton_iterator(false);
				arg.set_skeleton_iterator(false);
#endif
				
				return res;
			}
			
		private:
			template<typename Iterator>
			T backendDispatch(size_t size, Iterator arg)
			{
			//	assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());
				
				T res = this->m_start;
				T ret{};
				
				this->selectBackend(size);

#ifdef SKEPU_MPI
				const int rank = cluster::mpi_rank();
				const int numRanks = cluster::mpi_size();

				if (rank)
				{
					res = arg.getAddress()[0];
					arg++;
					if (size > 0)
						size--;
				}
#endif
				
				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					ret = this->Hybrid(size, res, arg);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					ret = this->CU(size, res, arg);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					ret = this->CL(size, res, arg);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					ret = this->OMP(size, res, arg);
					break;
#endif
				default:
					ret = this->CPU(size, res, arg);
				}
#ifdef SKEPU_MPI
				size_t byteSize = sizeof(T);
				std::vector<T> partsum(numRanks);
				cluster::allgather(&ret, byteSize,&partsum[0],byteSize);

				ret = partsum[0];
				for (size_t i = 1; i < numRanks; i++)
					ret = ReduceFunc::CPU(ret, partsum[i]);
#endif
				return ret;
			}
			
			
		};
		
		
		/*!
		 *  \class Reduce
		 *
		 *  \brief A class representing the Reduce skeleton both for 1D and 2D reduce operation for 1D Vector, 2D Dense Matrix/Sparse matrices.
		 *
		 *  This class defines the Reduce skeleton which support following reduction operations:
		 *  (a) (1D Reduction) Each element in the input range, yielding a scalar result by applying a commutative associative binary operator.
		 *     Here we consider dense/sparse matrix as vector thus reducing all (non-zero) elements of the matrix.
		 *  (b) (1D Reduction) Dense/Sparse matrix types: Where we reduce either row-wise or column-wise by applying a commutative associative binary operator. It returns
		 *     a \em SkePU vector of results that corresponds to reduction on either dimension.
		 *  (c) (2D Reduction) Dense/Sparse matrix types: Where we reduce both row- and column-wise by applying two commutative associative
		 *     binary operators, one for row-wise reduction and one for column-wise reduction. It returns a scalar result.
		 *  Two versions of this class are created using C++ partial class-template specialization to support
		 *  (a) 1D reduction (where a "single" reduction operator is applied on all elements or to 1 direction for 2D Dense/Sparse matrix).
		 *  (b) 2D reduction that works only for matrix (where two different reduction operations are used to reduce row-wise and column-wise separately.)
		 *  Once instantiated, it is meant to be used as a function and therefore overloading
		 *  \p operator(). The Reduce skeleton needs to be created with
		 *  a 1 or 2 binary user function for 1D reduction and 2D reduction respectively.
		 */
		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		class Reduce2D : public Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>
		{
			using T = typename ReduceFuncRowWise::Ret;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::Reduce2D;
			static constexpr bool prefers_matrix = true;
			
			Reduce2D(CUDARowWise row, CUDAColWise col) : Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>(row), m_cuda_colwise_kernel(col) {}
			
		private:
			CUDAColWise m_cuda_colwise_kernel;
			
			
			
			
		private:
			T CPU(MatrixIterator<T>& arg, size_t size);
			
#ifdef SKEPU_OPENMP
			
			T OMP(MatrixIterator<T>& arg, size_t size);
			
#endif
			
#ifdef SKEPU_CUDA
			
			T CU(T &res, const MatrixIterator<T>& arg, size_t numRows);
			
			T reduceSingleThread_CU(size_t deviceID, T &res, const MatrixIterator<T>& arg, size_t numRows);
			
			T reduceMultiple_CU(size_t numDevices, T &res, const MatrixIterator<T>& arg, size_t numRows);
			
#endif
			
#ifdef SKEPU_OPENCL
			
			T CL(T &res, const MatrixIterator<T>& arg, size_t numRows);
			
			T reduceSingle_CL(size_t deviceID, T &res, const MatrixIterator<T>& arg, size_t numRows);
			
			T reduceNumDevices_CL(size_t numDevices, T &res, const MatrixIterator<T>& arg, size_t numRows);
			
#endif
			
#ifdef SKEPU_HYBRID
			
			T Hybrid(T &res, Matrix<T>& arg);
			
#endif
			
			
			
		public:
			T operator()(Vector<T>& arg)
			{
				return Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>::operator()(arg);
			}
			
			T operator()(Matrix<T>& arg)
			{
			//	assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());
				
				this->selectBackend(arg.size());
				
				T res = this->m_start;
				T ret{};
				
				Matrix<T> &arg_tr = (this->m_mode == ReduceMode::ColWise) ? arg.transpose(*this->m_selected_spec) : arg;
#ifdef SKEPU_MPI
				const int rank = cluster::mpi_rank();
				const int numRanks = cluster::mpi_size();
				arg_tr.set_skeleton_iterator(true);
				size_t size = arg_tr.part_size();

				auto it = arg_tr.begin();
#else
				const int rank = 0;
				const int numRanks = 1;
				auto it = arg_tr.begin();
				size_t size = arg_tr.size();
#endif
				
				
				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					ret = Hybrid(res, arg_tr);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					ret = CU(res, arg_tr.begin(), arg_tr.total_rows());
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					ret = CL(res, arg_tr.begin(), arg_tr.total_rows());
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					ret = OMP(it, size);
					break;
#endif
				default:
					ret = CPU(it, size);
				}
#ifdef SKEPU_MPI
				arg_tr.set_skeleton_iterator(false);
				size_t byteSize = sizeof(T);
				std::vector<T> partsum(numRanks);
				cluster::allgather(&ret, byteSize,&partsum[0],byteSize);

				for (auto const &el : partsum)
					res = ReduceFuncColWise::CPU(res,el);
				// ret = partsum[0];
				// for (size_t i = 1; i < numRanks; i++)
				// 	ret = ReduceFuncColWise::CPU(ret, partsum[i]);
#else
				res = ReduceFuncColWise::CPU(res,ret);
#endif
				return res;
			}
			
		};
		
	} // end namespace backend

#ifdef SKEPU_MERCURIUM

template<typename T>
class Reduce1D : public SeqSkeletonBase
{
	using RedFunc = std::function<T(T, T)>;

public:
	void setReduceMode(ReduceMode mode);
	void setStartValue(T val);

	template<template<class> class Container>
	typename std::enable_if<is_skepu_container<Container<T>>::value, T>::type
	operator()(Container<T>& arg);

	Vector<T> &operator()(Vector<T> &res, Matrix<T>& arg);

protected:
	RedFunc redFunc;
	Reduce1D(RedFunc red);
};

template<typename T>
class Reduce2D: public Reduce1D<T>
{
	using RedFunc = std::function<T(T, T)>;

public:
	T operator()(Vector<T>& arg);
	T operator()(Matrix<T>& arg);

private:
	RedFunc colRedFunc;
	Reduce2D(RedFunc rowRed, RedFunc colRed);
};

template<typename T>
auto inline
Reduce(std::function<T(T,T)>)
-> Reduce1D<T>;

template<typename T>
auto inline
Reduce(T(*)(T,T))
-> Reduce1D<T>;

template<typename T>
auto inline
Reduce(T op)
-> decltype(Reduce(&T::operator()))
{
	return Reduce(&T::operator());
}

template<typename T>
auto inline
Reduce(T(*)(T,T), T(*)(T,T))
-> Reduce2D<T>;

template<typename T>
auto inline
Reduce(std::function<T(T,T)>, std::function<T(T,T)>)
-> Reduce2D<T>;

template<typename T, typename U>
auto inline
Reduce(T row, U col)
-> decltype(Reduce(lambda_cast(row), lambda_cast(col)));

#endif // SKEPU_MERCURIUM

} // end namespace skepu


#include "impl/reduce/reduce_cpu.inl"
#include "impl/reduce/reduce_omp.inl"
#include "impl/reduce/reduce_cl.inl"
#include "impl/reduce/reduce_cu.inl"
#include "impl/reduce/reduce_hy.inl"

#endif // REDUCE_H
