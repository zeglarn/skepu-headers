#ifndef STARPU_MATRIX_CONTAINER_INL
#define STARPU_MATRIX_CONTAINER_INL

#include <skepu2/cluster/starpu_matrix_container.hpp>
#include <skepu2/cluster/cut_structure.hpp>
#include <skepu2/cluster/cluster.hpp>
#include <stddef.h>

#include <unistd.h>

#include <starpu.h>

namespace skepu2
{
	namespace cluster
	{
		template<typename T>
		starpu_matrix_container<T>
		::starpu_matrix_container(const size_t & height,
		                          const size_t & width)
				:
			m_row_struct { height,
				std::min<size_t>(skepu2::cluster::mpi_size(), height) },
			m_col_struct { width,
					std::min<size_t>(skepu2::cluster::mpi_size(), width) }
		{
			mpi_size();
			partition();
			m_unpartitioned_valid = false;
		}


		template<typename T>
		starpu_matrix_container<T>
		::~starpu_matrix_container()
		{
			starpu_task_wait_for_all();

			invalidate_unpartition();
			if (m_unpartitioned_data != 0)
			{
				starpu_free((void*)m_unpartitioned_data);
				m_unpartitioned_data = 0;
			}



			for (size_t i {}; i < m_children.size(); ++i) {
				auto & handle = m_children[i];
				auto & data = m_child_data[i];


				starpu_data_unregister(handle);

				if(data != 0) {
						starpu_free((void*)data);
				}
			}

			m_children.clear();

		}


		/**
		 * @brief Get an index into `m_children` from a block column/row
		 *
		 * @param row
		 * @param col
		 * @return size_t 1D index of the block
		 */
		template<typename T>
		size_t starpu_matrix_container<T>
		::block_row_col_to_idx(const size_t & block_row,
		                       const size_t & block_col) const
		{
			assert(block_row < m_row_struct.block_count()
			       && "block row index out of bounds");
			assert(block_col < m_col_struct.block_count()
			       && "block column index out of bounds");
			return block_row * m_col_struct.block_count() + block_col;
		}

		/**
		 * @brief Get an index into the local pointer of a handle for an
		 * element
		 *
		 * @param row
		 * @param col
		 * @return size_t 1D index of the block
		 */
		template<typename T>
		size_t starpu_matrix_container<T>
		::local_elem_idx(starpu_data_handle_t & handle,
		                 const size_t & row,
		                 const size_t & col) const
		{
			assert(row < m_row_struct.size() && "row index out of bounds");
			assert(col < m_col_struct.size() && "column index out of bounds");
			const size_t local_ld = std::max(starpu_matrix_get_local_ld(handle),
																			 starpu_matrix_get_nx(handle));
			const size_t local_idx = m_row_struct.offset_in_block(row)
					* local_ld
					+ m_col_struct.offset_in_block(col);

			assert(starpu_matrix_get_ny(handle) > 0);
			assert(starpu_matrix_get_nx(handle) > 0);
			assert(starpu_matrix_get_local_ld(handle) > 0);
			assert(local_idx <
						 starpu_matrix_get_ny(handle)
						 *starpu_matrix_get_local_ld(handle));
			return local_idx;
		}


		/**
		 * @brief Get element at position in matrix
		 *
		 * @param row
		 * @param col
		 * @return T
		 */
		template <typename T>
		T starpu_matrix_container<T>
		::operator()(const size_t & row, const size_t & col)
		{
			assert (row < m_row_struct.size() && col < m_col_struct.size());
			starpu_data_handle_t & handle = get_block_by_elem(row, col);
			starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, handle);


			starpu_data_acquire(handle, STARPU_R);
			const T* local = (const T*)starpu_matrix_get_local_ptr(handle);
			auto idx = local_elem_idx(handle, row, col);
			T res = local[local_elem_idx(handle, row, col)];
			starpu_data_release(handle);

			return res;
		}

		template<typename T>
		void starpu_matrix_container<T>
		::set(const size_t & row, const size_t & col, const T & value)
		{
			invalidate_unpartition();
			starpu_data_handle_t & handle = get_block_by_elem(row, col);

			// TODO: Is this correct?
			starpu_mpi_cache_flush(MPI_COMM_WORLD, handle);
			if (elem_owner(row, col) == skepu2::cluster::mpi_rank())
			{
				starpu_data_acquire(handle, STARPU_RW);
				T* local = (T*)starpu_matrix_get_local_ptr(handle);
				local[local_elem_idx(handle, row, col)] = value;
				starpu_data_release(handle);
			}
		}

		template<typename T>
		void starpu_matrix_container<T>
		::set(const size_t & i, const T & value)
		{
			set(i/width(), i%width(), value);
		}


		/**
		 * @brief Get copy of element at position (col/width(), col%width())
		 *
		 * @param col Description of col
		 * @return T
		 */
		template <typename T>
		T starpu_matrix_container<T>
		::operator[](const size_t & col)
		{
			return (*this)(col/width(), col%width());
		}


		/**
		 * @brief Get a handle for the block containing the element at (row, col)
		 *
		 * @param row
		 * @param col
		 * @return starpu_data_handle_t &
		 */
		template <typename T>
		starpu_data_handle_t& starpu_matrix_container<T>
		::get_block_by_elem(const size_t & row, const size_t & col)
		{
			const size_t block_row = m_row_struct.block(row);
			const size_t block_col = m_col_struct.block(col);
			return m_children[block_row_col_to_idx(block_row, block_col)];
		}



		/**
		 * @brief Get a handle for a block
		 *
		 * @param block_row
		 * @param block_col
		 * @return starpu_data_handle_t &
		 */
		template <typename T>
		starpu_data_handle_t& starpu_matrix_container<T>
		::get_block(const size_t & block_row, const size_t & block_col)
		{
			return m_children[block_row_col_to_idx(block_row, block_col)];
		}



		/**
		 * @brief Return the rank owning the block
		 *
		 * @param block_row
		 * @param block_col
		 * @return size_t owning rank
		 */
		template <typename T>
		size_t starpu_matrix_container<T>
		::block_owner(const size_t & block_row, const size_t & block_col) const
		{
			return block_row_col_to_idx(block_row, block_col)
				% skepu2::cluster::mpi_size();
		}

		/**
		 * @brief Return the rank owning the block owning the element
		 *
		 * @param block_row
		 * @param block_col
		 * @return size_t owning rank
		 */
		template <typename T>
		size_t starpu_matrix_container<T>
		::elem_owner(const size_t & row, const size_t & col) const
		{
			const size_t block_row = m_row_struct.block(row);
			const size_t block_col = m_col_struct.block(col);
			return block_owner(block_row, block_col);
		}



		/**
		 * @brief Allocate and partition the matrix onto the
		 * participating ranks.
		 *
		 */
		template <typename T>
		void starpu_matrix_container<T>
		::partition()
		{
			m_children.resize(m_row_struct.block_count()*m_col_struct.block_count());
			m_child_data.resize(m_children.size());

			for (size_t row_block {};
			     row_block < m_row_struct.block_count();
			     ++row_block) {
				for (size_t col_block {};
				     col_block < m_col_struct.block_count();
				     ++col_block) {
					uintptr_t data = (uintptr_t)NULL;
					int home_node = -1;
					const size_t owner = block_owner(row_block, col_block);

					// Only allocate data if we are the owner.
					if (owner == skepu2::cluster::mpi_rank()) {
						starpu_malloc((void**)&data,
						              sizeof(T)*
						              m_row_struct.block_size(row_block)*
						              m_col_struct.block_size(col_block));
						home_node = STARPU_MAIN_RAM;
						m_n_owned_children++;
					}

					m_child_data[block_row_col_to_idx(row_block, col_block)] = data;

					// Select the handle
					auto & handle =
						m_children[block_row_col_to_idx(row_block, col_block)];


					starpu_matrix_data_register(&handle,
					                            home_node,
					                            data,
					                            // stride
					                            m_col_struct.block_size(col_block),
					                            // width
					                            m_col_struct.block_size(col_block),
					                            // height
					                            m_row_struct.block_size(row_block),
					                            sizeof(T));

					assert(starpu_matrix_get_nx(handle)
					       == m_col_struct.block_size(col_block));

					assert(starpu_matrix_get_ny(handle)
					       == m_row_struct.block_size(row_block));

					starpu_mpi_data_register(handle,
					                         skepu2::cluster::mpi_tag(),
					                         owner);
				}
			}
		}



		/**
		 * @brief The height of the matrix
		 *
		 * @return size_t
		 */
		template <typename T>
		size_t starpu_matrix_container<T>
		::height() const
		{
			return m_row_struct.size();
		}



		/**
		 * @brief The width of the matrix
		 *
		 * @return size_t
		 */
		template <typename T>
		size_t starpu_matrix_container<T>
		::width() const
		{
			return m_col_struct.size();
		}



		/**
		 * @brief The total number of elements contained within the
		 * matrix
		 *
		 * @return size_t
		 */
		template <typename T>
		size_t starpu_matrix_container<T>
		::size() const
		{
			return width()*height();
		}


		template <typename T>
		starpu_data_handle_t starpu_matrix_container<T>
		::allgather()
		{
			if(m_unpartitioned_valid)
			{
				return m_unpartitioned;
			}

			// Allocate and register the full matrix on each node
			if (m_unpartitioned_data == 0)
			{

				starpu_malloc((void**)&m_unpartitioned_data,
				              sizeof(T)*m_row_struct.size()*m_col_struct.size());
			}
			auto data = m_unpartitioned_data;
			T* data_ptr = (T*)data;


			assert(m_col_struct.size() > 0);
			assert(m_row_struct.size() > 0);

			starpu_matrix_data_register(&m_unpartitioned,
			                            STARPU_MAIN_RAM,
			                            data,
			                            m_col_struct.size(), // stride
			                            m_col_struct.size(), // width
			                            m_row_struct.size(), // height
			                            sizeof(T));

			assert(starpu_matrix_get_nx(m_unpartitioned) == m_col_struct.size());
			assert(starpu_matrix_get_ny(m_unpartitioned) == m_row_struct.size());

			starpu_mpi_data_register(m_unpartitioned,
			                         skepu2::cluster::mpi_tag(),
			                         skepu2::cluster::mpi_rank());


			starpu_data_acquire(m_unpartitioned, STARPU_W);



			for (auto & child : m_children) {
				starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, child);
			}

			for (size_t block_row {};
			     block_row < m_row_struct.block_count();
			     ++block_row)
			{
				for (size_t block_col {};
				     block_col < m_col_struct.block_count();
				     ++block_col)
				{
					auto handle = get_block(block_row, block_col);
					size_t local_cols = starpu_matrix_get_nx(handle);
					size_t local_rows = starpu_matrix_get_ny(handle);
					size_t local_ld   = starpu_matrix_get_local_ld(handle);

					starpu_data_acquire(handle, STARPU_R);
					T* local_data = (T*) starpu_matrix_get_local_ptr(handle);

					size_t row_offset = m_row_struct.block_start_idx(block_row);
					size_t col_offset = m_col_struct.block_start_idx(block_col);

					for (size_t row {}; row < local_rows; ++row) {
						for (size_t col {}; col < local_cols; ++col) {
							data_ptr[width()*(row + row_offset) + (col + col_offset)] =
								local_data[row*local_ld + col];
						}
					}

					starpu_data_release(handle);
				}
			}

			starpu_data_release(m_unpartitioned);

			m_unpartitioned_valid = true;
			return m_unpartitioned;
		}



		/**
		 * @brief Invalidate any container arguments deriving from this
		 * data
		 *
		 * @return bool true if anything change, false otherwise.
		 */
		template <typename T>
		bool starpu_matrix_container<T>
		::invalidate_unpartition()
		{

			bool res = m_unpartitioned_valid;
			if (m_unpartitioned_valid)
			{

					// TODO: Should not be needed!
				starpu_task_wait_for_all();

				starpu_data_unregister(m_unpartitioned);
			}
			m_unpartitioned_valid = false;

			return res;
		}






		/**
		 * @brief Returns the number of rows between the given row and the
		 * next block boundary
		 *
		 * @param row
		 * @return size_t
		 */
		template <typename T>
		size_t starpu_matrix_container<T>
		::row_block_height(const size_t & row) const
		{
			return m_row_struct.size_in_block(row);
		}

		/**
		 * @brief Returns the number of cols between the given col and the
		 * next block boundary
		 *
		 * @param row
		 * @return size_t
		 */
		template <typename T>
		size_t starpu_matrix_container<T>
		::col_block_width(const size_t & col) const
		{
			return m_col_struct.size_in_block(col);
		}


		/**
		 * @brief Find the largest possible 2D cut within a single 2D block,
		 * starting from `idx`
		 *
		 * @param idx Starting location
		 * @return helpers::handle_cut
		 */
		template <typename T>
		helpers::handle_cut starpu_matrix_container<T>
		::largest_cut(const Index2D & idx)
		{
			assert(idx.row < m_row_struct.size());
			assert(idx.col < m_col_struct.size());

			Offset2D local_offset;
			Size2D local_size;

			starpu_data_handle_t & handle = get_block_by_elem(idx.row, idx.col);

			local_offset.row = m_row_struct.offset_in_block(idx.row);
			local_offset.col = m_col_struct.offset_in_block(idx.col);
			local_offset.i = local_offset.col;

			local_size.row = m_row_struct.size_in_block(idx.row);
			local_size.col = m_col_struct.size_in_block(idx.col);
			local_size.i = local_size.col;
			return { handle, local_offset, local_size };
		}
	} // cluster
} // skepu2

#endif /* STARPU_MATRIX_CONTAINER_INL */
