/*! \file matrix_mpi.inl
 *  \brief Contains the definitions of mpi specific member functions for the Matrix container.
 */

#ifdef SKEPU_MPI

namespace skepu
{
    /*
    Row-wise partitions.
    */
    template<typename T>
    void Matrix<T>::partition_prepare()
    {
        this->partition.prepare(this->m_rows,this->m_cols);
    }

    template<typename T>
    void Matrix<T>::partition_prepare(size_t major_dim, size_t minor_dims)
    {
        this->partition.prepare(major_dim, minor_dims);
    }

    template<typename T>
    size_t Matrix<T>::part_begin()
    {
        return this->partition.part_begin();
    }

    template<typename T>
    size_t Matrix<T>::part_end()
    {
        return this->partition.part_end();
    }

    template<typename T>
    void Matrix<T>::allgather()
    {
        if (!this->dirty) return;

        this->dirty = false;

#ifdef SKEPU_MPI_DEBUG
        if (!cluster::mpi_rank())
            std::cout << "<<<[ " << this->getName() << " is running allgather. ]>>>\n";
#endif

        skepu::cluster::allgatherv_inplace(
            this->m_data,
            this->partition.byte_counts,
            this->partition.byte_displs
        );
    }

    template<typename T>
    void Matrix<T>::gather_to_root()
    {
        if (!this->dirty) return;

#ifdef SKEPU_MPI_DEBUG
        if (!cluster::mpi_rank())
            std::cout << "<<<[ " << this->getName() << " is gathering to root. ]>>>\n";
#endif

        skepu::cluster::gather_to_root_inplace(
            this->m_data,
            this->partition.byte_counts,
            this->partition.byte_displs,
            &this->m_data[this->part_begin()]
        );
    }

    template<typename T>
    void Matrix<T>::scatter_from_root()
    {
        this->dirty = true;

#ifdef SKEPU_MPI_DEBUG
        if (!cluster::mpi_rank())
            std::cout << "<<<[ " << this->getName() << " is scattering from root. ]>>>\n";
#endif

        skepu::cluster::scatter_from_root_inplace(
            this->m_data,
            this->partition.byte_counts,
            this->partition.byte_displs,
            &this->m_data[this->part_begin()]
        );
    }

    template<typename T>
    void Matrix<T>::flush_MPI()
    {
        this->allgather();
    }
}


#endif // SKEPU_MPI