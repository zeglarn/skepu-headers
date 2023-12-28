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
    size_t Matrix<T>::part_begin() const
    {
        return this->partition.part_begin();
    }

    template<typename T>
    size_t Matrix<T>::part_end() const
    {
        return this->partition.part_end();
    }

    template<typename T>
    size_t Matrix<T>::part_size() const
    {
        return this->partition.part_end() - this->partition.part_begin();
    }

    template<typename T>
    void Matrix<T>::allgather()
    {
        if (!this->dirty) return;

        this->mark_clean();
        
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
        this->mark_dirty();

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

    template<typename T>
    void Matrix<T>::mark_dirty()
    {
        this->dirty = true;
    }

    template<typename T>
    void Matrix<T>::mark_clean()
    {
        this->dirty = false;
    }

    template<typename T>
    bool Matrix<T>::is_dirty() const
    {
        return this->dirty;
    }


}


#endif // SKEPU_MPI