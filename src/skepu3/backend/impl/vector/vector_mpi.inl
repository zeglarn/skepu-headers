/*! \file vector_mpi.inl
 *  \brief Contains the definitions of mpi specific member functions for the Vector container.
 */

#ifdef SKEPU_MPI

namespace skepu
{
    /*
    Row-wise partitions.
    */
    template<typename T>
    void Vector<T>::partition_prepare()
    {
        this->partition.prepare(this->m_size, 1);
    }

    template<typename T>
    void Vector<T>::partition_prepare(size_t size)
    {
        this->partition.prepare(size, 1);
    }

    template<typename T>
    size_t Vector<T>::part_begin()
    {
        return this->partition.part_begin();
    }

    template<typename T>
    size_t Vector<T>::part_end()
    {
        return this->partition.part_end();
    }

    template<typename T>
    size_t Vector<T>::part_size()
    {
        return this->partition.part_end() - this->partition.part_begin();
    }

    template<typename T>
    void Vector<T>::allgather()
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
    void Vector<T>::gather_to_root()
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
    void Vector<T>::scatter_from_root()
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
    void Vector<T>::flush_MPI()
    {
        this->allgather();
    }

    template<typename T>
    void Vector<T>::mark_dirty()
    {
        this->dirty = true;
    }

    template<typename T>
    void Vector<T>::mark_clean()
    {
        this->dirty = false;
    }

    template<typename T>
    bool Vector<T>::is_dirty() const
    {
        return this->dirty;
    }
}


#endif // SKEPU_MPI