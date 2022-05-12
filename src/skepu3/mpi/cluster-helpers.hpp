#pragma once

namespace skepu
{
    namespace cluster
    {
        // If there is a possibility of reading across partitions, all
                // processes get full copy.

        // template<typename Container_t, typename ProxyTag>
        // void handle_container_arg(Container_t & c, ProxyTag) noexcept
        // {
        //     c.allgather();
        // }

        template<typename Container, typename ProxyTag>
        void handle_container_arg(Container & c, ProxyTag) noexcept
        {
            c.allgather();
        }


        template<typename Container>
        void handle_container_arg(Container, ProxyTag::MatRow)
        { }

        template<typename Container>
        void handle_read_write_access(Container &c, AccessMode mode)
        {
            if (hasWriteAccess(mode))
                c.mark_dirty();
        }





        template<typename Container>
        void gather_to_root(Container & c)
        {
            c.getParent().gather_to_root();
        }

        template<typename Container>
        void scatter_from_root(Container & c)
        {
            c.getParent().scatter_from_root();
        }

        // Else each process gets only its local partition to work
        // with.

        // template<typename Container_t>
        // void handle_container_arg(Container_t & c, skepu::ProxyTag::MatRow)
        // {
        //     c.partition();
        // }

    }
}
