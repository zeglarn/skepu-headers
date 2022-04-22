#pragma once

#include <memory>
#include <cassert>
#include <mpi.h>

namespace skepu
{
    namespace cluster
    {
        namespace state
        {
            struct internal_state
            {
                int rank;
                int num_ranks;
                int *counts;
                int *displs;

                internal_state()
                {
                    auto status = MPI_Init(NULL, NULL);
                    assert(status == MPI_SUCCESS);
                    status = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
                    assert(status == MPI_SUCCESS);
                    status = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                    assert(status == MPI_SUCCESS);

                    counts = new int[num_ranks];
                    displs = new int[num_ranks];
                }

                ~internal_state()
                {
                    MPI_Finalize();
                    delete [] counts;
                    delete [] displs;
                }
            };


            inline internal_state *s() {
                bool static initialized{true};
                auto deleter = [&](internal_state *ptr)
                {
                    initialized = false;
                    delete ptr;
                };

                std::unique_ptr<internal_state, decltype(deleter)> static g_state
                {
                    new internal_state, deleter
                };

                if(initialized)
                    return g_state.get();
                return 0;
            }
        } // namespace state

        static int mpi_rank() { return state::s()->rank; }
        static int mpi_size() { return state::s()->num_ranks; }

        template<typename T>
        struct Partition
        {
            int rank, num_ranks, typesize;
            int *counts{};
            int *displs{};
            int *byte_counts{};
            int *byte_displs{};

            Partition()
            {
                typesize = sizeof(T);
                num_ranks = skepu::cluster::mpi_size();
                rank = skepu::cluster::mpi_rank();
                counts = new int[num_ranks];
                displs = new int[num_ranks];
                byte_counts = new int[num_ranks];
                byte_displs = new int[num_ranks];
            }

            ~Partition()
            {
                delete [] counts;
                delete [] displs;
                delete [] byte_counts;
                delete [] byte_displs;
            }

            void prepare(size_t major_len, size_t minor_len)
            {                    
                size_t begin, end, part_size, rem;

                part_size = major_len / num_ranks;
                rem = major_len % num_ranks;

                for (size_t i = 0; i < num_ranks; i++)
                {
                    begin = part_size * i + (i < rem ? i : rem);
                    end = begin + part_size + (i < rem ? 1 : 0);

                    displs[i] = begin * minor_len;
                    counts[i] = (end - begin) * minor_len;

                    byte_displs[i] = displs[i] * typesize;
                    byte_counts[i] = counts[i] * typesize;
                }
            }

            size_t part_begin()
            {
                return displs[rank];
            }

            size_t part_end() 
            { 
                return displs[rank] + counts[rank]; 
            }
        };
        


        static void barrier()
        {
            if ( state::s() )
                MPI_Barrier(MPI_COMM_WORLD);
        }

        static void gather(
            const void *sendbuf,
            int sendcount,
            void *recvbuf,
            int recvcount,
            int root
        )
        {
            if ( state::s() )
                MPI_Gather(
                    sendbuf,
                    sendcount,
                    MPI_BYTE,
                    recvbuf,
                    recvcount,
                    MPI_BYTE,
                    root,
                    MPI_COMM_WORLD
                );
        }


        static void gatherv(
            const void *sendbuf,
            int sendcount,
            void *recvbuf,
            const int *recvcounts,
            const int *displacements,
            int root
        )
        {
            if ( state::s() )
                MPI_Gatherv(
                    sendbuf,
                    sendcount,
                    MPI_BYTE,
                    recvbuf,
                    recvcounts,
                    displacements,
                    MPI_BYTE,
                    root,
                    MPI_COMM_WORLD
                );
        }

        static void scatter(
            const void *sendbuf,
            int sendcount,
            void *recvbuf,
            int recvcount,
            int root
        )
        {
            if ( state::s() )
                MPI_Scatter(
                    sendbuf,
                    sendcount,
                    MPI_BYTE,
                    recvbuf,
                    recvcount,
                    MPI_BYTE,
                    root,
                    MPI_COMM_WORLD
                );
        }

        static void scatterv(
            const void *sendbuf,
            const int *sendcounts,
            const int *displacements,
            void *recvbuf,
            int recvcount,
            int root
        )
        {
            if ( state::s() )
                MPI_Scatterv(
                    sendbuf,
                    sendcounts,
                    displacements,
                    MPI_BYTE,
                    recvbuf,
                    recvcount,
                    MPI_BYTE,
                    root,
                    MPI_COMM_WORLD
                );
        }

        static void allgather(
            const void *sendbuf,
            int sendcount,
            void *recvbuf,
            int recvcount
        )
        {
            if ( state::s() )
                MPI_Allgather(
                    sendbuf,
                    sendcount,
                    MPI_BYTE,
                    recvbuf,
                    recvcount,
                    MPI_BYTE,
                    MPI_COMM_WORLD
                );
        }

        static void allgatherv(
            const void *sendbuf,
            int sendcount,
            void *recvbuf,
            const int *recvcounts,
            const int *displacements
        )
        {
            if ( state::s() )
                MPI_Allgatherv(
                    sendbuf,
                    sendcount,
                    MPI_BYTE,
                    recvbuf,
                    recvcounts,
                    displacements,
                    MPI_BYTE,
                    MPI_COMM_WORLD
                );
        }

        static void allgatherv_inplace(
            void *recvbuf,
            int *recvcounts,
            int *displacements)
        {
            MPI_Allgatherv(
                MPI_IN_PLACE,
                0,
                MPI_DATATYPE_NULL,
                recvbuf,
                recvcounts,
                displacements,
                MPI_BYTE,
                MPI_COMM_WORLD);
        }

        static void gather_to_root_inplace(
            void *rootbuffer,
            int *counts,
            int *displs,
            const void *sendbuffer
        )
        {
            int rank = mpi_rank();
            if (!rank)
            {
                MPI_Gatherv(
                    MPI_IN_PLACE,
                    0,
                    MPI_DATATYPE_NULL,
                    rootbuffer,
                    counts,
                    displs,
                    MPI_BYTE,
                    0,
                    MPI_COMM_WORLD);

            }
            else
            {
                MPI_Gatherv(
                    sendbuffer,
                    counts[rank],
                    MPI_BYTE,
                    rootbuffer,
                    counts,
                    displs,
                    MPI_BYTE,
                    0,
                    MPI_COMM_WORLD);
            }


        }

        static void scatter_from_root_inplace(
            const void *rootbuffer,
            const int *counts,
            const int *displs,
            void *recvbuffer
            )
        {
            int rank = mpi_rank();
            if (!rank)
            {
                MPI_Scatterv(
                    rootbuffer,
                    counts,
                    displs,
                    MPI_BYTE,
                    MPI_IN_PLACE,
                    0,
                    MPI_BYTE,
                    0,
                    MPI_COMM_WORLD
                );
            }
            else
            {
                MPI_Scatterv(
                    rootbuffer,
                    counts,
                    displs,
                    MPI_BYTE,
                    recvbuffer,
                    counts[rank],
                    MPI_BYTE,
                    0,
                    MPI_COMM_WORLD
                );
            }
        }

        // If there is a possibility of reading across partitions, all
        // processes get full copy.

        template<typename Container_t, typename ProxyTag>
        void handle_container_arg(Container_t & c, ProxyTag)
        {

            c.allgather();
            
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
    } // namespace cluster
    
} // namespace skepu
