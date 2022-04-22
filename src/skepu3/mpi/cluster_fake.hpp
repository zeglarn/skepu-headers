#pragma once

#include <cstddef>
#include <vector>
// Mockup of the cluster interface, to allow the SkePU precompiler to
// compile the code.

namespace skepu
{
	namespace cluster
	{
		static size_t mpi_rank() { return 0; }
        static size_t mpi_size() { return 0; }

        static void barrier()
        { }

        static void gather(const void *,int,void *,int,int)
        { }


        static void gatherv(const void *,int,void *,const int *,const int *,int)
        { }

        static void scatter(const void *,int,void *,int,int)
        { }

        static void scatterv(const void *,const int *,const int *,void *,int,int)
        { }

        static void allgather(const void *,int,void *,int)
        { }

        static void allgatherv(const void *,int,void *,const int *,const int *)
        { }
	}
}
