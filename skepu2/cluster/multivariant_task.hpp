#ifndef MULTIVARIANT_TASK_HPP
#define MULTIVARIANT_TASK_HPP

#include <cstddef>
#include <starpu.h>
#include <tuple>
#include <skepu2/cluster/index.hpp>
#include <skepu2/impl/meta_helpers.hpp>
#include <skepu2/cluster/container_cut.hpp>
#include <skepu2/cluster/mat.hpp>
#include <skepu2/cluster/starpu_var.hpp>
#include <skepu2/cluster/reduce_mode.hpp>

namespace skepu2
{
	namespace cluster
	{
		/**
		 * This class encapsulates some data access patterns common to all
		 * skeletons, using the curiously
		 * recccuring template pattern. This places some requirements on
		 * its subclasses.
		 *
		 * Each parameter is a tuple of the underlying datatype of each
		 * desired buffer.
		 *
		 * @param ResultArgs    // STARPU_W
		 * @param ElwiseArgs    // STARPU_R
		 * @param ContainerArgs // STARPU_R
		 * @param UniformArgs
		 *
		 */
		template<typename ResultArgs,
		         typename ElwiseArgs,
		         typename ContainerArgs,
		         typename UniformArgs,
		         typename Self>
		class multivariant_task
		{
		private:
			using HandleT =  decltype(std::tuple_cat(ResultArgs{},
			                                         ElwiseArgs{},
			                                         ContainerArgs{}));

			static constexpr size_t n_result =
				std::tuple_size<ResultArgs>::value;

			static constexpr size_t n_elwise =
				std::tuple_size<ElwiseArgs>::value;

			static constexpr size_t n_container =
				std::tuple_size<ContainerArgs>::value;

			static constexpr size_t n_uniform =
				std::tuple_size<UniformArgs>::value;

			static constexpr size_t n_handles =
				n_result + n_elwise + n_container;

			static constexpr typename
			make_pack_indices<n_handles, 0>::type handle_indices{};

			static constexpr typename
			make_pack_indices<n_result, 0>::type result_handle_indices{};

			static constexpr typename
			make_pack_indices<n_result+n_elwise, n_result>::type
			elwise_handle_indices{};

			static constexpr typename
			make_pack_indices<n_result+n_elwise+n_container, n_result+n_elwise>::type
			container_handle_indices{};

			static constexpr typename make_pack_indices<n_result, 0>::type ri{};
			static constexpr typename make_pack_indices<n_elwise, 0>::type ei{};
			static constexpr typename make_pack_indices<n_container, 0>::type ci{};
			static constexpr typename make_pack_indices<n_uniform, 0>::type ui{};

			static constexpr size_t n_arg = n_result+n_elwise+n_container+n_uniform;
			static constexpr typename
			make_pack_indices<n_arg, n_arg - n_uniform>::type uniform_indices{};

			struct task_data
			{
				Size2D size;
				Index2D index;
				helpers::starpu_handle_offsets offsets;
				void* self;
			};


			static void cpu_starpu_func(void** buffers, void* args);

			template<size_t... HI,
			         size_t... RI,
			         size_t... EI,
			         size_t... CI,
			         size_t... UI>
			static void cpu_starpu_redirect(void** buffers,
			                                void* args,
			                                pack_indices<HI...>,
			                                pack_indices<RI...>,
			                                pack_indices<EI...>,
			                                pack_indices<CI...>,
			                                pack_indices<UI...>);

		public:
			multivariant_task();

			template<typename... Args,
			         size_t... RI,
			         size_t... EI,
			         size_t... CI,
			         size_t... UI>
			void element_aligned_impl(pack_indices<RI...>,
			                          pack_indices<EI...>,
			                          pack_indices<CI...>,
			                          pack_indices<UI...>,
			                          Size2D size,
			                          Args& ...args);

			template<typename... Args>
			void element_aligned(Size2D size, Args & ...args);

			template<typename... Args,
			         size_t... RI,
			         size_t... EI,
			         size_t... CI,
			         size_t... UI>
			void
			element_aligned_res_per_block_impl(pack_indices<RI...>,
			                                   pack_indices<EI...>,
			                                   pack_indices<CI...>,
			                                   pack_indices<UI...>,
			                                   Size2D size,
			                                   Args& ...args);

			template<typename... Args>
			void
			element_aligned_res_per_block(Size2D size,
			                              Args & ...args);


			template<typename... Args,
			         size_t... RI,
			         size_t... EI,
			         size_t... CI,
			         size_t... UI>
			void element_aligned_sweep_impl(pack_indices<RI...>,
			                                pack_indices<EI...>,
			                                pack_indices<CI...>,
			                                pack_indices<UI...>,
			                                Size2D size,
			                                const skepu2::SweepMode dir,
			                                Args& ...args);

			template<typename... Args>
			void element_aligned_sweep(Size2D size,
			                           const skepu2::SweepMode dir,
			                           Args & ...args);

		public:
			starpu_codelet* cl;
			struct starpu_perfmodel* perf_model;
		};
	} // cluster
} // skepu2

#include <skepu2/cluster/impl/multivariant_task.inl>

#endif /* MULTIVARIANT_TASK_HPP */
