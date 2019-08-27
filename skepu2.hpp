#pragma once

#if defined(SKEPU_OPENMP) && (defined(SKEPU_CUDA) || defined(SKEPU_OPENCL))
# define SKEPU_HYBRID

# if defined(SKEPU_CUDA) && !(defined(SKEPU_OPENCL) && defined(SKEPU_HYBRID_FORCE_OPENCL))
#  define SKEPU_HYBRID_USE_CUDA
# endif
#endif

#include "skepu2/impl/backend.hpp"
#include "skepu2/backend/helper_methods.h"
#include "skepu2/impl/common.hpp"
#include "skepu2/impl/timer.hpp"
#include "skepu2/backend/tuner.h"
#include "skepu2/backend/hybrid_tuner.h"

#ifndef SKEPU_PRECOMPILED

#include "skepu2/map.hpp"
#include "skepu2/reduce.hpp"
#include "skepu2/scan.hpp"
#include "skepu2/mapoverlap.hpp"
#include "skepu2/mapreduce.hpp"
#include "skepu2/call.hpp"

#else

#include "skepu2/backend/skeleton_base.h"
#include "skepu2/backend/map.h"
#include "skepu2/backend/reduce.h"
#include "skepu2/backend/mapreduce.h"
#include "skepu2/backend/scan.h"
#include "skepu2/backend/mapoverlap.h"
#include "skepu2/backend/call.h"

#endif // SKEPU_PRECOMPILED
