#include <stdint.h>
#include <stddef.h>

typedef double TElement;
typedef uint64_t TIndex;

void gather(TElement* __restrict__ dst, TElement* __restrict__ src, const TIndex* __restrict__ indices, const size_t array_size) {
    #pragma omp parallel for simd
    for (size_t idx = 0; idx < array_size; idx++)
        dst[idx] = src[indices[idx]];
}

void scatter(TElement* __restrict__ dst, TElement* __restrict__ src, const TIndex* __restrict__ indices, const size_t array_size) {
    #pragma omp parallel for simd
    for (size_t idx = 0; idx < array_size; idx++)
        dst[indices[idx]] = src[idx];
}
