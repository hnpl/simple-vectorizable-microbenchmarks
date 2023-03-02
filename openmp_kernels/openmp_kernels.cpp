#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <iterator>
#include <cassert>
#include <cstdint>

typedef uint64_t TElement;
typedef uint64_t TIndex;

void gather(TElement* __restrict__ dst, TElement* __restrict__ src, TIndex* __restrict__ indices, const size_t& array_size) {
    #pragma omp parallel
    {
        #pragma omp simd
        for (size_t idx = 0; idx < array_size; idx++) {
            dst[idx] = src[indices[idx]];
        }
    }
}

void scatter(TElement* __restrict__ dst, TElement* __restrict__ src, TIndex* __restrict__ indices, const size_t& array_size) {
    #pragma omp parallel
    {
        #pragma omp simd
        for (size_t idx = 0; idx < array_size; idx++) {
            dst[indices[idx]] = src[idx];
        }
    }
}

void read_inputs(std::vector<TElement>& dst, std::vector<TElement>& src, std::vector<TIndex>& indices) {
    std::ifstream src_file("src_file.txt");
    std::istream_iterator<TElement> src_start(src_file), src_end;
    src = std::vector<TElement>(src_start, src_end);
    src_file.close();

    dst.resize(src.size());

    std::ifstream idx_file("idx_file.txt");
    std::istream_iterator<TIndex> idx_start(idx_file), idx_end;
    indices = std::vector<TIndex>(idx_start, idx_end);
    idx_file.close();
}

void verify_gather(const std::vector<TElement>& dst, const std::vector<TElement>& src, const std::vector<TIndex>& indices) {
    for (size_t idx = 0; idx < indices.size(); idx++) {
        assert(dst[idx] == src[indices[idx]]);
    }
}

int main() {
    std::vector<TElement> dst, src;
    std::vector<TIndex> indices;
    read_inputs(dst, src, indices);
    gather(dst.data(), src.data(), indices.data(), indices.size());
    verify_gather(dst, src, indices);
    return 0;
}
