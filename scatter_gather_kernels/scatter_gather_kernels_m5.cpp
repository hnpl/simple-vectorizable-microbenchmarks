#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <iterator>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <sys/time.h>

#include "gem5/m5ops.h"

#include "json.hpp"

using json = nlohmann::json;

typedef uint64_t TElement;
typedef uint64_t TIndex;

double get_second() {
    struct timeval tp;
    struct timezone tzp;
    int i;
    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

__attribute__((optimize("tree-vectorize")))
void gather(TElement* __restrict__ dst, TElement* __restrict__ src, const TIndex* __restrict__ indices, const size_t& array_size) {
    #pragma omp parallel
    {
        #pragma omp simd
        for (size_t idx = 0; idx < array_size; idx++) {
            dst[idx] = src[indices[idx]];
        }
    }
}

__attribute__((optimize("tree-vectorize")))
void scatter(TElement* __restrict__ dst, TElement* __restrict__ src, const TIndex* __restrict__ indices, const size_t& array_size) {
    #pragma omp parallel
    {
        #pragma omp simd
        for (size_t idx = 0; idx < array_size; idx++) {
            dst[indices[idx]] = src[idx];
        }
    }
}

enum KernelType { Gather = 0, Scatter };

class ScatterGatherKernel {
  private:
    std::vector<TIndex> index;
    KernelType kernel_type;
    TIndex multiplicity;
  public:
    ScatterGatherKernel() { kernel_type = KernelType::Gather; multiplicity = 0; }
    ScatterGatherKernel(std::vector<TIndex>& index, const KernelType& kernel_type, const TIndex& multiplicity) {
        this->setIndex(index);
        this->setType(kernel_type);
        this->setMultiplicity(multiplicity);
    }
    void setIndex(const std::vector<TIndex>& index) {
        this->index = std::move(index);
    }
    void setType(const KernelType& kernel_type) {
        this->kernel_type = kernel_type;
    }
    void setMultiplicity(const TIndex& multiplicity) {
        this->multiplicity = multiplicity;
    }
    size_t getMaxIndex() const {
        auto it = std::max_element(this->index.begin(), this->index.end());
        return std::max(*it, this->index.size());
    }
    void doPrint() const {
        std::cout << "Kernel" << std::endl;
        std::cout << "  + type: ";
        if (this->kernel_type == KernelType::Gather)
            std::cout << "Gather";
        else
            std::cout << "Scatter";
        std::cout << std::endl;
        std::cout << "  + size: " << this->index.size() << std::endl;
        std::cout << "  + multiplicity: " << this->multiplicity << std::endl;
    }
    void execute(std::vector<TElement>& dst, std::vector<TElement>& src) const {
        std::cout << "Executing ";
        this->doPrint();
        switch (this->kernel_type) {
            case KernelType::Gather:
                for (size_t iter = 0; iter < this->multiplicity; iter++)
                    gather(dst.data(), src.data(), this->index.data(), this->index.size());
                break;
            case KernelType::Scatter:
                for (size_t iter = 0; iter < this->multiplicity; iter++)
                    scatter(dst.data(), src.data(), this->index.data(), this->index.size());
                break;
            default:
                std::cout << "Warn: unregconized kernel type" << std::endl;
                break;
        }
    }
};

std::vector<ScatterGatherKernel> read_spatter_json(const char* filename) {
    std::vector<ScatterGatherKernel> kernels;
    std::ifstream idx_file(filename);
    json j = json::parse(idx_file);
    const size_t numKernels = j.size();
    kernels.reserve(numKernels);
    for (size_t i = 0; i < numKernels; i++) {
        kernels.push_back(ScatterGatherKernel());
        kernels.back().setIndex(j[i]["pattern"]);
        kernels.back().setMultiplicity(j[i]["count"]);
        if (j[i]["kernel"] == "Gather")
            kernels.back().setType(KernelType::Gather);
        else if (j[i]["kernel"] == "Scatter")
            kernels.back().setType(KernelType::Scatter);
    }
    idx_file.close();
    return kernels;
}

void executeKernels(const char* filename) {
    auto kernels = read_spatter_json(filename);
    size_t array_size = 0;
    for (auto const& k: kernels)
        array_size = std::max(array_size, k.getMaxIndex()+1);
    array_size = pow(2, int(log(array_size) / log(2)) + 1);

    std::vector<TElement> src = std::vector<TElement>(array_size, 2);
    std::vector<TElement> dst = std::vector<TElement>(array_size, 3);
    double t_total = 0;
    double t_start = 0;
    double t_end = 0;

    m5_exit(0);
    for (auto const& k: kernels) {
        t_start = get_second();
        k.execute(dst, src);
        t_end = get_second();
        t_total += (t_end - t_start);
        std::cout << "Execute time: " << (t_end - t_start) << " seconds" << std::endl;
    }
    std::cout << "Total Elapsed Time: " << (t_total) << " seconds" << std::endl;
    m5_exit(0);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_json_file>" << std::endl;
        return 1;
    }
    executeKernels(argv[1]);
    return 0;
}
