#include<stdint.h>
#include<vector>
#include<cassert>
#include<chrono>
#include<iostream>

#ifdef GEM5_ANNOTATION
#include <gem5/m5ops.h>
#endif

typedef uint64_t TElement;
typedef uint64_t TIndex;

extern "C" void gather(TElement* __restrict__ dst, TElement* __restrict__ src, const TIndex* __restrict__ indices, const size_t array_size);
size_t get_num_omp_threads();

class IndexGenerator
{
    private:
        TIndex seed;
        TIndex mod;
        TIndex next_index;
    public:
        IndexGenerator()
        {
        }
        IndexGenerator(TIndex seed, TIndex mod)
        {
            // To have a sparse distribution of the indices, `mod` should be a prime number
            this->seed = seed;
            this->mod = mod;
            this->next_index = seed;
        }
        void reset()
        {
            this->next_index = this->seed;
        }
        TIndex next()
        {
            TIndex curr_index = this->next_index;
            this->next_index *= seed;
            this->next_index %= mod;
            return curr_index;
        }
};

int main()
{
    const int SIZE = 10000018;
    const int N_INDEX = 10000018;

    const size_t num_threads = get_num_omp_threads();
    std::cout << "Number of threads: " << num_threads << std::endl;

    std::vector<TElement> src(SIZE, 0);
    std::vector<TElement> dst(SIZE, 0);
    std::vector<TIndex> indices(N_INDEX, 0);

    // Initialize the arrays
    for (TIndex i = 0; i < SIZE; i++)
        src[i] = i;

    // Creating index
    const TElement seed = 31;
    const TElement mod = 10000019;
    // This index generator will essentially perform a permutation of the src to the dst
    IndexGenerator rng(seed, mod);
    for (TIndex i = 0; i < N_INDEX; i++)
        indices[i] = rng.next()-1;

#ifdef GEM5_ANNOTATION
    m5_work_begin(0, 0);
#endif
    const auto t_start = std::chrono::steady_clock::now();
    // Performing indexed-loads
    gather(dst.data(), src.data(), indices.data(), indices.size());
    const auto t_end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);

#ifdef GEM5_ANNOTATION
    m5_work_end(0, 0);
#endif

    // https://stackoverflow.com/questions/57538507/how-to-convert-stdchronoduration-to-double-seconds
    using namespace std::literals::chrono_literals;
    double time = elapsed_time / 1.0s; // converting from duration to double
    double data_size = 1.0 * SIZE * sizeof(TElement) * 2;
    double bandwidth_bytes = data_size / time;
    double bandwidth_GiB = bandwidth_bytes / 1024.0 / 1024.0 / 1024.0;

    std::cout << "Elapsed_time: " << time << " s" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_GiB << " GiB/s" << std::endl;

    // Checking result
    rng.reset();
    for (TIndex i = 0; i < N_INDEX; i++)
        assert(dst[i] == rng.next()-1);

    return 0;
}

size_t get_num_omp_threads()
{
    size_t num_threads = 0;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp atomic
    num_threads++;
#endif
    return num_threads;
}

