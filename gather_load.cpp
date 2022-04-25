#include<iostream>
#include<cassert>

void gather_load(const uint64_t* src, uint64_t* index, uint64_t* dst, const uint64_t& n_index)
{
    for (uint64_t i = 0; i < n_index; i++)
        dst[i] = src[index[i]];
}

int main()
{
    const int SIZE = 100003;
    const int N_INDEX = 100000;

    uint64_t src[SIZE];
    uint64_t dst[SIZE];
    uint64_t index[N_INDEX];

    // Initialize the arrays
    for (uint64_t i = 0; i < SIZE; i++)
        src[i] = i;
    for (uint64_t i = 0; i < SIZE; i++)
        dst[i] = 0;
    for (uint64_t i = 0; i < N_INDEX; i++)
        index[i] = 0;

    // Creating index
    volatile const uint64_t seed = 3;
    const uint64_t mod = 100003;
    uint64_t w = seed;
    for (uint64_t i = 0; i < N_INDEX; i++)
    {
        index[i] = w;
        w *= seed;
        w %= mod;
    }

    // Performing indexed-loads
    gather_load(src, index, dst, N_INDEX);

    // Checking
    w = seed;
    for (uint64_t i = 0; i < N_INDEX; i++)
    {
        assert(dst[i] == w);
        w *= seed;
        w %= mod;
    }

    return 0;
}

