#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>

#include<vector>
#include<iostream>
#include<cstdlib>

// https://stackoverflow.com/questions/3437404/min-and-max-in-c
#define max(a,b) \
	({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
	({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

typedef double TElement;

static TElement expected_a, expected_b, expected_c;
static double t_copy = 0.0;
static double t_scale = 0.0;
static double t_add = 0.0;
static double t_triad = 0.0;
static int error_count_a = 0;
static double max_error_a = 0.0;
static double min_error_a = 0.0;
static int error_count_b = 0;
static double max_error_b = 0.0;
static double min_error_b = 0.0;
static int error_count_c = 0;
static double max_error_c = 0.0;
static double min_error_c = 0.0;

#define ARRAY_SIZE 10000000

// functions
void array_verify(const std::vector<TElement>&, TElement, int*, TElement*, TElement*);
void report(const size_t&);

// external functions
extern "C" double do_copy(TElement* __restrict__ dst, TElement* __restrict__ src, const size_t array_size);
extern "C" double do_scale(TElement* __restrict__ dst, TElement* __restrict__ src, const TElement scale_factor, const size_t array_size);
extern "C" double do_add(TElement* __restrict__ dst, TElement* __restrict__ src1, TElement* __restrict__ src2, const size_t array_size);
extern "C" double do_triad(TElement* __restrict__ dst, TElement* __restrict__ src1, TElement* __restrict__src2, const TElement scale_factor, const size_t array_size);

// interfacing to the external functions
double copy(std::vector<TElement>& dst, std::vector<TElement>& src);
double scale(std::vector<TElement>& dst, std::vector<TElement>& src, const TElement& scale_factor);
double add(std::vector<TElement>& dst, std::vector<TElement>& src1, std::vector<TElement>& src2);
double triad(std::vector<TElement>& dst, std::vector<TElement>& src1, std::vector<TElement>& src2, const TElement& scale_factor);

// main
int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <array_size>" << std::endl;
        return 1;
    }
    size_t array_size = atoi(argv[1]);
    static std::vector<TElement> a(array_size, 1);
    static std::vector<TElement> b(array_size, 2);
    static std::vector<TElement> c(array_size, 0);
    static const TElement scale_factor = 3.0;

    // warm up
    copy(c, a);
    scale(b, c, scale_factor);
    add(c, a, b);
    triad(a, b, c, scale_factor);

    // 1 iteration
    t_copy = copy(c, a);
    t_scale = scale(b, c, scale_factor);
    t_add = add(c, a, b);
    t_triad = triad(a, b, c, scale_factor);

    expected_a = 1.0;
    expected_b = 2.0;
    expected_c = 0.0;
    for (size_t iter = 0; iter < 2; iter++)
    {
        expected_c = expected_a;
        expected_b = expected_c * scale_factor;
        expected_c = expected_a + expected_b;
        expected_a = expected_b + expected_c * scale_factor;
    }

    array_verify(a, expected_a, &error_count_a, &min_error_a, &max_error_a);
    array_verify(b, expected_b, &error_count_b, &min_error_b, &max_error_b);
    array_verify(c, expected_c, &error_count_c, &min_error_c, &max_error_c);

    report(array_size);
}

void
array_verify(const std::vector<TElement>& arr, TElement expected_value,
             int* error_count, TElement* min_error, TElement* max_error)

{
    *error_count = 0;
    *min_error = DBL_MAX;
    *max_error = 0.0;

    for (auto const & element: arr)
    {
        if (element != expected_value)
        {
            *error_count += 1;
            TElement diff = abs(element - expected_value);
            *min_error = min(diff, *min_error);
            *max_error = max(diff, *max_error);
        }
    }
}

double copy(std::vector<TElement>& dst, std::vector<TElement>& src)
{
    return do_copy(dst.data(), src.data(), dst.size());
}
double scale(std::vector<TElement>& dst, std::vector<TElement>& src, const TElement& scale_factor)
{
    return do_scale(dst.data(), src.data(), scale_factor, dst.size());
}
double add(std::vector<TElement>& dst, std::vector<TElement>& src1, std::vector<TElement>& src2)
{
    return do_add(dst.data(), src1.data(), src2.data(), dst.size());
}
double triad(std::vector<TElement>& dst, std::vector<TElement>& src1, std::vector<TElement>& src2, const TElement& scale_factor)
{
    return do_triad(dst.data(), src1.data(), src2.data(), scale_factor, dst.size());
}


void report(const size_t& array_size)
{
    double data_size_bytes = array_size * sizeof(TElement);
    double data_size_GiB = data_size_bytes / 1024.0 / 1024.0 / 1024.0;
    double copy_bandwidth = 2.0 * data_size_GiB / t_copy;
    double scale_bandwidth = 2.0 * data_size_GiB / t_scale;
    double add_bandwidth = 3.0 * data_size_GiB / t_add;
    double triad_bandwidth = 3.0 * data_size_GiB / t_triad;
    printf("Each array size: %f GiB\n", data_size_GiB);
    printf("Copy\n");
    printf("Bandwidth: %f GiB/s\n", copy_bandwidth);
    printf("Time: %f s\n", t_copy);
    printf("Scale\n");
    printf("Bandwidth: %f GiB/s\n", scale_bandwidth);
    printf("Time: %f s\n", t_scale);
    printf("Add\n");
    printf("Bandwidth: %f GiB/s\n", add_bandwidth);
    printf("Time: %f s\n", t_add);
    printf("Triad\n");
    printf("Bandwidth: %f GiB/s\n", triad_bandwidth);
    printf("Time: %f s\n", t_triad);
    printf("array_a error_count: %d, max_error: %f\n", error_count_a, max_error_a);
    printf("array_b error_count: %d, max_error: %f\n", error_count_b, max_error_b);
    printf("array_c error_count: %d, max_error: %f\n", error_count_c, max_error_c);
}
