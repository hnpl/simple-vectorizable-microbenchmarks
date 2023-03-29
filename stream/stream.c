#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>

#ifndef ARRAY_SIZE
#    define ARRAY_SIZE 1000000
#endif

#ifdef NTHREADS
extern void omp_set_num_threads(int);
#endif

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

static TElement a[ARRAY_SIZE], b[ARRAY_SIZE], c[ARRAY_SIZE];
static TElement expected_a, expected_b, expected_c;
static double avgtime[4] = {0, 0, 0, 0},
              maxtime[4] = {0, 0, 0, 0},
              mintime[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
static char *label[4] = {"Copy", "Scale", "Add", "Triad"};
static double t_copy = 0.0;
static int error_count_a = 0;
static double max_error_a = 0.0;
static double min_error_a = 0.0;
static int error_count_b = 0;
static double max_error_b = 0.0;
static double min_error_b = 0.0;
static int error_count_c = 0;
static double max_error_c = 0.0;
static double min_error_c = 0.0;
// functions
double get_second();
void array_init(TElement*, TElement);
void array_verify(TElement*, TElement, int*, TElement*, TElement*);
double do_copy(TElement*, TElement*);
void report();

// main
int main()
{
#ifdef NTHREADS
    omp_set_num_threads(NTHREADS);
#endif

    array_init(a, 1.0);
    array_init(b, 2.0);
    array_init(c, 0.0);

    // warm up
    do_copy(c, a);

    t_copy = do_copy(c, a);

    expected_a = 1.0;
    expected_b = 2.0;
    expected_c = 1.0;

    array_verify(a, expected_a, &error_count_a, &min_error_a, &max_error_a);
    array_verify(b, expected_b, &error_count_b, &min_error_b, &max_error_b);
    array_verify(c, expected_c, &error_count_c, &min_error_c, &max_error_c);

    report();
}

double get_second()
{
    struct timeval tp;
    struct timezone tzp;
    int i;
    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void __attribute__((optimize("no-tree-vectorize")))
array_init(TElement* arr, TElement v)
{
    for (int k = 0; k < ARRAY_SIZE; k++)
        arr[k] = v;
}

void __attribute__((optimize("no-tree-vectorize")))
array_verify(TElement* arr, TElement expected_value,
             int* error_count, TElement* min_error, TElement* max_error)

{
    *error_count = 0;
    *min_error = DBL_MAX;
    *max_error = 0.0;

    for (int k = 0; k < ARRAY_SIZE; k++)
    {
        if (arr[k] != expected_value)
        {
            *error_count += 1;
            TElement diff = abs(arr[k] - expected_value);
            *min_error = min(diff, *min_error);
            *max_error = max(diff, *max_error);
        }
    }
}

void report()
{
    double data_size_bytes = ARRAY_SIZE * sizeof(TElement);
    double data_size_GiB = data_size_bytes / 1024.0 / 1024.0 / 1024.0;
    double copy_bandwidth = 2.0 * data_size_GiB / t_copy;
    printf("Copy\n");
    printf("Bandwidth: %f GiB/s\n", copy_bandwidth);
    printf("Time: %f s\n", t_copy);
    printf("array_a error_count: %d, max_error: %f\n", error_count_a, max_error_a);
    printf("array_b error_count: %d, max_error: %f\n", error_count_b, max_error_b);
    printf("array_c error_count: %d, max_error: %f\n", error_count_c, max_error_c);
}

//double do_copy(double* __restrict__ dst, double* __restrict__ src)
double do_copy(double* dst, double* src)
{
    double t_start = get_second();
#pragma omp parallel for
    for (int k = 0; k < ARRAY_SIZE; k++)
        dst[k] = src[k];
    return get_second() - t_start;
}
