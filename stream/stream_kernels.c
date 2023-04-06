#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>

typedef double TElement;

double get_second()
{
    struct timeval tp;
    struct timezone tzp;
    int i;
    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


double do_copy(TElement* restrict dst, TElement* restrict src, const size_t array_size)
{
    double t_start = get_second();
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < array_size; k++)
        dst[k] = src[k];
    return get_second() - t_start;
}

double do_scale(TElement* restrict dst, TElement* restrict src, const TElement scale_factor, const size_t array_size)
{
    double t_start = get_second();
    #pragma omp parallel for
    for (size_t k = 0; k < array_size; k++)
        dst[k] = scale_factor * src[k];
    return get_second() - t_start;
}

double do_add(TElement* restrict dst, TElement* restrict src1, TElement* restrict src2, const size_t array_size)
{
    double t_start = get_second();
    #pragma omp parallel for
    for (size_t k = 0; k < array_size; k++)
        dst[k] = src1[k] + src2[k];
    return get_second() - t_start;
}

double do_triad(TElement* restrict dst, TElement* restrict src1, TElement* restrict src2, const TElement scale_factor, const size_t array_size)
{
    double t_start = get_second();
    #pragma omp parallel for
    for (size_t k = 0; k < array_size; k++)
        dst[k] = src1[k] + scale_factor * src2[k];
    return get_second() - t_start;
}
