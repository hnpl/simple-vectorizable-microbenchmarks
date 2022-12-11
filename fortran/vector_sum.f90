REAL FUNCTION calc_sum(arr) RESULT(total)
    REAL, INTENT(IN) :: arr(1000000)
    total = sum(arr)
END FUNCTION calc_sum

PROGRAM vector_sum
    REAL :: arr(1000000)
    REAL :: total = 0
    CALL random_number(arr)
    total = calc_sum(arr)
    print *, total
END PROGRAM
