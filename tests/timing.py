from timeit import timeit


def time_function(func, *args, reps=10):
    """
    Passes *args into a function, func, and times it reps times, returns the average time in milliseconds (ms).
    """

    avg_time = timeit(lambda: func(*args), number=reps) / reps

    return avg_time * 1000
