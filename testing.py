import time
import random
from itertools import accumulate
from functools import reduce
from operator import mul, add

n_tests = 100
n_iters = 10000
n_vars = 20

n_iters_r = range(n_iters)


# for mutliplying each element of two lists and then summing

fastest_msum = {}

for i in range(n_tests):
    a = [random.uniform(0, 1) for i in n_iters_r]
    b = [random.uniform(0, 1) for i in n_iters_r]
    c = random.uniform(0, 1)

    times = {}

    t = time.time()

    c = 0.0
    for j in n_iters_r:
        c += a[j] * b[j]

    times["for loop"] = time.time() - t

    t = time.time()

    c = sum([x * y for x, y in zip(a, b)])

    times["sum, list comp"] = time.time() - t

    t = time.time()

    c = sum(map(mul, a, b))

    times["sum, map"] = time.time() - t

    if i == 0:
        for key in times:
            fastest_msum[key] = 0
    else:
        min_time = float('inf')
        min_key = ""
        for key in times:
            if times[key] < min_time:
                min_key = key
                min_time = times[key]
        fastest_msum[min_key] += 1

print(fastest_msum)


# for summing a list:
# sum() vs reduce() vs itertools.accumulate()?

fastest_sum = {}

for i in range(n_tests):
    a = [random.uniform(0, 1) for i in n_iters_r]
    c = random.uniform(0, 1)

    times = {}

    t = time.time()

    c = 0.0
    for j in n_iters_r:
        c += a[j]

    times["for loop"] = time.time() - t

    t = time.time()

    c = sum(a)

    times["sum"] = time.time() - t

    t = time.time()

    c = reduce(add, a)

    times["reduce"] = time.time() - t


    if i == 0:
        for key in times:
            fastest_sum[key] = 0
    else:
        min_time = float('inf')
        min_key = ""
        for key in times:
            if times[key] < min_time:
                min_key = key
                min_time = times[key]
        fastest_sum[min_key] += 1

print(fastest_sum)