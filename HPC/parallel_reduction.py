import random
import time
import threading


# ─────────────────────────────────────────────
# Sequential functions
# ─────────────────────────────────────────────
def sum_serial(arr):
    return sum(arr)


def average_serial(arr):
    return sum(arr) // len(arr)


def min_serial(arr):
    return min(arr)


def max_serial(arr):
    return max(arr)


# ─────────────────────────────────────────────
# Parallel reduction helper
# Splits array into chunks, each thread computes
# a partial result, then reduces across threads —
# mirrors OpenMP  reduction(op : var)
# ─────────────────────────────────────────────
def parallel_reduce(arr, func):
    num_threads    = min(8, len(arr))
    chunk_size     = (len(arr) + num_threads - 1) // num_threads
    chunks         = [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]
    partial_results = [None] * len(chunks)

    def worker(idx, chunk):
        partial_results[idx] = func(chunk)

    threads = [threading.Thread(target=worker, args=(i, c))
               for i, c in enumerate(chunks)]
    for t in threads: t.start()
    for t in threads: t.join()

    return func(partial_results)


def sum_parallel(arr):
    return parallel_reduce(arr, sum)


def average_parallel(arr):
    total = parallel_reduce(arr, sum)
    return total // len(arr)


def min_parallel(arr):
    return parallel_reduce(arr, min)


def max_parallel(arr):
    return parallel_reduce(arr, max)


# ─────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────
def timed(fn, *args):
    start  = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


# ─────────────────────────────────────────────
# Input helper
# ─────────────────────────────────────────────
def input_int(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = int(input(prompt))
            if min_val is not None and val < min_val:
                print(f"  Value must be >= {min_val}.")
            elif max_val is not None and val > max_val:
                print(f"  Value must be <= {max_val}.")
            else:
                return val
        except ValueError:
            print("  Invalid input. Please enter an integer.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 45)
    print("   Parallel Reduction: Sum, Avg, Min, Max")
    print("=" * 45)

    N         = input_int("Enter size of array:          ", min_val=1)
    range_min = input_int("Enter minimum value of elements: ")
    range_max = input_int("Enter maximum value of elements: ", min_val=range_min)

    arr = [random.randint(range_min, range_max) for _ in range(N)]
    print(f"\nGenerated array of {N} element(s) in range [{range_min}, {range_max}].\n")

    # ── 1. SUM ──────────────────────────────────
    sum_s, t_sum_s = timed(sum_serial,   arr)
    sum_p, t_sum_p = timed(sum_parallel, arr)
    print("1. SUM\n")
    print(f"   Serial result:     {sum_s}")
    print(f"   Parallel result:   {sum_p}")
    print(f"   Serial duration:   {t_sum_s:.10f} seconds")
    print(f"   Parallel duration: {t_sum_p:.10f} seconds")
    print(f"   Speedup:           {t_sum_s / t_sum_p:.10f}\n")

    # ── 2. AVERAGE ──────────────────────────────
    avg_s, t_avg_s = timed(average_serial,   arr)
    avg_p, t_avg_p = timed(average_parallel, arr)
    print("2. AVERAGE\n")
    print(f"   Serial result:     {avg_s}")
    print(f"   Parallel result:   {avg_p}")
    print(f"   Serial duration:   {t_avg_s:.10f} seconds")
    print(f"   Parallel duration: {t_avg_p:.10f} seconds")
    print(f"   Speedup:           {t_avg_s / t_avg_p:.10f}\n")

    # ── 3. MIN ──────────────────────────────────
    min_s, t_min_s = timed(min_serial,   arr)
    min_p, t_min_p = timed(min_parallel, arr)
    print("3. MIN\n")
    print(f"   Serial result:     {min_s}")
    print(f"   Parallel result:   {min_p}")
    print(f"   Serial duration:   {t_min_s:.10f} seconds")
    print(f"   Parallel duration: {t_min_p:.10f} seconds")
    print(f"   Speedup:           {t_min_s / t_min_p:.10f}\n")

    # ── 4. MAX ──────────────────────────────────
    max_s, t_max_s = timed(max_serial,   arr)
    max_p, t_max_p = timed(max_parallel, arr)
    print("4. MAX\n")
    print(f"   Serial result:     {max_s}")
    print(f"   Parallel result:   {max_p}")
    print(f"   Serial duration:   {t_max_s:.10f} seconds")
    print(f"   Parallel duration: {t_max_p:.10f} seconds")
    print(f"   Speedup:           {t_max_s / t_max_p:.10f}\n")


if __name__ == "__main__":
    main()