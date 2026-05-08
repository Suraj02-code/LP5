import time
import threading


# ─────────────────────────────────────────────
# Sequential Linear Regression
# ─────────────────────────────────────────────
def sequential_lr(x, y):
    n = len(x)
    sum_x = sum_y = sum_xy = sum_x2 = 0.0

    start = time.perf_counter()
    for i in range(n):
        sum_x  += x[i]
        sum_y  += y[i]
        sum_xy += x[i] * y[i]
        sum_x2 += x[i] * x[i]

    beta1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    beta0 = (sum_y - beta1 * sum_x) / n
    time_taken = time.perf_counter() - start

    return beta0, beta1, time_taken


# ─────────────────────────────────────────────
# Parallel Linear Regression
# ─────────────────────────────────────────────
def parallel_lr(x, y):
    n = len(x)
    num_threads = min(8, n)
    chunk_size  = (n + num_threads - 1) // num_threads
    partial     = [[0.0, 0.0, 0.0, 0.0] for _ in range(num_threads)]

    def worker(tid, start_idx, end_idx):
        sx = sy = sxy = sx2 = 0.0
        for i in range(start_idx, end_idx):
            sx  += x[i]
            sy  += y[i]
            sxy += x[i] * y[i]
            sx2 += x[i] * x[i]
        partial[tid] = [sx, sy, sxy, sx2]

    start = time.perf_counter()

    threads = []
    for tid in range(num_threads):
        s = tid * chunk_size
        e = min(s + chunk_size, n)
        t = threading.Thread(target=worker, args=(tid, s, e))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    sum_x = sum_y = sum_xy = sum_x2 = 0.0
    for p in partial:
        sum_x  += p[0]
        sum_y  += p[1]
        sum_xy += p[2]
        sum_x2 += p[3]

    beta1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    beta0 = (sum_y - beta1 * sum_x) / n
    time_taken = time.perf_counter() - start

    return beta0, beta1, time_taken


# ─────────────────────────────────────────────
# Input helper
# ─────────────────────────────────────────────
def input_int(prompt, min_val=None):
    while True:
        try:
            val = int(input(prompt))
            if min_val is not None and val < min_val:
                print(f"  Value must be >= {min_val}.")
            else:
                return val
        except ValueError:
            print("  Invalid input. Please enter an integer.")


def input_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  Invalid input. Please enter a number.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 45)
    print("       Linear Regression (Serial + Parallel)")
    print("=" * 45)

    n = input_int("Enter number of data points: ", min_val=2)

    print(f"\nEnter {n} values for x:")
    x = [input_float(f"  x[{i+1}]: ") for i in range(n)]

    print(f"\nEnter {n} values for y:")
    y = [input_float(f"  y[{i+1}]: ") for i in range(n)]

    print()

    beta0_seq, beta1_seq, time_seq = sequential_lr(x, y)
    beta0_par, beta1_par, time_par = parallel_lr(x, y)

    print("Sequential Execution:")
    print(f"  beta0: {beta0_seq}, beta1: {beta1_seq}, Time: {time_seq}s")
    print(f"  Equation: y = {beta1_seq}x + {beta0_seq}\n")

    print("Parallel Execution:")
    print(f"  beta0: {beta0_par}, beta1: {beta1_par}, Time: {time_par}s")
    print(f"  Equation: y = {beta1_par}x + {beta0_par}\n")

    speedup = (time_seq / time_par) if time_par > 0 else 0
    print(f"Speedup: {speedup}")


if __name__ == "__main__":
    main()