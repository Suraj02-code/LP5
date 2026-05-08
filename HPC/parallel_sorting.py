import random
import time
import threading
import copy


def print_array(arr):
    print("Array: [", " ".join(map(str, arr)), "]")


# ─────────────────────────────────────────────
# Sequential Bubble Sort
# ─────────────────────────────────────────────
def sequential_bubble_sort(arr):
    temp = arr[:]
    start = time.perf_counter()

    n = len(temp)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if temp[j] > temp[j + 1]:
                temp[j], temp[j + 1] = temp[j + 1], temp[j]

    seq_time = time.perf_counter() - start
    print(f"\nSequential Bubble Sort Time: {seq_time:.10f} s")
    print_array(temp)
    return seq_time


# ─────────────────────────────────────────────
# Parallel Odd-Even Sort  (mirrors OpenMP version)
# Odd phase and Even phase each run neighbours
# in parallel using threads.
# ─────────────────────────────────────────────
def parallel_odd_even_sort(arr, seq_time):
    temp = arr[:]
    n = len(temp)
    lock = threading.Lock()

    def compare_swap(i):
        """Swap temp[i] and temp[i+1] if out of order."""
        nonlocal sorted_flag
        if temp[i] > temp[i + 1]:
            with lock:
                if temp[i] > temp[i + 1]:      # double-checked
                    temp[i], temp[i + 1] = temp[i + 1], temp[i]
                    sorted_flag[0] = False

    start = time.perf_counter()
    sorted_flag = [False]

    while not sorted_flag[0]:
        sorted_flag[0] = True

        # ── Odd phase: pairs (1,2), (3,4), (5,6) … ──
        threads = []
        for i in range(1, n - 1, 2):
            t = threading.Thread(target=compare_swap, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # ── Even phase: pairs (0,1), (2,3), (4,5) … ──
        threads = []
        for i in range(0, n - 1, 2):
            t = threading.Thread(target=compare_swap, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    par_time = time.perf_counter() - start
    print(f"\nParallel Bubble Sort Time: {par_time:.10f} s")
    print_array(temp)
    speedup = seq_time / par_time if par_time > 0 else float("inf")
    print(f"\nSpeedup: {speedup:.10f}")


# ─────────────────────────────────────────────
# Merge helper (in-place on a list slice)
# ─────────────────────────────────────────────
def merge(arr, left, mid, right):
    left_part  = arr[left : mid + 1]
    right_part = arr[mid + 1 : right + 1]
    i = j = 0
    k = left
    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]; i += 1
        else:
            arr[k] = right_part[j]; j += 1
        k += 1
    while i < len(left_part):
        arr[k] = left_part[i]; i += 1; k += 1
    while j < len(right_part):
        arr[k] = right_part[j]; j += 1; k += 1


# ─────────────────────────────────────────────
# Sequential Merge Sort
# ─────────────────────────────────────────────
def sequential_merge_sort(arr, left, right):
    if left < right:
        mid = (left + right) // 2
        sequential_merge_sort(arr, left, mid)
        sequential_merge_sort(arr, mid + 1, right)
        merge(arr, left, mid, right)


# ─────────────────────────────────────────────
# Parallel Merge Sort  (two halves in separate threads)
# ─────────────────────────────────────────────
def parallel_merge_sort(arr, left, right, depth=0):
    """
    depth controls how many levels of parallelism we spawn.
    Beyond depth 4, overhead outweighs benefit for small arrays,
    so we fall back to sequential.
    """
    if left < right:
        mid = (left + right) // 2

        if depth < 4:
            t1 = threading.Thread(target=parallel_merge_sort,
                                  args=(arr, left, mid, depth + 1))
            t2 = threading.Thread(target=parallel_merge_sort,
                                  args=(arr, mid + 1, right, depth + 1))
            t1.start(); t2.start()
            t1.join();  t2.join()
        else:
            sequential_merge_sort(arr, left, mid)
            sequential_merge_sort(arr, mid + 1, right)

        merge(arr, left, mid, right)


# ─────────────────────────────────────────────
# Wrapper: run sequential or parallel merge sort
# ─────────────────────────────────────────────
def perform_merge_sort(arr, parallel, seq_time=None):
    temp = arr[:]
    n = len(temp)

    start = time.perf_counter()
    if parallel:
        parallel_merge_sort(temp, 0, n - 1)
    else:
        sequential_merge_sort(temp, 0, n - 1)
    time_taken = time.perf_counter() - start

    label = "Parallel" if parallel else "Sequential"
    print(f"\n{label} Merge Sort Time: {time_taken:.10f} s")
    print_array(temp)

    if parallel and seq_time is not None:
        speedup = seq_time / time_taken if time_taken > 0 else float("inf")
        print(f"\nSpeedup: {speedup:.10f}")
        return None
    else:
        return time_taken     # return seq_time for later speedup calc


# ─────────────────────────────────────────────
# Main menu
# ─────────────────────────────────────────────
def main():
    size = int(input("Enter size of array: "))
    arr = [random.randint(0, size - 1) for _ in range(size)]

    seq_bubble_time = None
    seq_merge_time  = None

    while True:
        print("\n-------------------------")
        print("         Menu")
        print("-------------------------")
        print(" 1. Bubble Sort")
        print(" 2. Merge Sort")
        print(" 3. Exit")
        print("-------------------------")
        choice = input(" Enter your choice: ").strip()
        print()

        if choice == "1":
            print("\nInput Array: ", end=""); print_array(arr)
            seq_bubble_time = sequential_bubble_sort(arr)
            parallel_odd_even_sort(arr, seq_bubble_time)

        elif choice == "2":
            print("\nInput Array: ", end=""); print_array(arr)
            seq_merge_time = perform_merge_sort(arr, parallel=False)
            perform_merge_sort(arr, parallel=True, seq_time=seq_merge_time)

        elif choice == "3":
            print("Exiting Program")
            break

        else:
            print("Invalid choice! Please try again.")


if __name__ == "__main__":
    main()