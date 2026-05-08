import random
import time
import threading
from collections import deque


class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = [[] for _ in range(vertices)]
        self.visited = [False] * vertices
        self.lock = threading.Lock()

    def add_edge(self, a, b):
        self.graph[a].append(b)
        self.graph[b].append(a)  # undirected

    def reset_visited(self):
        self.visited = [False] * self.vertices

    # ─────────────────────────────────────────────
    # Sequential DFS  (iterative, stack-based)
    # ─────────────────────────────────────────────
    def dfs(self, start):
        self.reset_visited()
        stack = [start]
        self.visited[start] = True
        result = []

        while stack:
            current = stack.pop()
            result.append(current)
            for neighbor in self.graph[current]:
                if not self.visited[neighbor]:
                    self.visited[neighbor] = True
                    stack.append(neighbor)

        print(" ".join(map(str, result)))

    # ─────────────────────────────────────────────
    # Parallel DFS  (threads explore neighbours)
    # ─────────────────────────────────────────────
    def parallel_dfs(self, start):
        self.reset_visited()
        stack = [start]
        self.visited[start] = True
        result = []

        while stack:
            current = stack.pop()
            result.append(current)

            threads = []

            def explore_neighbor(neighbor):
                if not self.visited[neighbor]:
                    with self.lock:
                        if not self.visited[neighbor]:
                            self.visited[neighbor] = True
                            stack.append(neighbor)

            for neighbor in self.graph[current]:
                t = threading.Thread(target=explore_neighbor, args=(neighbor,))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

        print(" ".join(map(str, result)))

    # ─────────────────────────────────────────────
    # Sequential BFS  (queue-based)
    # ─────────────────────────────────────────────
    def bfs(self, start):
        self.reset_visited()
        queue = deque([start])
        self.visited[start] = True
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)
            for neighbor in self.graph[current]:
                if not self.visited[neighbor]:
                    self.visited[neighbor] = True
                    queue.append(neighbor)

        print(" ".join(map(str, result)))

    # ─────────────────────────────────────────────
    # Parallel BFS  (threads explore neighbours)
    # ─────────────────────────────────────────────
    def parallel_bfs(self, start):
        self.reset_visited()
        queue = deque([start])
        self.visited[start] = True
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            threads = []

            def explore_neighbor(neighbor):
                if not self.visited[neighbor]:
                    with self.lock:
                        if not self.visited[neighbor]:
                            self.visited[neighbor] = True
                            queue.append(neighbor)

            for neighbor in self.graph[current]:
                t = threading.Thread(target=explore_neighbor, args=(neighbor,))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

        print(" ".join(map(str, result)))


# ─────────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────
def main():
    print("=" * 45)
    print("   Graph BFS & DFS  (Serial + Parallel)")
    print("=" * 45)

    V = input_int("Enter number of vertices: ", min_val=1)
    E = input_int("Enter number of edges:    ", min_val=0)
    start_node = input_int(f"Enter start node (0 to {V - 1}): ", min_val=0, max_val=V - 1)

    print("\nHow would you like to define edges?")
    print("  1. Enter edges manually")
    print("  2. Generate edges randomly")
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  Please enter 1 or 2.")

    g = Graph(V)

    if choice == "1":
        print(f"\nEnter {E} edges as  'u v'  (nodes 0 to {V - 1}):")
        added = 0
        while added < E:
            try:
                u, v = map(int, input(f"  Edge {added + 1}: ").split())
                if u < 0 or u >= V or v < 0 or v >= V:
                    print(f"  Nodes must be between 0 and {V - 1}.")
                elif u == v:
                    print("  Self-loops not allowed.")
                else:
                    g.add_edge(u, v)
                    added += 1
            except ValueError:
                print("  Enter two integers separated by a space.")
    else:
        random.seed(42)
        added = 0
        attempts = 0
        while added < E and attempts < E * 10:
            a = random.randint(0, V - 1)
            b = random.randint(0, V - 1)
            if a != b:
                g.add_edge(a, b)
                added += 1
            attempts += 1
        print(f"\nRandomly generated {added} edge(s).")

    print()

    # ── Sequential DFS ──────────────────────────
    print("Sequential DFS:")
    t = time.perf_counter()
    g.dfs(start_node)
    time1 = time.perf_counter() - t
    print(f"Time: {time1:.6f} seconds")

    # ── Parallel DFS ────────────────────────────
    print("\nParallel DFS:")
    t = time.perf_counter()
    g.parallel_dfs(start_node)
    time2 = time.perf_counter() - t
    print(f"Time: {time2:.6f} seconds")
    print(f"Speedup (DFS): {time1 / time2 if time2 > 0 else float('inf'):.4f}")

    # ── Sequential BFS ──────────────────────────
    print("\nSequential BFS:")
    t = time.perf_counter()
    g.bfs(start_node)
    time3 = time.perf_counter() - t
    print(f"Time: {time3:.6f} seconds")

    # ── Parallel BFS ────────────────────────────
    print("\nParallel BFS:")
    t = time.perf_counter()
    g.parallel_bfs(start_node)
    time4 = time.perf_counter() - t
    print(f"Time: {time4:.6f} seconds")
    print(f"Speedup (BFS): {time3 / time4 if time4 > 0 else float('inf'):.4f}")


if __name__ == "__main__":
    main()