"""
RNG for visualization of propagation trees
"""
import random

def main():
    num_rumor = 1538
    num_nonrumor = 1849

    random.seed(42)
    rumor_idx = [random.randint(1, num_rumor) for _ in range(3)]
    nonrumor_idx = [random.randint(1, num_nonrumor) for _ in range(3)]

    print("Rumor Info Indices: ", rumor_idx)
    print("Nonrumor Info Indices: ", nonrumor_idx)

if __name__ == "__main__":
    main()
