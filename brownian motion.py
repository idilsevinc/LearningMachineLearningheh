import numpy as np
import matplotlib.pyplot as plt


def quadratic_variation(B):
    return np.cumsum(np.power(np.diff(B, axis=0, prepend=0), 2), axis=0)


def compute_msd(B):
    n = len(B)
    max_lag = n // 2
    msd = np.zeros(max_lag)

    for tau in range(1, max_lag):
        displacements = B[tau:] - B[:-tau]
        msd[tau] = np.mean(np.sum(displacements ** 2, axis=1))

    return msd


def main():
    n = 10000
    d = 2
    T = 1
    times = np.linspace(0., T, n)
    print(times)
    dt = times[1] - times[0]
    # Bt2 - Bt1 ~ Normal with mean 0 and variance t2-t1
    dB = np.sqrt(dt) * np.random.normal(size=(n - 1, d))
    B0 = np.zeros(shape=(1, d))
    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)

    msd_values = compute_msd(B)

    plt.figure(figsize=(10, 5))
    plt.plot(times, B[:, 0], label="X-coordinate")
    plt.plot(times, B[:, 1], label="Y-coordinate")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Brownian Motion in 2D")
    plt.legend()
    plt.show()

    # Plot MSD
    plt.figure(figsize=(10, 5))
    plt.plot(msd_values, label="MSD")
    plt.xlabel("Lag Time")
    plt.ylabel("Mean Squared Displacement")
    plt.title("MSD vs. Lag Time")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
