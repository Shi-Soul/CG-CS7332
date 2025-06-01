
def task1():
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.linspace(0, 1, 100)
    gamma_x = t
    gamma_y = t**2
    eta_x = 2*t + 1
    eta_y = t**3 + 4*t + 1

    plt.plot(gamma_x, gamma_y, label='γ(t)')
    plt.plot(eta_x, eta_y, label='η(t)')
    plt.scatter(1, 1, color='red', zorder=5)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('γ(t) and η(t)')
    plt.grid(True)
    plt.show()
    
def task2():
    import numpy as np
    import matplotlib.pyplot as plt

    def B(i, k, t, knots):
        if k == 0:
            return np.where((knots[i] <= t) & (t < knots[i+1]), 1, 0)
        else:
            term1 = (t - knots[i]) / (knots[i + k] - knots[i]) * B(i, k-1, t, knots) if knots[i + k] != knots[i] else 0
            term2 = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) * B(i + 1, k-1, t, knots) if knots[i + k + 1] != knots[i + 1] else 0
            return term1 + term2

    knots = [0, 1, 3, 4, 5]
    t_vals = np.linspace(-3, 8, 1000)

    # Plot each B(i, k) function separately and then combine into one figure
    fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    # fig.suptitle('B-spline Basis Functions for $B_{0,4}(t)$')

    for k in range(4):
        for i in range(len(knots) - k - 1):
            B_ik = B(i, k, t_vals, knots)
            axs[k].plot(t_vals, B_ik, label=f'$B_{{{i},{k+1}}}(t)$')
            axs[k].set_title(f'B-spline Basis Function $B_{{i,{k+1}}}(t)$')
            axs[k].set_ylabel('Value')
            axs[k].legend()
            axs[k].grid(True)

    axs[-1].set_xlabel('$t$')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    # task1()
    try:
        task2()
    except Exception as e:
        print(e)
        import pdb; pdb.post_mortem()
