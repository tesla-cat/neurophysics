import numpy as np
import matplotlib.pyplot as plt

def f_xcal(xy, theta_p=0.5, theta_d=0.1):
    dw = np.where(
        xy > theta_p * theta_d,
        xy - theta_p,
        -xy * (1-theta_d)/theta_d
    )
    return dw


if __name__=='__main__':
    xy = np.linspace(0, 1, 100)
    dw = f_xcal(xy)
    
    plt.plot(xy, dw)
    plt.show()
