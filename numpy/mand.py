# use boolean indexing to generate an image of the Mandelbrot set
# https://numpy.org/devdocs/user/quickstart.html#indexing-with-boolean-arrays
# https://en.wikipedia.org/wiki/Mandelbrot_set

import numpy as np
import matplotlib.pyplot as plt


def mandelbrot(h, w, maxit=20):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    y, x = np.ogrid[-1.4 : 1.4 : h * 1j, -2 : 0.8 : w * 1j]
    c = x + y * 1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)
    for i in range(maxit):
        z = z ** 2 + c
        # who is diverging
        diverge = z * np.conj(z) > 2 ** 2
        # who is diverging now
        div_now = diverge & (divtime == maxit)
        # when
        divtime[div_now] = i
        # avoid diverging too much
        z[diverge] = 2
    return divtime


# plot
plt.imshow(mandelbrot(400, 400, 20))
plt.show()
