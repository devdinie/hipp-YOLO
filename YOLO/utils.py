import re
import numpy as np
from PIL import Image
from functools import reduce
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def compose(*funcs):
        # return lambda x: reduce(lambda v, f: f(v), funcs, x)
        if funcs: return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
        else: raise ValueError("Composition of empty sequence not supported.")

