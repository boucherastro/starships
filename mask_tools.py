import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate.polyint import _Interpolator1D


class interp1d_masked(interp1d):

    iprint = True

    def __init__(self, x, y, **kwargs):

        if isinstance(y, np.ma.MaskedArray):
            # Initialize a 1D linear interpolation class like in interp1d
            _Interpolator1D.__init__(self, x, y, axis=-1)

            # Keep "kind" in memory
            self.kind = kwargs.pop('kind')

            # Split into non masked chunks
            y_new = split_mask(y)
            x_new = split_mask(x, y.mask)

            f_list = []
            x_range = []
            for xi, yi in zip(x_new, y_new):
                # Specify a pad where a value is return even...
                # ... if it is out of range.
                try:
                    dx = np.diff(xi[[0, 1, -2, -1]])[[0, -1]] / 2
#                     print([xi[0] - dx[0], xi[-1] + dx[-1]])
                    x_range.append([xi[0] - dx[0], xi[-1] + dx[-1]])
                    # Specify kwargs
                    kwargs['fill_value'] = (yi[0], yi[-1])
                    kwargs['bounds_error'] = False
                    # Force "kind" if not enough values
                    if xi.size < 4:
                        kind = ['linear', 'quadratic'][xi.size - 2]
                    else:
                        kind = self.kind
                    # Compute list of fct for each chunk
                    f_list.append(interp1d(xi, yi, kind=kind, **kwargs))
                except IndexError:
                    if self.iprint:
                        print("Skip single isolated value: ", xi)

            # Make sure x_range doesn't overlap
            x_range = np.array(x_range)
            diff = np.diff(x_range.ravel())
            if (diff < 0).any():
                print(np.where(diff < 0))
                print('x_range: ', x_range)
                raise ValueError('Chunks are overlapping')

            # Keep in object attributes
            self.x_range = x_range
            self.f_list = f_list

            # Define evaluation function
            self._evaluate = self._evaluate_list

        else:
            super().__init__(x, y, **kwargs)

    def _evaluate_list(self, x_new):

        x_new = np.asarray(x_new)
        y_new = np.ones_like(x_new) * np.nan
        x_range = self.x_range
        f_list = self.f_list
        for [x1, x2], f in zip(x_range, f_list):
            cond = (x1 < x_new) & (x_new <= x2)
            y_new[cond] = f(x_new[cond])

        return np.ma.array(y_new, mask=np.isnan(y_new))


def split_mask(y, mask=None):

    if mask is None:
        mask = y.mask

    if not mask.shape:
        mask = np.zeros_like(y)

    # Where the non-masked values are
    i1, i2 = where_not_masked(mask)

    return [np.array(y[i:j]) for i, j in zip(i1, i2)]


def where_not_masked(mask):

    mask = mask.astype(int)

    # Pad with mask at each extremity
    mask = np.concatenate([[1], mask, [1]])

    i1 = np.where(np.diff(mask) == -1)[0]
    i2 = np.where(np.diff(mask) == 1)[0]

    return i1, i2


def mask_fct(x, mask):

    kwargs = {}
    kwargs['kind'] = 'nearest'
    kwargs['fill_value'] = True
    kwargs['bounds_error'] = False

    return interp1d(x, mask, **kwargs)
