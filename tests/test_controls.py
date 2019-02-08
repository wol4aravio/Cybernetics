import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from cybernetics.controls.piecewise_constant_controller import PiecewiseConstantController as PWC
from cybernetics.controls.piecewise_linear_controller import PiecewiseLinearController as PWL


def test_pwc():
    controller = PWC(time_grid=np.arange(0.0, 1.1, 0.1), values=list(range(0, 11)))
    assert controller(0) == 0
    assert controller(0.01) == 0
    assert controller(1e9) == 10
    with pytest.raises(ValueError):
        _ = controller(-1)


def test_pwl():
    controller = PWL(time_grid=[0, 1, 3], values=[0, -1, 3])
    assert_almost_equal(controller(0), 0.0)
    assert_almost_equal(controller(1), -1.0)
    assert_almost_equal(controller(3), 3.0)
    assert_almost_equal(controller(1.5), 0.0)
    assert_almost_equal(controller(2), 1.0)
    assert_almost_equal(controller(2.5), 2.0)
    with pytest.raises(ValueError):
        _ = controller(-1)
    with pytest.raises(ValueError):
        _ = controller(1e3)
