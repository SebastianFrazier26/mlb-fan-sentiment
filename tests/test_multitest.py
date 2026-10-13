import numpy as np

from fansent.multitest import benjamini_hochberg, holm


def test_bh_hand_computed():
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


def test_holm_hand_computed():
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])


def test_caps_at_one_and_empty():
    assert holm([0.9, 0.8]).max() == 1.0
    assert benjamini_hochberg([0.9, 0.8]).max() <= 1.0
    assert holm([]).size == 0 and benjamini_hochberg([]).size == 0


def test_holm_never_below_bh():
    p = np.random.default_rng(0).uniform(size=30) ** 2
    assert (holm(p) >= benjamini_hochberg(p) - 1e-12).all()
