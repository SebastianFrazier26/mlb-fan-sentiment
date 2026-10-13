import numpy as np
import pandas as pd

from fansent.correlate import correct, family_of, format_table
from fansent.multitest import benjamini_hochberg, holm


def _fb(rows):
    return pd.DataFrame(rows, columns=["metric", "fanbase", "perm_p"])


def test_family_assignment():
    assert [family_of(m) for m in ("OPS", "AVG", "ERA", "WHIP", "FLD_PCT", "ERRORS")] == [
        "metrics"
    ] * 6
    assert family_of("COMPOSITE") == "COMPOSITE"
    assert family_of("OAA") == "OAA"
    assert family_of("CS_PCT") == "exploratory"


def test_q_values_computed_within_family_only():
    rows = (
        [("OPS", "a", 0.01), ("ERA", "a", 0.04), ("AVG", "b", 0.03)]
        + [("COMPOSITE", "a", 0.02), ("COMPOSITE", "b", 0.5)]
        + [("OAA", "a", 0.03)]
    )
    out = correct(_fb(rows))
    m = out[out["family"] == "metrics"]
    np.testing.assert_allclose(m["q_bh"], benjamini_hochberg([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(m["p_holm"], holm([0.01, 0.04, 0.03]))
    c = out[out["family"] == "COMPOSITE"]
    np.testing.assert_allclose(c["q_bh"], benjamini_hochberg([0.02, 0.5]))
    # a lone-test family is uncorrected, unaffected by the other families
    oaa = out[out["family"] == "OAA"]
    assert oaa["q_bh"].iloc[0] == 0.03 and oaa["p_holm"].iloc[0] == 0.03


def test_exploratory_excluded_from_correction():
    rows = [("OPS", "a", 0.01), ("CS_PCT", "a", 0.001), ("CS_PCT", "b", 0.002)]
    out = correct(_fb(rows))
    cs = out[out["metric"] == "CS_PCT"]
    assert cs["exploratory"].all() and cs[["q_bh", "p_holm"]].isna().all().all()
    assert (cs["perm_p"] == [0.001, 0.002]).all()
    ops = out[out["metric"] == "OPS"].iloc[0]
    assert not ops["exploratory"] and ops["q_bh"] == 0.01


def test_table_grouped_by_family():
    fb = correct(_fb([("OPS", "a", 0.01), ("CS_PCT", "a", 0.2), ("COMPOSITE", "a", 0.03)]))
    for c in ("mu_mean", "ci_low", "ci_high", "p_pos", "rhat_max"):
        fb[c] = 0.1
    text = format_table(fb, "bh", 0.05)
    assert text.index("family: COMPOSITE") < text.index("family: metrics")
    assert text.index("family: metrics") < text.index("family: exploratory")
    assert "NA" in text
