import sys
import os
import signal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
from llm_data_quality_assistant import corruption_functions as cf


class TimeoutException(Exception):
    pass


def timeout(seconds=2):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException(f"Test timed out after {seconds} seconds")

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)

        return wrapper

    return decorator


def make_df():
    return pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": ["foo", "bar", "baz"],
            "d": [1.1, 2.2, 3.3],
            "e": [True, False, True],
        }
    )


def test_swap_rows():
    _test_swap_rows()


@timeout(2)
def _test_swap_rows():
    df = make_df()
    out = cf.swap_rows(df.copy(), np.array([0, 1]))
    assert not out.equals(df)
    # No swap if <2 rows
    out2 = cf.swap_rows(df.copy(), np.array([0]))
    assert out2.equals(df)
    # Out of bounds
    with pytest.raises(IndexError):
        cf.swap_rows(df.copy(), np.array([0, 10]))
    # Not 1D
    with pytest.raises(ValueError):
        cf.swap_rows(df.copy(), np.array([[0, 1]]))


def test_delete_rows():
    _test_delete_rows()


@timeout(2)
def _test_delete_rows():
    df = make_df()
    out = cf.delete_rows(df.copy(), np.array([0, 1]))
    assert out.isnull().iloc[0, 0]
    # Not 1D
    with pytest.raises(ValueError):
        cf.delete_rows(df.copy(), np.array([[0, 1]]))


def test_shuffle_columns():
    _test_shuffle_columns()


@timeout(2)
def _test_shuffle_columns():
    df = make_df()
    out = cf.shuffle_columns(df.copy(), np.array([0, 1]))
    assert not out.equals(df)
    # Not 1D
    with pytest.raises(ValueError):
        cf.shuffle_columns(df.copy(), np.array([[0, 1]]))


def test_outlier():
    _test_outlier()


@timeout(2)
def _test_outlier():
    df = make_df()
    out = cf.outlier(df.copy(), np.array([[0, 0], [1, 1]]))
    assert out.iat[0, 0] != 1 or out.iat[1, 1] != 5
    # Not 2D
    with pytest.raises(ValueError):
        cf.outlier(df.copy(), np.array([0, 1]))


def test_null():
    _test_null()


@timeout(2)
def _test_null():
    df = make_df()
    out = cf.null(df.copy(), np.array([[0, 0], [1, 1]]))
    assert out.isnull().iloc[0, 0] and out.isnull().iloc[1, 1]
    # Not 2D
    with pytest.raises(ValueError):
        cf.null(df.copy(), np.array([0, 1]))


def test_typo():
    _test_typo()


@timeout(2)
def _test_typo():
    df = make_df()
    out = cf.typo(df.copy(), np.array([[0, 2], [1, 2]]))
    assert out.iat[0, 2] != "foo" or out.iat[1, 2] != "bar"
    # Not 2D
    with pytest.raises(ValueError):
        cf.typo(df.copy(), np.array([0, 1]))


def test_incorrect_datatype():
    _test_incorrect_datatype()


@timeout(2)
def _test_incorrect_datatype():
    df = make_df()
    out = cf.incorrect_datatype(df.copy(), np.array([[0, 0], [0, 2], [0, 4]]))
    assert isinstance(out.iat[0, 0], str)
    assert isinstance(out.iat[0, 2], int)
    assert isinstance(out.iat[0, 4], str)
    # Not 2D
    with pytest.raises(ValueError):
        cf.incorrect_datatype(df.copy(), np.array([0, 1]))


def test_reverse_rows():
    _test_reverse_rows()


@timeout(2)
def _test_reverse_rows():
    df = make_df()
    out = cf.reverse_rows(df.copy(), np.array([0, 1]))
    assert not out.equals(df)
    # Not 1D
    with pytest.raises(ValueError):
        cf.reverse_rows(df.copy(), np.array([[0, 1]]))


def test_swap_cells():
    _test_swap_cells()


@timeout(2)
def _test_swap_cells():
    df = make_df()
    out = cf.swap_cells(df.copy(), np.array([[0, 0], [1, 1], [2, 2]]))
    # At least one value should be different
    assert any(
        out.iat[r, c] != make_df().iat[r, c] for r, c in [[0, 0], [1, 1], [2, 2]]
    )
    # Not 2D
    with pytest.raises(ValueError):
        cf.swap_cells(df.copy(), np.array([0, 1]))
    # <2 cells: no change
    out2 = cf.swap_cells(df.copy(), np.array([[0, 0]]))
    assert out2.equals(df)


def test_case_error():
    _test_case_error()


@timeout(2)
def _test_case_error():
    df = make_df()
    out = cf.case_error(df.copy(), np.array([[0, 2], [1, 2]]))
    assert out.iat[0, 2] != "foo" or out.iat[1, 2] != "bar"
    # Not 2D
    with pytest.raises(ValueError):
        cf.case_error(df.copy(), np.array([0, 1]))
    # Non-string
    with pytest.raises(TypeError):
        cf.case_error(df.copy(), np.array([[0, 0]]))


def test_truncate():
    _test_truncate()


@timeout(2)
def _test_truncate():
    df = make_df()
    out = cf.truncate(df.copy(), np.array([[0, 2], [1, 2]]))
    assert len(out.iat[0, 2]) < len("foo") or len(out.iat[1, 2]) < len("bar")
    # Not 2D
    with pytest.raises(ValueError):
        cf.truncate(df.copy(), np.array([0, 1]))
    # Non-string
    with pytest.raises(TypeError):
        cf.truncate(df.copy(), np.array([[0, 0]]))


def test_rounding_error():
    _test_rounding_error()


@timeout(2)
def _test_rounding_error():
    df = make_df()
    out = cf.rounding_error(df.copy(), np.array([[0, 0], [1, 1]]))
    assert out.iat[0, 0] % 10 == 0 and out.iat[1, 1] % 10 == 0
    # Not 2D
    with pytest.raises(ValueError):
        cf.rounding_error(df.copy(), np.array([0, 1]))
