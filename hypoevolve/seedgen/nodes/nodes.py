from typing import Any

import numpy as np
import pandas as pd

from hypoevolve.seedgen.nodes.base import Node, NodeIOTypes


class DATA(Node):
    """
    Leaf node that provides market data for a specific feature (OPEN, CLOSE, etc.) of a given ticker
    """

    def __init__(
        self,
        label: str,
        ticker: str | None = None,
        provider: Any | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        super(DATA, self).__init__(
            input_types=[],
            output_type=NodeIOTypes.FLOAT,
            max_childs=0,
        )
        self.label = label
        self.ticker = ticker
        self.provider = provider
        self.start_date = start_date
        self.end_date = end_date

    @property
    def name(self):
        return type(self).__name__ + f"[{self.label}]"

    @property
    def params(self) -> dict:
        params = {
            "label": self.label,
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }
        return params

    def activate(self):
        if not self.provider:
            raise ValueError("No provider provided")

        if not self.provider.has(self.ticker):
            raise ValueError(f"No data for {self.ticker}")

        series = self.provider.get(self.ticker)[self.label]

        # Apply datetime slicing if start_date or end_date is provided
        if self.start_date or self.end_date:
            series = series.loc[self.start_date : self.end_date]

        return series


class SMA(Node):
    """
    Node that calculates the simple moving average over a given period (p)
    """

    def __init__(self, period: int):
        super(SMA, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq.rolling(self.period).mean()


class ADD(Node):
    """
    Node that adds two time series values
    """

    def __init__(self):
        super(ADD, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=2,
        )

    @property
    def params(self) -> dict:
        return {}

    def activate(self, seq1: pd.Series, seq2: pd.Series) -> pd.Series:
        return seq1 + seq2


class SHIFT(Node):
    """
    Node that shifts time series data by a specified period (p) to the past or future
    """

    def __init__(self, period: int):
        super(SHIFT, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq.shift(self.period)


class DIFF(Node):
    """
    Node that calculates the difference between the current value and the value from a specified period (p) ago
    """

    def __init__(self, period: int):
        super(DIFF, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq.diff(self.period)


class PctChange(Node):
    """
    Node that calculates the percentage change between the current value and the value from a specified period (p) ago
    """

    def __init__(self, period: int):
        super(PctChange, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq.pct_change(self.period, fill_method=None).ffill()


class ShiftSign(Node):
    """
    Node that inverts the sign of time series values
    """

    def __init__(self):
        super(ShiftSign, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )

    @property
    def params(self) -> dict:
        return {}

    def activate(self, seq: pd.Series) -> pd.Series:
        return -seq


class ABS(Node):
    """
    Node that calculates the absolute value of time series values
    """

    def __init__(self):
        super(ABS, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )

    @property
    def params(self) -> dict:
        return {}

    def activate(self, seq: pd.Series) -> pd.Series:
        return np.abs(seq)


class DIV(Node):
    """
    Node that divides the first time series value by the second time series value
    """

    def __init__(self):
        super(DIV, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=2,
        )

    @property
    def params(self) -> dict:
        return {}

    def activate(self, seq1: pd.Series, seq2: pd.Series) -> pd.Series:
        return seq1 / (seq2 + 1e-10)


class SUB(Node):
    """
    Node that subtracts the second time series value from the first time series value
    """

    def __init__(self):
        super(SUB, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=2,
        )

    @property
    def params(self) -> dict:
        return {}

    def activate(self, seq1: pd.Series, seq2: pd.Series) -> pd.Series:
        return seq1 - seq2


class Comparison(Node):
    """
    Node that compares whether the first time series value is greater than the second and returns a boolean value
    """

    def __init__(self):
        super(Comparison, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=2,
        )

    @property
    def params(self) -> dict:
        return {}

    def activate(self, seq1: pd.Series, seq2: pd.Series) -> pd.Series:
        return seq1 > seq2


class NewHigh(Node):
    """
    Node that detects whether the current value has reached a new high within a specified period (p)
    """

    def __init__(self, period: int):
        super(NewHigh, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq.rolling(self.period).max() == seq


class NewLow(Node):
    """
    Node that detects whether the current value has reached a new low within a specified period (p)
    """

    def __init__(self, period: int):
        super(NewLow, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq.rolling(self.period).min() == seq


class ZSCORE(Node):
    """
    Node that calculates the z-score using the rolling mean and standard deviation over a specified period (p)
    """

    def __init__(self, period: int):
        super(ZSCORE, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        rolling_mean = seq.rolling(self.period).mean()
        rolling_std = seq.rolling(self.period).std()
        zscore = (seq - rolling_mean) / rolling_std
        return zscore


class STD(Node):
    """
    Node that calculates the rolling standard deviation over a specified period (p)
    """

    def __init__(self, period: int):
        super(STD, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq.rolling(self.period).std()


class MAX(Node):
    """
    Node that calculates the rolling maximum over a specified period (p)
    """

    def __init__(self, period: int):
        super(MAX, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )

        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq.rolling(self.period).max()


class MIN(Node):
    """
    Node that calculates the rolling minimum over a specified period (p)
    """

    def __init__(self, period: int):
        super(MIN, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )

        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq.rolling(self.period).min()


class SKEW(Node):
    """
    Node that calculates the rolling skewness over a specified period (p)
    """

    def __init__(self, period: int):
        super(SKEW, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq.rolling(self.period).skew()


class KURT(Node):
    """
    Node that calculates the rolling kurtosis over a specified period (p)
    """

    def __init__(self, period: int):
        super(KURT, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq.rolling(self.period).kurt()


class LargerThan(Node):
    """
    Node that compares whether the time series value is greater than a specified threshold (n) and returns a boolean value
    """

    def __init__(self, n: float):
        super(LargerThan, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=1,
        )
        self.n = n

    @property
    def name(self):
        return type(self).__name__ + f"(n={self.n})"

    @property
    def params(self) -> dict:
        return {"n": self.n}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq > self.n


class SmallerThan(Node):
    """
    Node that compares whether the time series value is smaller than a specified threshold (n) and returns a boolean value
    """

    def __init__(self, n: float):
        super(SmallerThan, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=1,
        )
        self.n = n

    @property
    def name(self):
        return type(self).__name__ + f"(n={self.n})"

    @property
    def params(self) -> dict:
        return {"n": self.n}

    def activate(self, seq: pd.Series) -> pd.Series:
        return seq < self.n


class ZBetween(Node):
    """
    Node that checks whether the z-score of the time series is within a specified range (lo, hi) and returns a boolean value
    """

    def __init__(self, period: int, lo: float, hi: float):
        super(ZBetween, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=1,
        )
        self.period = period
        self.lo = lo
        self.hi = hi

    @property
    def name(self):
        return (
            type(self).__name__
            + f"(p={self.period}, lo={round(self.lo, 4)}, hi={round(self.hi, 4)})"
        )

    @property
    def params(self) -> dict:
        return {"period": self.period, "lo": self.lo, "hi": self.hi}

    def activate(self, seq: pd.Series) -> pd.Series:
        rolling_mean = seq.rolling(self.period).mean()
        rolling_std = seq.rolling(self.period).std()
        zscore = (seq - rolling_mean) / (rolling_std + 1e-5)
        return (zscore >= self.lo) & (zscore <= self.hi)


class EqualApprox(Node):
    """
    Node that checks whether two time series values are approximately equal and returns a boolean value
    """

    def __init__(self, tol=1e-2):
        super(EqualApprox, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=2,
        )
        self.tol = tol

    @property
    def name(self):
        return type(self).__name__ + "()"

    @property
    def params(self) -> dict:
        return {"tol": self.tol}

    def activate(self, seq1: pd.Series, seq2: pd.Series) -> pd.Series:
        return (seq1 - seq2).abs() <= self.tol


class ZEXP(Node):
    """
    Node that calculates the z-score of the time series and then applies the exponential function over a specified period (p)
    """

    def __init__(self, period: int):
        super(ZEXP, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        rolling_mean = seq.rolling(self.period).mean()
        rolling_std = seq.rolling(self.period).std()
        zscore = (seq - rolling_mean) / (rolling_std + 1e-5)
        # Clip z-score to prevent overflow: exp(20) ≈ 4.85e8, exp(-20) ≈ 2.06e-9
        zscore_clipped = np.clip(zscore, -20, 20)
        return np.exp(zscore_clipped)


class ZSigmoid(Node):
    """
    Node that calculates the z-score of the time series and then applies the sigmoid function over a specified period (p)
    """

    def __init__(self, period: int):
        super(ZSigmoid, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.FLOAT,
            max_childs=1,
        )

        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        rolling_mean = seq.rolling(self.period).mean()
        rolling_std = seq.rolling(self.period).std()
        zscore = (seq - rolling_mean) / (rolling_std + 1e-5)
        # Clip z-score to prevent overflow in sigmoid: exp(20) and exp(-20) are safe bounds
        zscore_clipped = np.clip(zscore, -15, 15)
        return 1.0 / (1.0 + np.exp(-zscore_clipped))


class CrossUp(Node):
    """
    Node that detects when the first time series crosses above the second time series
    """

    def __init__(self):
        super(CrossUp, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=2,
        )

    @property
    def params(self) -> dict:
        return {}

    def activate(self, seq1: pd.Series, seq2: pd.Series) -> pd.Series:
        return (seq1 > seq2) & (seq1.shift(1) <= seq2.shift(1))


class CrossDown(Node):
    """
    Node that detects when the first time series crosses below the second time series
    """

    def __init__(self):
        super(CrossDown, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=2,
        )

    @property
    def params(self) -> dict:
        return {}

    def activate(self, seq1: pd.Series, seq2: pd.Series) -> pd.Series:
        return (seq1 < seq2) & (seq1.shift(1) >= seq2.shift(1))


class UpStreak(Node):
    """
    Node that detects when the time series increases consecutively for a specified period (p)
    """

    def __init__(self, period: int):
        super(UpStreak, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        inc = (seq.diff() > 0).astype(int)

        # 런 길이 계산: 동일 상태 구간별 카운트 누적
        grp = (inc != inc.shift()).cumsum()
        runlen = inc.groupby(grp).cumsum()

        out = runlen >= self.period
        return out.fillna(False)


class DownStreak(Node):
    """
    Node that detects when the time series decreases consecutively for a specified period (p)
    """

    def __init__(self, period: int):
        super(DownStreak, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=1,
        )
        self.period = period

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period})"

    @property
    def params(self) -> dict:
        return {"period": self.period}

    def activate(self, seq: pd.Series) -> pd.Series:
        dec = (seq.diff() < 0).astype(int)

        grp = (dec != dec.shift()).cumsum()
        runlen = dec.groupby(grp).cumsum()

        out = runlen >= self.period
        return out.fillna(False)


class MeanRevertKick(Node):
    """
    Node that detects when the p rolling z-score exceeds a threshold (z_th) and then reverts toward the mean within a specified period (dmax)
    """

    def __init__(
        self,
        period: int,
        z_th: float,
        dmax: int,
        eps: float,
    ):
        super(MeanRevertKick, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=1,
        )

        self.period = period
        self.z_th = z_th
        self.dmax = dmax
        self.eps = eps

    @property
    def name(self):
        return (
            type(self).__name__
            + f"(p={self.period}, z_th={round(self.z_th, 3)}, dmax={self.dmax})"
        )

    @property
    def params(self) -> dict:
        return {
            "period": self.period,
            "z_th": self.z_th,
            "dmax": self.dmax,
            "eps": self.eps,
        }

    def activate(self, seq: pd.Series) -> pd.Series:
        """
        1. z_window로 z-score 계산
        2. |z_t| >= z_th 진입점 이후,
        3. 향후 Dmax 내에서 |z|가 (|z_t|-eps) 미만으로 감소하면 트리거.
        """

        mu = seq.rolling(self.period, min_periods=self.period).mean()
        sd = seq.rolling(self.period, min_periods=self.period).std()
        z = (seq - mu) / (sd + 1e-5)
        abs_z = z.abs()

        enter = abs_z >= self.z_th

        # 향후 Dmax 내 최소 |z| 계산 (자기 시점 이후 타임 스텝부터 체크)
        future_min_abs_z = (
            abs_z[::-1].rolling(self.dmax + 1, min_periods=1).min()[::-1].shift(-1)
        )

        kick = future_min_abs_z < (abs_z - self.eps)
        out = enter & kick
        return out.fillna(False)


class PullbackWithinBand(Node):
    """
    Node that detects when the time series value reenters the Bollinger Bands (SMA(p) +/- k*sig(p)) from outside
    """

    def __init__(self, period: int, k: float):
        super(PullbackWithinBand, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=1,
        )
        self.period = period
        self.k = float(k)

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period}, k={round(self.k, 3)})"

    @property
    def params(self) -> dict:
        return {"period": self.period, "k": self.k}

    def activate(self, seq: pd.Series) -> pd.Series:
        mu = seq.rolling(self.period, min_periods=self.period).mean()
        sd = seq.rolling(self.period, min_periods=self.period).std()
        upper = mu + self.k * sd
        lower = mu - self.k * sd

        # 상단→내부, 하단→내부 재진입
        reenter_from_top = (seq.shift(1) > upper.shift(1)) & (seq <= upper)
        reenter_from_bot = (seq.shift(1) < lower.shift(1)) & (seq >= lower)

        out = reenter_from_top | reenter_from_bot
        return out.fillna(False)


class DrawdownExceed(Node):
    """
    Node that detects when the drawdown from the highest point within a specified period (lb) exceeds a threshold (pct)
    """

    def __init__(self, pct: float, lookback: int):
        super(DrawdownExceed, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=1,
        )
        assert 0.0 < pct < 1.0

        self.pct = pct
        self.lookback = lookback

    @property
    def name(self):
        return type(self).__name__ + f"(pct={round(self.pct, 3)}, lb={self.lookback})"

    @property
    def params(self) -> dict:
        return {"pct": self.pct, "lookback": self.lookback}

    def activate(self, seq: pd.Series) -> pd.Series:
        roll_max = seq.rolling(self.lookback, min_periods=self.lookback).max()
        dd = 1.0 - (seq / roll_max)
        out = dd >= self.pct
        return out.fillna(False)


class JumpDetect(Node):
    """
    Node that detects sudden jumps when the absolute change in the time series exceeds a specified upper percentile (q_tail) of the distribution within a given period (p)
    """

    def __init__(self, period: int, q_tail: float):
        super(JumpDetect, self).__init__(
            input_types=[NodeIOTypes.FLOAT],
            output_type=NodeIOTypes.BINARY,
            max_childs=1,
        )
        assert 0.0 < q_tail < 1.0
        self.period = period
        self.q_tail = q_tail

    @property
    def name(self):
        return type(self).__name__ + f"(p={self.period}, tail={round(self.q_tail, 3)})"

    @property
    def params(self) -> dict:
        return {"period": self.period, "q_tail": self.q_tail}

    def activate(self, seq: pd.Series) -> pd.Series:
        r = seq.diff().abs()
        thresh = r.rolling(self.period, min_periods=self.period).quantile(
            1.0 - self.q_tail
        )
        out = r >= thresh
        return out.fillna(False)
