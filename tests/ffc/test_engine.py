"""
Tests for SimpleFFCEngine
"""
from unittest import TestCase

from numpy import (
    full,
    isnan,
    nan,
)
from numpy.testing import assert_array_equal
from pandas import (
    date_range,
    rolling_mean,
    Timestamp,
    DataFrame,
)
from pandas.util.testing import assert_frame_equal
from testfixtures import TempDirectory

from zipline.assets import AssetFinder
from zipline.data.equities import USEquityPricing
from zipline.data.ffc.synthetic import (
    ConstantLoader,
    NullAdjustmentReader,
    SyntheticDailyBarWriter,
)
from zipline.data.ffc.loaders.us_equity_pricing import (
    USEquityPricingLoader,
    BcolzDailyBarReader,
)
from zipline.finance.trading import TradingEnvironment
from zipline.modelling.engine import SimpleFFCEngine
from zipline.modelling.factor import TestingFactor
from zipline.modelling.factor.technical import (
    MaxDrawdown,
    SimpleMovingAverage,
)
from zipline.utils.test_utils import (
    make_rotating_asset_info,
    make_simple_asset_info,
)


class RollingSumDifference(TestingFactor):
    window_length = 3
    inputs = [USEquityPricing.open, USEquityPricing.close]

    def from_windows(self, open, close):
        return (open - close).sum(axis=0)


class ConstantInputTestCase(TestCase):

    def setUp(self):
        self.constants = {
            # Every day, assume every stock starts at 2, goes down to 1,
            # goes up to 4, and finishes at 3.
            USEquityPricing.low: 1,
            USEquityPricing.open: 2,
            USEquityPricing.close: 3,
            USEquityPricing.high: 4,
        }
        self.assets = [1, 2, 3]
        self.dates = date_range('2014-01-01', '2014-02-01', freq='D', tz='UTC')
        self.asset_info = make_simple_asset_info(
            self.assets,
            start_date=self.dates[0],
            end_date=self.dates[-1],
        )
        self.asset_finder = AssetFinder(self.asset_info)

    def test_single_factor(self):

        loader = ConstantLoader(
            known_assets=self.assets,
            adjustments={},
            constants=self.constants,
        )
        engine = SimpleFFCEngine(loader, self.dates, self.asset_finder)
        result_shape = (num_dates, num_assets) = (5, len(self.assets))
        dates = self.dates[10:10 + num_dates]

        factor = RollingSumDifference()

        result = engine.factor_matrix({'f': factor}, dates[0], dates[-1])
        self.assertEqual(set(result.columns), {'f'})

        assert_array_equal(
            result['f'].unstack().values,
            full(result_shape, -factor.window_length),
        )

    def test_multiple_rolling_factors(self):

        loader = ConstantLoader(
            known_assets=self.assets,
            adjustments={},
            constants=self.constants,
        )
        engine = SimpleFFCEngine(loader, self.dates, self.asset_finder)
        shape = num_dates, num_assets = (5, len(self.assets))
        dates = self.dates[10:10 + num_dates]

        short_factor = RollingSumDifference(window_length=3)
        long_factor = RollingSumDifference(window_length=5)
        high_factor = RollingSumDifference(
            window_length=3,
            inputs=[USEquityPricing.open, USEquityPricing.high],
        )

        results = engine.factor_matrix(
            {'short': short_factor, 'long': long_factor, 'high': high_factor},
            dates[0],
            dates[-1],
        )
        self.assertEqual(set(results.columns), {'short', 'high', 'long'})

        # row-wise sum over an array whose values are all (1 - 2)
        assert_array_equal(
            results['short'].unstack().values,
            full(shape, -short_factor.window_length),
        )
        assert_array_equal(
            results['long'].unstack().values,
            full(shape, -long_factor.window_length),
        )
        # row-wise sum over an array whose values are all (1 - 3)
        assert_array_equal(
            results['high'].unstack().values,
            full(shape, -2 * high_factor.window_length),
        )

    def test_numeric_factor(self):
        constants = self.constants
        loader = ConstantLoader(
            known_assets=self.assets,
            adjustments={},
            constants=constants,
        )
        engine = SimpleFFCEngine(loader, self.dates, self.asset_finder)
        num_dates = 5
        dates = self.dates[10:10 + num_dates]
        high, low = USEquityPricing.high, USEquityPricing.low
        open, close = USEquityPricing.open, USEquityPricing.close

        high_minus_low = RollingSumDifference(inputs=[high, low])
        open_minus_close = RollingSumDifference(inputs=[open, close])
        avg = (high_minus_low + open_minus_close) / 2

        results = engine.factor_matrix(
            {
                'high_low': high_minus_low,
                'open_close': open_minus_close,
                'avg': avg,
            },
            dates[0],
            dates[-1],
        )

        high_low_result = results['high_low'].unstack()
        expected_high_low = 3.0 * (constants[high] - constants[low])
        assert_frame_equal(
            high_low_result,
            DataFrame(
                expected_high_low,
                index=dates,
                columns=self.assets,
            )
        )

        open_close_result = results['open_close'].unstack()
        expected_open_close = 3.0 * (constants[open] - constants[close])
        assert_frame_equal(
            open_close_result,
            DataFrame(
                expected_open_close,
                index=dates,
                columns=self.assets,
            )
        )

        avg_result = results['avg'].unstack()
        expected_avg = (expected_high_low + expected_open_close) / 2.0
        assert_frame_equal(
            avg_result,
            DataFrame(
                expected_avg,
                index=dates,
                columns=self.assets,
            )
        )


class SyntheticBcolzTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.first_asset_start = Timestamp('2015-04-01', tz='UTC')
        cls.env = TradingEnvironment.instance()
        cls.trading_day = cls.env.trading_day
        cls.asset_info = make_rotating_asset_info(
            num_assets=6,
            first_start=cls.first_asset_start,
            frequency=cls.trading_day,
            periods_between_starts=4,
            asset_lifetime=8,
        )
        cls.all_assets = cls.asset_info.index
        cls.all_dates = date_range(
            start=cls.first_asset_start,
            end=cls.asset_info['end_date'].max(),
            freq=cls.trading_day,
        )

        cls.finder = AssetFinder(cls.asset_info)

        cls.temp_dir = TempDirectory()
        cls.temp_dir.create()

        cls.writer = SyntheticDailyBarWriter(
            asset_info=cls.asset_info[['start_date', 'end_date']],
        )
        table = cls.writer.write(
            cls.temp_dir.getpath('testdata.bcolz'),
            cls.all_dates,
            cls.all_assets,
        )

        cls.ffc_loader = USEquityPricingLoader(
            BcolzDailyBarReader(table),
            NullAdjustmentReader(),
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_SMA(self):
        engine = SimpleFFCEngine(
            self.ffc_loader,
            self.env.trading_days,
            self.finder,
        )
        dates, assets = self.all_dates, self.all_assets
        window_length = 5
        SMA = SimpleMovingAverage(
            inputs=(USEquityPricing.close,),
            window_length=window_length,
        )

        results = engine.factor_matrix(
            {'sma': SMA},
            dates[window_length],
            dates[-1],
        )
        raw_closes = self.writer.expected_values_2d(dates, assets, 'close')
        expected_sma_result = rolling_mean(
            raw_closes,
            window_length,
            min_periods=1,
        )
        expected_sma_result[isnan(raw_closes)] = nan
        expected_sma_result = expected_sma_result[window_length:]

        sma_result = results['sma'].unstack()
        assert_frame_equal(
            sma_result,
            DataFrame(
                expected_sma_result,
                index=dates[window_length:],
                columns=assets,
            ),
        )

    def test_drawdown(self):
        # The monotonically-increasing data produced by SyntheticDailyBarWriter
        # exercises two pathological cases for MaxDrawdown.  The actual
        # computed results are pretty much useless (everything is either NaN)
        # or zero, but verifying we correctly handle those corner cases is
        # valuable.
        engine = SimpleFFCEngine(
            self.ffc_loader,
            self.env.trading_days,
            self.finder,
        )
        dates, assets = self.all_dates, self.all_assets
        window_length = 5
        drawdown = MaxDrawdown(
            inputs=(USEquityPricing.close,),
            window_length=window_length,
        )

        results = engine.factor_matrix(
            {'drawdown': drawdown},
            dates[window_length],
            dates[-1],
        )

        dd_result = results['drawdown']

        # We expect NaNs when the asset was undefined, otherwise 0 everywhere,
        # since the input is always increasing.
        expected = self.writer.expected_values_2d(dates, assets, 'close')
        expected[~isnan(expected)] = 0
        expected = expected[window_length:]

        assert_frame_equal(
            dd_result.unstack(),
            DataFrame(
                expected,
                index=dates[window_length:],
                columns=assets,
            ),
        )
