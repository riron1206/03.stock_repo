#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals

import datetime
import unittest

import pandas as pd

import japandas as jpd


class TestCalendar(unittest.TestCase):

    def setUp(self):
        self.expected = [datetime.datetime(2014, 1, 1, 0, 0),
                         datetime.datetime(2014, 1, 13, 0, 0),
                         datetime.datetime(2014, 2, 11, 0, 0),
                         datetime.datetime(2014, 3, 21, 0, 0),
                         datetime.datetime(2014, 4, 29, 0, 0),
                         datetime.datetime(2014, 5, 3, 0, 0),
                         datetime.datetime(2014, 5, 4, 0, 0),
                         datetime.datetime(2014, 5, 5, 0, 0),
                         datetime.datetime(2014, 5, 6, 0, 0),
                         datetime.datetime(2014, 7, 21, 0, 0),
                         datetime.datetime(2014, 9, 15, 0, 0),
                         datetime.datetime(2014, 9, 23, 0, 0),
                         datetime.datetime(2014, 10, 13, 0, 0),
                         datetime.datetime(2014, 11, 3, 0, 0),
                         datetime.datetime(2014, 11, 23, 0, 0),
                         datetime.datetime(2014, 11, 24, 0, 0),
                         datetime.datetime(2014, 12, 23, 0, 0)]

        self.start_date = datetime.datetime(2014, 1, 1)
        self.end_date = datetime.datetime(2014, 12, 31)

    def test_calendar(self):

        calendar = jpd.JapaneseHolidayCalendar()
        holidays_0 = calendar.holidays(self.start_date,
                                       self.end_date)

        holidays_1 = calendar.holidays(self.start_date.strftime('%Y-%m-%d'),
                                       self.end_date.strftime('%Y-%m-%d'))
        holidays_2 = calendar.holidays(pd.Timestamp(self.start_date),
                                       pd.Timestamp(self.end_date))

        self.assertEqual(holidays_0.to_pydatetime().tolist(), self.expected)
        self.assertEqual(holidays_1.to_pydatetime().tolist(), self.expected)
        self.assertEqual(holidays_2.to_pydatetime().tolist(), self.expected)

    def test_cday(self):
        calendar = jpd.JapaneseHolidayCalendar()
        cday = pd.offsets.CDay(calendar=calendar)

        dt = datetime.datetime(2014, 1, 12)
        self.assertEqual(dt - cday, datetime.datetime(2014, 1, 10))
        self.assertEqual(dt + cday, datetime.datetime(2014, 1, 14))

        dt = datetime.datetime(2014, 1, 10)
        self.assertEqual(dt - cday, datetime.datetime(2014, 1, 9))
        self.assertEqual(dt + cday, datetime.datetime(2014, 1, 14))

        dt = datetime.datetime(2014, 4, 28)
        self.assertEqual(dt - cday, datetime.datetime(2014, 4, 25))
        self.assertEqual(dt + cday, datetime.datetime(2014, 4, 30))

        dt = datetime.datetime(2014, 5, 3)
        self.assertEqual(dt - cday, datetime.datetime(2014, 5, 2))
        self.assertEqual(dt + cday, datetime.datetime(2014, 5, 7))

        dt = datetime.datetime(2014, 5, 6)
        self.assertEqual(dt - cday, datetime.datetime(2014, 5, 2))
        self.assertEqual(dt + cday, datetime.datetime(2014, 5, 7))

    def test_factory(self):
        calendar = pd.tseries.holiday.get_calendar('JapaneseHolidayCalendar')
        self.assertTrue(isinstance(calendar, jpd.JapaneseHolidayCalendar))

        calendar = pd.tseries.holiday.get_calendar('TSEHolidayCalendar')
        self.assertTrue(isinstance(calendar, jpd.TSEHolidayCalendar))

    def test_holiday_attributes(self):
        calendar = jpd.JapaneseHolidayCalendar()
        self.assertEqual(calendar.rules[0].name, '元日')
        self.assertEqual(calendar.rules[0].year, 1970)
        self.assertEqual(calendar.rules[0].month, 1)
        self.assertEqual(calendar.rules[0].day, 1)

    def test_jpholiday_holidays(self):
        calendar = jpd.JapaneseHolidayCalendar()
        holidays = calendar.holidays()
        for y in range(1970, 2030):
            for m, d in [(1, 1)]:
                dt = datetime.date(y, m, d)
                self.assertTrue(dt in holidays)

        for e in self.expected:
            self.assertTrue(dt in holidays)

    def test_tseholiday_holidays(self):
        calendar = jpd.TSEHolidayCalendar()
        holidays = calendar.holidays()
        for y in range(1970, 2031):
            for m, d in [(1, 1), (1, 2), (1, 3), (12, 31)]:
                dt = datetime.date(y, m, d)
                self.assertTrue(dt in holidays)

        # test initial / final date explicitly
        self.assertTrue(datetime.date(1970, 1, 1) in holidays)
        self.assertTrue(datetime.date(2030, 12, 31) in holidays)
        for e in self.expected:
            self.assertTrue(dt in holidays)

    def test_holiday_bug(self):
        # GH 42

        for calendar in [jpd.TSEHolidayCalendar(),
                         jpd.JapaneseHolidayCalendar()]:
            holidays = calendar.holidays()

            self.assertFalse(datetime.datetime(1993, 9, 5) in holidays)
            self.assertTrue(datetime.datetime(1993, 9, 15) in holidays)

            self.assertFalse(datetime.datetime(2020, 8, 12) in holidays)
            # http://www8.cao.go.jp/chosei/shukujitsu/gaiyou.html#tokurei
            self.assertFalse(datetime.datetime(2020, 8, 11) in holidays)

    def test_heisei_emperor_abdication_holiday(self):

        for calendar in [jpd.TSEHolidayCalendar(),
                         jpd.JapaneseHolidayCalendar()]:
            holidays = calendar.holidays()

            self.assertTrue(datetime.datetime(2018, 12, 23) in holidays)
            self.assertFalse(datetime.datetime(2019, 12, 23) in holidays)

            self.assertFalse(datetime.datetime(2019, 2, 23) in holidays)
            self.assertTrue(datetime.datetime(2020, 2, 23) in holidays)

    def test_tokurei(self):
        # http://www8.cao.go.jp/chosei/shukujitsu/gaiyou.html#tokurei

        for calendar in [jpd.TSEHolidayCalendar(),
                         jpd.JapaneseHolidayCalendar()]:
            holidays = calendar.holidays()

            # 海の日
            self.assertTrue(datetime.datetime(2020, 7, 23) in holidays)
            self.assertFalse(datetime.datetime(2020, 7, 20) in holidays)
            self.assertTrue(datetime.datetime(2021, 7, 19) in holidays)

            # 山の日
            self.assertTrue(datetime.datetime(2020, 8, 10) in holidays)
            self.assertFalse(datetime.datetime(2020, 8, 11) in holidays)
            self.assertTrue(datetime.datetime(2021, 8, 11) in holidays)

            # スポーツの日
            self.assertTrue(datetime.datetime(2020, 7, 24) in holidays)
            self.assertFalse(datetime.datetime(2020, 10, 12) in holidays)
            self.assertTrue(datetime.datetime(2021, 10, 11) in holidays)

    def test_new_era(self):

        for calendar in [jpd.TSEHolidayCalendar(),
                         jpd.JapaneseHolidayCalendar()]:
            holidays = calendar.holidays()

            self.assertFalse(datetime.datetime(2019, 4, 26) in holidays)
            self.assertFalse(datetime.datetime(2019, 4, 27) in holidays)
            self.assertFalse(datetime.datetime(2019, 4, 28) in holidays)
            self.assertTrue(datetime.datetime(2019, 4, 29) in holidays)
            self.assertTrue(datetime.datetime(2019, 4, 30) in holidays)
            self.assertTrue(datetime.datetime(2019, 5, 1) in holidays)
            self.assertTrue(datetime.datetime(2019, 5, 2) in holidays)
            self.assertTrue(datetime.datetime(2019, 5, 3) in holidays)
            self.assertTrue(datetime.datetime(2019, 5, 4) in holidays)
            self.assertTrue(datetime.datetime(2019, 5, 5) in holidays)
            self.assertTrue(datetime.datetime(2019, 5, 6) in holidays)
            self.assertFalse(datetime.datetime(2019, 5, 7) in holidays)
            self.assertFalse(datetime.datetime(2019, 12, 23) in holidays)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
