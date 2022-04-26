import sys
sys.coinit_flags = 0
import time
from decimal import *
from tcoreapi_mq import * 
import tcoreapi_mq
import re

import math
import numpy as np
from scipy.interpolate import CubicSpline
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidget, QTableWidgetItem, QDockWidget
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QObject
from pyqtgraph import GraphicsLayoutWidget, PlotCurveItem, PlotItem, LegendItem, ViewBox
import sys
import os
import time
import calendar
from enum import Enum
import csv
import pygame


# Maturity
class Maturity(Enum):
    M1 = 1; M2 = 2; M3 = 3; Q1 = 4; Q2 = 5; Q3 = 6

# Stock Type
class StockType(Enum):
    etf50 = 1; h300 = 2; gz300 = 3; s300 = 4

# Future Type
class FutureType(Enum):
    IF = 1; IH = 2

# Option Type
class OptionType(Enum):
    C = 1; P = 2

str_to_type: dict = {}
type_to_str: dict = {}
for sty in [('etf50', StockType.etf50), ('h300', StockType.h300), ('gz300', StockType.gz300), ('s300', StockType.s300)]:
    str_to_type[sty[0]] = sty[1]
    type_to_str[sty[1]] = sty[0]
for fty in [('IF', FutureType.IF), ('IH', FutureType.IH)]:
    str_to_type[fty[0]] = fty[1]
    type_to_str[fty[1]] = fty[0]
for mat in [('M1', Maturity.M1), ('M2', Maturity.M2), ('M3', Maturity.M3), ('Q1', Maturity.Q1), ('Q2', Maturity.Q2), ('Q3', Maturity.Q3)]:
    str_to_type[mat[0]] = mat[1]
    type_to_str[mat[1]] = mat[0]

g_QuoteZMQ = None
g_QuoteSession = ''
q_data = None

freq_for_screen = 1900
freq_for_mixed_screen = 1900
QuoteID = []
Mat = {}
for ty in [StockType.gz300, StockType.etf50, StockType.h300, StockType.s300, FutureType.IF, FutureType.IH]:
    Mat[ty] = []

holiday = (calendar.datetime.date(2020, 1, 1),
                      calendar.datetime.date(2020, 1, 24),
                      calendar.datetime.date(2020, 1, 27),
                      calendar.datetime.date(2020, 1, 28),
                      calendar.datetime.date(2020, 1, 29),
                      calendar.datetime.date(2020, 1, 30),
                      calendar.datetime.date(2020, 4, 6),
                      calendar.datetime.date(2020, 5, 1),
                      calendar.datetime.date(2020, 5, 4),
                      calendar.datetime.date(2020, 5, 5),
                      calendar.datetime.date(2020, 6, 25),
                      calendar.datetime.date(2020, 6, 26),
                      calendar.datetime.date(2020, 10, 1),
                      calendar.datetime.date(2020, 10, 2),
                      calendar.datetime.date(2020, 10, 5),
                      calendar.datetime.date(2020, 10, 6),
                      calendar.datetime.date(2020, 10, 7),
                      calendar.datetime.date(2020, 10, 8),

                      calendar.datetime.date(2021, 1, 1),
                      calendar.datetime.date(2021, 2, 11),
                      calendar.datetime.date(2021, 2, 12),
                      calendar.datetime.date(2021, 2, 15),
                      calendar.datetime.date(2021, 2, 16),
                      calendar.datetime.date(2021, 2, 17),
                      calendar.datetime.date(2021, 4, 5),
                      calendar.datetime.date(2021, 5, 3),
                      calendar.datetime.date(2021, 6, 14),
                      calendar.datetime.date(2021, 9, 20),
                      calendar.datetime.date(2021, 9, 21),
                      calendar.datetime.date(2021, 10, 1),
                      calendar.datetime.date(2021, 10, 4),
                      calendar.datetime.date(2021, 10, 5),
                      calendar.datetime.date(2021, 10, 6),
                      calendar.datetime.date(2021, 10, 7),
    ) # 2020 + 2021


def cdf(x: float):

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1
    if x < 0:
        sign = -1
    x = math.fabs(x) / math.sqrt(2.0);
    t = 1.0 / (1.0 + p * x);
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(- x * x)
    return 0.5 * (1.0 + sign * y)

def pdf(x: float):

    pi = 3.141592653589793
    return 1 / (math.sqrt(2 * pi)) * math.exp(- x * x / 2)

def BS(oty: OptionType, K: float, T: float, S: float, sigma: float):

    sigmaSqrtT = sigma * math.sqrt(T)
    d1 = math.log(S / K) / sigmaSqrtT + 0.5 * sigmaSqrtT
    d2 = d1 - sigmaSqrtT
    if oty == OptionType.C:
        return S * cdf(d1) - K * cdf(d2)
    else:
        return K * cdf(-d2) - S * cdf(-d1)

class OptionInfo:

    def __init__(self, sty: StockType, mat: Maturity, oty: OptionType, K: float, P: float, ask: float, bid: float):
        self.sty = sty
        self.mat = mat
        self.oty = oty
        self.K = K
        self.P = P
        self.P_yc: float = -1
        self.ask = ask
        self.bid = bid
        self.T: float = 1
        self.S: float = 1
        self.cb: dict = {}
        self.cb['if'] = False; self.cb['start_time'] = time.time()
        self.settlement_price: float = -1
        self.deposit: float = -1
        self.written: bool = False

    def midbidaskspread(self):
        return (self.ask+self.bid)/2

    def iv(self):
        a = 0.0001; b = 3; NTRY = 50; FACTOR = 1.6; S = self.S; T = self.T; K = self.K; P = self.midbidaskspread(); oty = self.oty
        f1 = BS(oty, K, T, S, a) - P; f2 = BS(oty, K, T, S, b) - P
        # RootBracketing
        for j in range(NTRY):
            if f1 * f2 < 0:
                break
            else:
                if abs(f1) < abs(f2):
                    a += FACTOR * (a - b)
                    f1 = BS(oty, K, T, S, a) - P
                else:
                    b += FACTOR * (b - a)
                    f2 = BS(oty, K, T, S, b) - P
        # rfbisect
        tol = 1e-6
        while (b - a) > tol:
            c = (a + b) / 2.0
            if abs(BS(oty, K, T, S, c) - P) < tol:
                return c
            else:
                if (BS(oty, K, T, S, a) - P) * (BS(oty, K, T, S, c) - P) < 0:
                    b = c
                else:
                    a = c
        return c

    def iv_s(self, s: float):
        a = 0.0001; b = 3; NTRY = 50; FACTOR = 1.6; S = s; T = self.T; K = self.K; P = self.midbidaskspread(); oty = self.oty
        f1 = BS(oty, K, T, S, a) - P; f2 = BS(oty, K, T, S, b) - P
        # RootBracketing
        for j in range(NTRY):
            if f1 * f2 < 0:
                break
            else:
                if abs(f1) < abs(f2):
                    a += FACTOR * (a - b)
                    f1 = BS(oty, K, T, S, a) - P
                else:
                    b += FACTOR * (b - a)
                    f2 = BS(oty, K, T, S, b) - P
        # rfbisect
        tol = 1e-6
        while (b - a) > tol:
            c = (a + b) / 2.0
            if abs(BS(oty, K, T, S, c) - P) < tol:
                return c
            else:
                if (BS(oty, K, T, S, a) - P) * (BS(oty, K, T, S, c) - P) < 0:
                    b = c
                else:
                    a = c
        return c

    def iv_p(self, p: float):
        a = 0.0001; b = 3; NTRY = 50; FACTOR = 1.6; S = self.S; T = self.T; K = self.K; P = p; oty = self.oty
        f1 = BS(oty, K, T, S, a) - P; f2 = BS(oty, K, T, S, b) - P
        # RootBracketing
        for j in range(NTRY):
            if f1 * f2 < 0:
                break
            else:
                if abs(f1) < abs(f2):
                    a += FACTOR * (a - b)
                    f1 = BS(oty, K, T, S, a) - P
                else:
                    b += FACTOR * (b - a)
                    f2 = BS(oty, K, T, S, b) - P
        # rfbisect
        tol = 1e-6
        while (b - a) > tol:
            c = (a + b) / 2.0
            if abs(BS(oty, K, T, S, c) - P) < tol:
                return c
            else:
                if (BS(oty, K, T, S, a) - P) * (BS(oty, K, T, S, c) - P) < 0:
                    b = c
                else:
                    a = c
        return c

    def delta(self):
        iv = self.iv(); S = self.S; T = self.T
        if self.oty == OptionType.C:
            return cdf(math.log(S / self.K) / (iv * math.sqrt(T)) + 0.5 * iv * math.sqrt(T))
        else:
            return cdf(math.log(S / self.K) / (iv * math.sqrt(T)) + 0.5 * iv * math.sqrt(T)) - 1

    def vega(self):
        iv = self.iv(); S = self.S; T = self.T
        return S * math.sqrt(T) * pdf(math.log(S / self.K) / (iv * math.sqrt(T)) + 0.5 * iv * math.sqrt(T))

    def gamma(self):
        iv = self.iv(); S = self.S; T = self.T
        return pdf(math.log(S / self.K) / (iv * math.sqrt(T)) + 0.5 * iv * math.sqrt(T)) / S / iv / math.sqrt(T)

    def _deposit(self, ul_yc: float):
        if self.sty == StockType.gz300:
            if self.oty == OptionType.C:
                self.deposit = self.settlement_price * 100 + max(ul_yc * 100 * 0.1 - max(self.K - ul_yc, 0) * 100, 0.5 * ul_yc * 100 * 0.1)
            elif self.oty == OptionType.P:
                self.deposit = self.settlement_price * 100 + max(ul_yc * 100 * 0.1 - max(ul_yc - self.K, 0) * 100, 0.5 * self.K * 100 * 0.1)
        elif self.sty in [StockType.etf50, StockType.h300, StockType.s300]:
            if self.oty == OptionType.C:
                self.deposit = (self.settlement_price + max(0.12 * ul_yc - max(self.K - ul_yc, 0), 0.07 * ul_yc)) * 10000
            elif self.oty == OptionType.P:
                self.deposit = min(self.settlement_price + max(0.12 * ul_yc - max(ul_yc - self.K, 0), 0.07 * self.K), self.K) * 10000

class OptData: # one stock type

    def __init__(self, sty: StockType):
        self.sty = sty
        self.Mat_to_2005 = {}
        self._2005_to_Mat = {}
        self.T = {}
        self.initT = {}
        self.S = {}
        self.k0 = {}
        self.posi = {}
        self.OptionList = {}
        self.ul: float = -1
        self.ul_yc: float = -1
        self.ul_highest: float = -1
        self.ul_lowest: float = -1
        if sty == StockType.gz300:
            self.cm = 100
            self.matlist = [Maturity.M1, Maturity.M2, Maturity.M3, Maturity.Q1, Maturity.Q2, Maturity.Q3]
        elif sty in [StockType.etf50, StockType.h300, StockType.s300]:
            self.cm = 10000
            self.matlist = [Maturity.M1, Maturity.M2, Maturity.Q1, Maturity.Q2]

        self.k_list = {}
        self.getMat()
        
    def getMat(self):
        for mat in self.matlist:
            mt = Mat[self.sty][mat].month
            if mt < 10: str_month = '0' + str(mt) 
            else: str_month = str(mt)
            self.Mat_to_2005[mat] = str(Mat[self.sty][mat].year)[2:] + str_month
            self._2005_to_Mat[self.Mat_to_2005[mat]] = mat

        def num_weekend(date1: calendar.datetime.date, date2: calendar.datetime.date):
            num = 0
            oneday = calendar.datetime.timedelta(days = 1)
            date = calendar.datetime.date(date1.year, date1.month, date1.day)
            while date != date2:
                if date.weekday() == 5 or date.weekday() == 6 or date in holiday:
                    num += 1
                date += oneday
            return num

        c = calendar.Calendar(firstweekday=calendar.SUNDAY)
        t = time.localtime()
        year = t.tm_year; month = t.tm_mon; mday = t.tm_mday
        currentDate = calendar.datetime.date(year, month, mday)

        for mat in self.matlist:
            self.T[mat] = ((Mat[self.sty][mat] - currentDate).days - num_weekend(currentDate, Mat[self.sty][mat]))/244
            self.initT[mat] = self.T[mat]

    def subscribe_init(self, mat: Maturity):
        # QuoteID and Optionlists # IO2006-C-3500, 510050C2006M03000, 510300C2006M03800, 159919C2006M004000
        QuoteID_addin = []
        for id in QuoteID:
            if self.sty == StockType.gz300 and id[11:13] == 'IO' and id[16:20] == self.Mat_to_2005[mat]:
                QuoteID_addin.append(id)
            elif self.sty == StockType.etf50 and id[9:15] == '510050' and id[18:22] == self.Mat_to_2005[mat]:
                QuoteID_addin.append(id)
            elif self.sty == StockType.h300 and id[9:15] == '510300' and id[18:22] == self.Mat_to_2005[mat]:
                QuoteID_addin.append(id)
            elif self.sty == StockType.s300 and id[10:16] == '159919' and id[19:23] == self.Mat_to_2005[mat]:
                QuoteID_addin.append(id)
        
        self.OptionList[mat] = []
        QuoteID_addin_C_K = []
        for id in QuoteID_addin:
            if (self.sty == StockType.gz300 and id[21] == 'C') or (self.sty in [StockType.etf50, StockType.h300] and id[23] == 'C') or (self.sty == StockType.s300 and id[24] == 'C'):
                QuoteID_addin_C_K.append(float(id[last_C_P(id):]))
        QuoteID_addin_C_K.sort()
        for k in QuoteID_addin_C_K:
            self.OptionList[mat].append([OptionInfo(self.sty, mat, OptionType.C, k, 10000, 10001, 10000), OptionInfo(self.sty, mat, OptionType.P, k, 1, 2, 1)])

        self.k_list[mat] = QuoteID_addin_C_K
        self.S_k0_posi(mat)

    def S_k0_posi(self, mat: Maturity): # update
        optlist = self.OptionList[mat]
        n = len(optlist)
        future = [optlist[i][0].midbidaskspread() - optlist[i][1].midbidaskspread() + optlist[i][0].K for i in range(n)]
        future.sort()
        avg = np.mean(future[1:-1])
        self.S[mat] = avg
        self.posi[mat] = np.argmin(abs(np.array(self.k_list[mat]) - avg))
        self.k0[mat] = optlist[self.posi[mat]][0].K

    def vix(self, mat: Maturity):
        n = len(self.OptionList[mat])
        cen = self.posi[mat]
        if cen == 0 or cen == (n-1):
            return (self.OptionList[mat][cen][0].iv() + self.OptionList[mat][cen][1].iv()) / 2
        else:
            opt1 = self.OptionList[mat][cen - 1]
            opt2 = self.OptionList[mat][cen]
            opt3 = self.OptionList[mat][cen + 1]
            x = [opt1[0].K, opt2[0].K, opt3[0].K]
            y = [(opt1[0].iv() + opt1[1].iv()) / 2, (opt2[0].iv() + opt2[1].iv()) / 2, (opt3[0].iv() + opt3[1].iv()) / 2]
            cs = CubicSpline(x, y)
            return cs(self.S[mat])

    def skew_same_T(self, mat: Maturity):
        optlist = self.OptionList[mat]
        n = len(optlist)
        f0 = self.S[mat]
        k0 = self.k0[mat]
        p1 = - (1 + math.log(f0 / k0) - f0 / k0); p2 = 2 * math.log(k0 / f0) * (f0 / k0 - 1) + 1/2 * math.log(k0 / f0) ** 2; p3 = 3 * math.log(k0 / f0) ** 2 * (1/3 * math.log(k0 / f0) - 1 + f0 / k0)
        for i in range(n):
            if optlist[i][0].K <= f0:
                if i == 0:
                    p1 += - 1 / (optlist[i][1].K) ** 2 * optlist[i][1].midbidaskspread() * (optlist[i + 1][1].K - optlist[i][1].K)
                    p2 += 2 / (optlist[i][1].K) ** 2 * (1 - math.log(optlist[i][1].K / f0)) * optlist[i][1].midbidaskspread() * (optlist[i + 1][1].K - optlist[i][1].K)
                    p3 += 3 / (optlist[i][1].K) ** 2 * (2 * math.log(optlist[i][1].K / f0) - math.log(optlist[i][1].K / f0) ** 2) * optlist[i][1].midbidaskspread() * (optlist[i + 1][1].K - optlist[i][1].K)
                elif i == (n-1):
                    p1 += - 1 / (optlist[i][1].K) ** 2 * optlist[i][1].midbidaskspread() * (optlist[i][1].K - optlist[i - 1][1].K)
                    p2 += 2 / (optlist[i][1].K) ** 2 * (1 - math.log(optlist[i][1].K / f0)) * optlist[i][1].midbidaskspread() * (optlist[i][1].K - optlist[i - 1][1].K)
                    p3 += 3 / (optlist[i][1].K) ** 2 * (2 * math.log(optlist[i][1].K / f0) - math.log(optlist[i][1].K / f0) ** 2) * optlist[i][1].midbidaskspread() * (optlist[i][1].K - optlist[i - 1][1].K)
                else:
                    p1 += - 1 / (optlist[i][1].K) ** 2 * optlist[i][1].midbidaskspread() * (optlist[i + 1][1].K - optlist[i - 1][1].K) / 2
                    p2 += 2 / (optlist[i][1].K) ** 2 * (1 - math.log(optlist[i][1].K / f0)) * optlist[i][1].midbidaskspread() * (optlist[i + 1][1].K - optlist[i - 1][1].K) / 2
                    p3 += 3 / (optlist[i][1].K) ** 2 * (2 * math.log(optlist[i][1].K / f0) - math.log(optlist[i][1].K / f0) ** 2) * optlist[i][1].midbidaskspread() * (optlist[i + 1][1].K - optlist[i - 1][1].K) / 2
            elif optlist[i][0].K >= f0:
                if i == (n-1):
                    p1 += - 1 / (optlist[i][0].K) ** 2 * optlist[i][0].midbidaskspread() * (optlist[i][0].K - optlist[i - 1][0].K)
                    p2 += 2 / (optlist[i][0].K) ** 2 * (1 - math.log(optlist[i][0].K / f0)) * optlist[i][0].midbidaskspread() * (optlist[i][0].K - optlist[i - 1][0].K)
                    p3 += 3 / (optlist[i][0].K) ** 2 * (2 * math.log(optlist[i][0].K / f0) - math.log(optlist[i][0].K / f0) ** 2) * optlist[i][0].midbidaskspread() * (optlist[i][0].K - optlist[i - 1][0].K)
                elif i == 0:
                    p1 += - 1 / (optlist[i][0].K) ** 2 * optlist[i][0].midbidaskspread() * (optlist[i + 1][0].K - optlist[i][0].K)
                    p2 += 2 / (optlist[i][0].K) ** 2 * (1 - math.log(optlist[i][0].K / f0)) * optlist[i][0].midbidaskspread() * (optlist[i + 1][0].K - optlist[i][0].K)
                    p3 += 3 / (optlist[i][0].K) ** 2 * (2 * math.log(optlist[i][0].K / f0) - math.log(optlist[i][0].K / f0) ** 2) * optlist[i][0].midbidaskspread() * (optlist[i + 1][0].K - optlist[i][0].K)
                else:
                    p1 += - 1 / (optlist[i][0].K) ** 2 * optlist[i][0].midbidaskspread() * (optlist[i + 1][0].K - optlist[i - 1][0].K) / 2
                    p2 += 2 / (optlist[i][0].K) ** 2 * (1 - math.log(optlist[i][0].K / f0)) * optlist[i][0].midbidaskspread() * (optlist[i + 1][0].K - optlist[i - 1][0].K) / 2
                    p3 += 3 / (optlist[i][0].K) ** 2 * (2 * math.log(optlist[i][0].K / f0) - math.log(optlist[i][0].K / f0) ** 2) * optlist[i][0].midbidaskspread() * (optlist[i + 1][0].K - optlist[i - 1][0].K) / 2
        if p2 - p1 ** 2 > 0:
            return (p3 - 3 * p1 * p2 + 2 * p1 ** 3) / math.sqrt((p2 - p1 ** 2) ** 3)
        else:
            return 0

    def skew_same_T_partial(self, mat: Maturity, partial: int):
        optlist = self.OptionList[mat]
        n = len(optlist)
        f0 = self.S[mat]
        k0 = self.k0[mat]
        cen = self.posi[mat]
        p1 = - (1 + math.log(f0 / k0) - f0 / k0); p2 = 2 * math.log(k0 / f0) * (f0 / k0 - 1) + 1/2 * math.log(k0 / f0) ** 2; p3 = 3 * math.log(k0 / f0) ** 2 * (1/3 * math.log(k0 / f0) - 1 + f0 / k0)
        for i in range(max(cen - partial, 0), min(cen + partial + 1, n)):
            if optlist[i][0].K <= f0:
                if i == 0:
                    p1 += - 1 / (optlist[i][1].K) ** 2 * optlist[i][1].midbidaskspread() * (optlist[i + 1][1].K - optlist[i][1].K)
                    p2 += 2 / (optlist[i][1].K) ** 2 * (1 - math.log(optlist[i][1].K / f0)) * optlist[i][1].midbidaskspread() * (optlist[i + 1][1].K - optlist[i][1].K)
                    p3 += 3 / (optlist[i][1].K) ** 2 * (2 * math.log(optlist[i][1].K / f0) - math.log(optlist[i][1].K / f0) ** 2) * optlist[i][1].midbidaskspread() * (optlist[i + 1][1].K - optlist[i][1].K)
                elif i == (n-1):
                    p1 += - 1 / (optlist[i][1].K) ** 2 * optlist[i][1].midbidaskspread() * (optlist[i][1].K - optlist[i - 1][1].K)
                    p2 += 2 / (optlist[i][1].K) ** 2 * (1 - math.log(optlist[i][1].K / f0)) * optlist[i][1].midbidaskspread() * (optlist[i][1].K - optlist[i - 1][1].K)
                    p3 += 3 / (optlist[i][1].K) ** 2 * (2 * math.log(optlist[i][1].K / f0) - math.log(optlist[i][1].K / f0) ** 2) * optlist[i][1].midbidaskspread() * (optlist[i][1].K - optlist[i - 1][1].K)
                else:
                    p1 += - 1 / (optlist[i][1].K) ** 2 * optlist[i][1].midbidaskspread() * (optlist[i + 1][1].K - optlist[i - 1][1].K) / 2
                    p2 += 2 / (optlist[i][1].K) ** 2 * (1 - math.log(optlist[i][1].K / f0)) * optlist[i][1].midbidaskspread() * (optlist[i + 1][1].K - optlist[i - 1][1].K) / 2
                    p3 += 3 / (optlist[i][1].K) ** 2 * (2 * math.log(optlist[i][1].K / f0) - math.log(optlist[i][1].K / f0) ** 2) * optlist[i][1].midbidaskspread() * (optlist[i + 1][1].K - optlist[i - 1][1].K) / 2
            elif optlist[i][0].K >= f0:
                if i == (n-1):
                    p1 += - 1 / (optlist[i][0].K) ** 2 * optlist[i][0].midbidaskspread() * (optlist[i][0].K - optlist[i - 1][0].K)
                    p2 += 2 / (optlist[i][0].K) ** 2 * (1 - math.log(optlist[i][0].K / f0)) * optlist[i][0].midbidaskspread() * (optlist[i][0].K - optlist[i - 1][0].K)
                    p3 += 3 / (optlist[i][0].K) ** 2 * (2 * math.log(optlist[i][0].K / f0) - math.log(optlist[i][0].K / f0) ** 2) * optlist[i][0].midbidaskspread() * (optlist[i][0].K - optlist[i - 1][0].K)
                elif i == 0:
                    p1 += - 1 / (optlist[i][0].K) ** 2 * optlist[i][0].midbidaskspread() * (optlist[i + 1][0].K - optlist[i][0].K)
                    p2 += 2 / (optlist[i][0].K) ** 2 * (1 - math.log(optlist[i][0].K / f0)) * optlist[i][0].midbidaskspread() * (optlist[i + 1][0].K - optlist[i][0].K)
                    p3 += 3 / (optlist[i][0].K) ** 2 * (2 * math.log(optlist[i][0].K / f0) - math.log(optlist[i][0].K / f0) ** 2) * optlist[i][0].midbidaskspread() * (optlist[i + 1][0].K - optlist[i][0].K)
                else:
                    p1 += - 1 / (optlist[i][0].K) ** 2 * optlist[i][0].midbidaskspread() * (optlist[i + 1][0].K - optlist[i - 1][0].K) / 2
                    p2 += 2 / (optlist[i][0].K) ** 2 * (1 - math.log(optlist[i][0].K / f0)) * optlist[i][0].midbidaskspread() * (optlist[i + 1][0].K - optlist[i - 1][0].K) / 2
                    p3 += 3 / (optlist[i][0].K) ** 2 * (2 * math.log(optlist[i][0].K / f0) - math.log(optlist[i][0].K / f0) ** 2) * optlist[i][0].midbidaskspread() * (optlist[i + 1][0].K - optlist[i - 1][0].K) / 2
        if p2 - p1 ** 2 > 0:
            return (p3 - 3 * p1 * p2 + 2 * p1 ** 3) / math.sqrt((p2 - p1 ** 2) ** 3)
        else:
            return 0

    def skew(self, mat1: Maturity, mat2: Maturity): # 30-, 30+
        w = (self.T[mat2] - 1/12) / (self.T[mat2] - self.T[mat1])
        s1 = self.skew_same_T(mat1)
        s2 = self.skew_same_T(mat2)
        return 100 - 10 * (w * s1 + (1 - w) * s2)

    def set_deposit(self, opt: OptionInfo):
        opt._deposit(self.ul_yc)

class FutureData:

    def __init__(self, fty: FutureType):
        self.fty = fty
        self.Mat_to_2005 = {}
        self._2005_to_Mat = {}
        self.T = {}
        self.initT = {}
        self.P = {}
        self.P_yc = {}
        self.P_highest = {}
        self.P_lowest = {}
        self.matlist = [Maturity.M1, Maturity.M2, Maturity.Q1, Maturity.Q2]
        for mat in self.matlist:
            self.P[mat] = -1; self.P_yc[mat] = -1; self.P_highest[mat] = -1; self.P_lowest[mat] = -1

        self.getMat()
        
    def getMat(self):
        for mat in self.matlist:
            mt = Mat[self.fty][mat].month
            if mt < 10: str_month = '0' + str(mt) 
            else: str_month = str(mt)
            self.Mat_to_2005[mat] = str(Mat[self.fty][mat].year)[2:] + str_month
            self._2005_to_Mat[self.Mat_to_2005[mat]] = mat

        def num_weekend(date1: calendar.datetime.date, date2: calendar.datetime.date):
            num = 0
            oneday = calendar.datetime.timedelta(days = 1)
            date = calendar.datetime.date(date1.year, date1.month, date1.day)
            while date != date2:
                if date.weekday() == 5 or date.weekday() == 6 or date in holiday:
                    num += 1
                date += oneday
            return num

        c = calendar.Calendar(firstweekday=calendar.SUNDAY)
        t = time.localtime()
        year = t.tm_year; month = t.tm_mon; mday = t.tm_mday
        currentDate = calendar.datetime.date(year, month, mday)

        for mat in self.matlist:
            self.T[mat] = ((Mat[self.fty][mat] - currentDate).days - num_weekend(currentDate, Mat[self.fty][mat]))/244
            self.initT[mat] = self.T[mat]


class MyScreen(QMainWindow):

    def __init__(self, sty: StockType, parent=None):
        super(MyScreen, self).__init__(parent)
        self.sty = sty
        if sty == StockType.gz300:
            self.setWindowTitle('股指300监控屏')
        elif sty == StockType.etf50:
            self.setWindowTitle('50etf监控屏')
        elif sty == StockType.h300:
            self.setWindowTitle('沪300监控屏')
        elif sty == StockType.s300:
            self.setWindowTitle('深300监控屏')
        self.resize(1900, 900)
        if self.sty in [StockType.gz300, StockType.h300, StockType.s300]:
            addin = 'IF'
        elif self.sty == StockType.etf50:
            addin = 'IH'
        HorizontalHeaderLabels = ['', '标的价格', '涨跌幅', '月份', '合成期货', '升贴水', addin, '涨跌幅', '今日VIX', '昨收VIX', '今最大', '今最小', 'forward vol', 'VIX百分位', '近次月VIX差', '当月skew', '加权skew', '0.25IV差', 'skew比率', 'call', 'put', '95VIX', '105VIX']
        self.ncol = len(HorizontalHeaderLabels)
        self.nrow = 9
        self.myScreen = GraphicsLayoutWidget()
        self.setCentralWidget(self.myScreen)

        self.myTable = QTableWidget()
        self.myTable.setColumnCount(self.ncol)
        self.myTable.setRowCount(self.nrow)
        self.myTable.setHorizontalHeaderLabels(HorizontalHeaderLabels)
        self.myTable.setStyleSheet('QTableView{background-color: black; color: white;}''QTableCornerButton::section{background-color: black;}')
        self.myTable.horizontalHeader().setStyleSheet('QHeaderView::section{background-color: black; color: white;}''QHeaderView{background-color: black;}')
        self.myTable.horizontalHeader().setDefaultAlignment(Qt.AlignLeft)
        self.myTable.horizontalHeader().setFont(QFont('Times', 11, QFont.Bold))
        self.myTable.verticalHeader().setStyleSheet('QHeaderView::section{background-color: black; color: white;}''QHeaderView{background-color: black;}')
        for i in range(self.ncol):
            self.myTable.setColumnWidth(i, 75)
        self.myTable.setColumnWidth(HorizontalHeaderLabels.index(''), 100)
        self.myTable.setColumnWidth(HorizontalHeaderLabels.index('forward vol'), 85)
        self.myTable.setColumnWidth(HorizontalHeaderLabels.index('VIX百分位'), 95)
        self.myTable.setColumnWidth(HorizontalHeaderLabels.index('近次月VIX差'), 100)
        self.myTable.setColumnWidth(HorizontalHeaderLabels.index('当月skew'), 85)
        self.myTable.setColumnWidth(HorizontalHeaderLabels.index('加权skew'), 85)
        self.myTable.setColumnWidth(HorizontalHeaderLabels.index('0.25IV差'), 100)
        for i in range(self.nrow):
            self.myTable.setRowHeight(i, 5)
        self.dock = QDockWidget()
        self.dock.setWidget(self.myTable)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock)


        # vix1_vix2_vix3_vix4 & future
        self.fig_vix = PlotItem(); self.fig_vix.setTitle('vix & future', color = 'w')
        self.myScreen.addItem(self.fig_vix, row = None, col = None, rowspan = 1, colspan = 2)
        self.fig_vix.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_vix.legend.setParentItem(self.fig_vix.vb)
        self.curve_vix1 = self.fig_vix.plot(pen = '#FF5809', name = 'vix 当月')
        self.curve_vix2 = self.fig_vix.plot(pen = '#82CAFA', name = 'vix 次月')
        if sty in [StockType.etf50, StockType.h300, StockType.s300]:
            name = 'vix 季月'
        elif sty == StockType.gz300:
            name = 'vix 次次月'
        self.curve_vix3 = self.fig_vix.plot(pen = 'y', name = name)
        if sty in [StockType.etf50, StockType.h300, StockType.s300]:
            name = 'vix 次季月'
        elif sty == StockType.gz300:
            name = 'vix 季月'
        self.curve_vix4 = self.fig_vix.plot(pen = '#B7B7B7', name = name)
        self.curve_S_setName = self.fig_vix.plot(pen = 'g', name = '合成标的')

        self.fig_S = ViewBox()
        self.fig_vix.showAxis('right')
        self.fig_vix.scene().addItem(self.fig_S)
        self.fig_vix.getAxis('right').linkToView(self.fig_S)
        self.fig_S.setXLink(self.fig_vix)
        self.curve_S = PlotCurveItem(fillLevel = 0, brush = (0, 128, 0, 100), pen = 'g')
        self.fig_S.addItem(self.curve_S)
        self.updateViews()
        self.fig_vix.vb.sigResized.connect(self.updateViews) 

        # forward_vol
        self.fig_forward_vol = PlotItem(); self.fig_forward_vol.setTitle('Forward VOL', color = 'w')
        self.myScreen.addItem(self.fig_forward_vol, row = None, col = None, rowspan = 1, colspan = 1)
        self.fig_forward_vol.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_forward_vol.legend.setParentItem(self.fig_forward_vol.vb)
        self.curve_forward_vol1 = self.fig_forward_vol.plot(pen = '#FF5809', name = '近月 (1M)')
        if sty in [StockType.etf50, StockType.h300, StockType.s300]:
            name = '中月'
            if Mat[sty][Maturity.Q1].month - Mat[sty][Maturity.M2].month > 1:
                name += ' (3M)'
            else:
                name += ' (1M)'
        elif sty == StockType.gz300:
            name = '次近月 (1M)'
        self.curve_forward_vol2 = self.fig_forward_vol.plot(pen = '#82CAFA', name = name)
        if sty in [StockType.etf50, StockType.h300, StockType.s300]:
            name = '远月 (3M)'
        elif sty == StockType.gz300:
            name = '中月'
            if Mat[sty][Maturity.Q1].month - Mat[sty][Maturity.M3].month > 1:
                name += ' (3M)'
            else:
                name += ' (1M)'
        self.curve_forward_vol3 = self.fig_forward_vol.plot(pen = 'y', name = name)
        if sty == StockType.gz300:
            self.curve_forward_vol4 = self.fig_forward_vol.plot(pen = 'g', name = '远月 (3M)')
            self.curve_forward_vol5 = self.fig_forward_vol.plot(pen = '#B7B7B7', name = '次远月 (3M)')

        # skew1_skew2
        self.fig_skew = PlotItem(); self.fig_skew.setTitle('skew 当月 & 次月', color = 'w')
        self.myScreen.addItem(self.fig_skew, row = None, col = None, rowspan = 1, colspan = 1)
        self.fig_skew.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_skew.legend.setParentItem(self.fig_skew.vb)
        self.curve_skew1 = self.fig_skew.plot(pen = '#FF5809', name = 'skew 当月')
        self.curve_skew2 = self.fig_skew.plot(pen = '#82CAFA', name = 'skew 次月')

        self.myScreen.nextRow()

        # smile1
        self.fig_smile1 = PlotItem(); self.fig_smile1.setTitle('skew 当月', color = 'w')
        self.myScreen.addItem(self.fig_smile1, row = None, col = None, rowspan = 1, colspan = 1)
        self.fig_smile1.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_smile1.legend.setParentItem(self.fig_smile1.vb)
        self.curve_smile1 = self.fig_smile1.plot(pen = '#FF5809', symbolBrush = '#FF5809', symbolPen = '#FF5809', name = '今')
        self.curve_smile1_y = self.fig_smile1.plot(pen = '#82CAFA', symbolBrush = '#82CAFA', symbolPen = '#82CAFA', name = '昨')

        # smile2
        self.fig_smile2 = PlotItem(); self.fig_smile2.setTitle('skew 次月', color = 'w')
        self.myScreen.addItem(self.fig_smile2, row = None, col = None, rowspan = 1, colspan = 1)
        self.fig_smile2.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_smile2.legend.setParentItem(self.fig_smile2.vb)
        self.curve_smile2 = self.fig_smile2.plot(pen = '#FF5809', symbolBrush = '#FF5809', symbolPen = '#FF5809', name = '今')
        self.curve_smile2_y = self.fig_smile2.plot(pen = '#82CAFA', symbolBrush = '#82CAFA', symbolPen = '#82CAFA', name = '昨')

        # vix 期限结构
        self.fig_vix_tsir = PlotItem(); self.fig_vix_tsir.setTitle('VIX 期限结构', color = 'w')
        self.myScreen.addItem(self.fig_vix_tsir, row = None, col = None, rowspan = 1, colspan = 1)
        self.fig_vix_tsir.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_vix_tsir.legend.setParentItem(self.fig_vix_tsir.vb)
        self.curve_vix_tsir_today = self.fig_vix_tsir.plot(pen = 'y', symbolBrush = 'y', symbolPen = 'y', name = '今日')
        self.curve_vix_tsir_yesterday = self.fig_vix_tsir.plot(pen = '#82CAFA', symbolBrush = '#82CAFA', symbolPen = '#82CAFA', name = '昨收')
        self.curve_vix_tsir_max = self.fig_vix_tsir.plot(pen = 'r', symbolBrush = 'r', symbolPen = 'r', name = '最大值')
        self.curve_vix_tsir_min = self.fig_vix_tsir.plot(pen = 'g', symbolBrush = 'g', symbolPen = 'g', name = '最小值')

        # vix interpolate
        self.fig_vix_interpolate = PlotItem(); self.fig_vix_interpolate.setTitle('VIX 期限结构', color = 'w')
        self.myScreen.addItem(self.fig_vix_interpolate, row = None, col = None, rowspan = 1, colspan = 1)
        self.fig_vix_interpolate.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_vix_interpolate.legend.setParentItem(self.fig_vix_interpolate.vb)
        self.curve_vix100 = self.fig_vix_interpolate.plot(pen = 'y', symbolBrush = 'y', symbolPen = 'y', name = '今日')
        self.curve_vix105 = self.fig_vix_interpolate.plot(pen = 'r', symbolBrush = 'r', symbolPen = 'r', name = '105VIX')
        self.curve_vix95 = self.fig_vix_interpolate.plot(pen = 'g', symbolBrush = 'g', symbolPen = 'g', name = '95VIX')

        for orientation in ['left', 'bottom', 'right']:
            for fig in [self.fig_vix, self.fig_forward_vol, self.fig_skew, self.fig_smile1, self.fig_smile2, self.fig_vix_tsir, self.fig_vix_interpolate]:
                fig.getAxis(orientation).setPen('w')
                fig.getAxis(orientation).setTextPen('w')
                #fig.showGrid(x = False,y = True)


        self.timeList = [['09:30', 0], ['10:00', 0], ['10:30', 0], ['11:00', 0], ['11:30', 0], ['13:00', 0], ['13:30', 0], ['14:00', 0], ['14:30', 0]]
        self.timeShow: bool = False

        self.tstr: list = []
        self.vix_tsir: list = []
        if sty == StockType.gz300:
            self.vix_tsir = ['M1', 'M2', 'M3', 'Q1', 'Q2', 'Q3']
        elif sty in [StockType.etf50, StockType.h300, StockType.s300]:
            self.vix_tsir = ['M1', 'M2', 'Q1', 'Q2']
        self.S: list = []
        self.vix1: list = []
        self.vix2: list = []
        self.vix3: list = []
        self.vix4: list = []
        self.forward_vol1: list = []
        self.forward_vol2: list = []
        self.forward_vol3: list = []
        self.forward_vol4: list = []
        self.forward_vol5: list = []
        self.skew1: list = []
        self.skew2: list = []
        self.smile1_x: list = []
        self.smile1_y_x: list = []
        self.smile1: list = []
        self.smile1_y: list = []
        self.smile2_x: list = []
        self.smile2_y_x: list = []
        self.smile2: list = []
        self.smile2_y: list = []
        self.vix_tsir_today: list = []
        self.vix_tsir_yesterday: list = []
        self.vix_tsir_max: list = []
        self.vix_tsir_min: list = []
        self.vix105: list = []
        self.vix95: list = []


    def updateViews(self):
        self.fig_S.setGeometry(self.fig_vix.vb.sceneBoundingRect())
        self.fig_S.linkedViewChanged(self.fig_vix.vb, self.fig_S.XAxis)

    def plot(self, new_data: tuple):

        self.S.append(new_data[1][0])
        self.vix1.append(new_data[1][1])
        self.vix2.append(new_data[1][2])
        self.vix3.append(new_data[1][3])
        self.vix4.append(new_data[1][4])
        self.forward_vol1.append(new_data[2][0])
        self.forward_vol2.append(new_data[2][1])
        self.forward_vol3.append(new_data[2][2])
        if self.sty == StockType.gz300:
            self.forward_vol4.append(new_data[2][3])
            self.forward_vol5.append(new_data[2][4])
        self.skew1.append(new_data[3][0])
        self.skew2.append(new_data[3][1])
        self.smile1_x = new_data[4][0]
        self.smile1 = new_data[4][1]
        self.smile1_y_x = new_data[4][2]
        self.smile1_y = new_data[4][3]
        self.smile2_x = new_data[5][0]
        self.smile2 = new_data[5][1]
        self.smile2_y_x = new_data[5][2]
        self.smile2_y = new_data[5][3]
        self.vix_tsir_today = new_data[6][0]
        self.vix_tsir_yesterday = new_data[6][1]
        self.vix_tsir_max = new_data[6][2]
        self.vix_tsir_min = new_data[6][3]
        self.vix105 = new_data[7][0]
        self.vix95 = new_data[7][1]

        # table
        for i in range(len(new_data[0])):
            for j in range(len(new_data[0][i])):
                newItem = QTableWidgetItem(str(new_data[0][i][j])); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(j, i, newItem)
        for i in range(2):
            if str(new_data[0][1][1 + i])[-1] == '1':
                newItem = QTableWidgetItem(str(new_data[0][1][1 + i])[:-3]); newItem.setBackground(QBrush(QColor(220, 20, 60))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(1 + i, 1, newItem)
            else:
                newItem = QTableWidgetItem(str(new_data[0][1][1 + i])[:-3]); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(1 + i, 1, newItem)
        if new_data[0][1][3] == 'WARNING':
            newItem = QTableWidgetItem(str(new_data[0][1][3])); newItem.setBackground(QBrush(QColor(220, 20, 60))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(3, 1, newItem)
        if not new_data[0][1][4] == ' ':
            if new_data[0][1][4][2]:
                newItem = QTableWidgetItem(str(new_data[0][1][4][0])); newItem.setBackground(QBrush(QColor(255, 165, 0))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(4, 1, newItem)
            else:
                newItem = QTableWidgetItem(str(new_data[0][1][4][0])); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(4, 1, newItem)
            if new_data[0][1][4][1]:
                newItem = QTableWidgetItem(str(new_data[0][1][4][0])); newItem.setBackground(QBrush(QColor(220, 20, 60))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(4, 1, newItem)
            else:
                newItem = QTableWidgetItem(str(new_data[0][1][4][0])); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(4, 1, newItem)

        # graph
        tstr_ = time.strftime('%H:%M:%S', time.localtime())
        if self.timeShow == False and self.tstr != []:
            self.tstr[-1] = ''
        if [tstr_[:-3], 0] in self.timeList:
            self.timeShow = True
            self.timeList[self.timeList.index([tstr_[:-3], 0])][1] = 1
        else:
            self.timeShow = False
        self.tstr.append(tstr_)

        for curve_data in [(self.curve_S, self.S), (self.curve_vix1, self.vix1), (self.curve_vix2, self.vix2), (self.curve_vix3, self.vix3), (self.curve_vix4, self.vix4), (self.curve_forward_vol1, self.forward_vol1), (self.curve_forward_vol2, self.forward_vol2), (self.curve_forward_vol3, self.forward_vol3), (self.curve_skew1, self.skew1), (self.curve_skew2, self.skew2), (self.curve_vix_tsir_today, self.vix_tsir_today), (self.curve_vix_tsir_yesterday, self.vix_tsir_yesterday), (self.curve_vix_tsir_max, self.vix_tsir_max), (self.curve_vix_tsir_min, self.vix_tsir_min), (self.curve_vix100, self.vix_tsir_today), (self.curve_vix105, self.vix105), (self.curve_vix95, self.vix95)]:
            curve_data[0].setData(curve_data[1])
        self.curve_smile1.setData(self.smile1_x, self.smile1)
        self.curve_smile1_y.setData(self.smile1_y_x, self.smile1_y)
        self.curve_smile2.setData(self.smile2_x, self.smile2)
        self.curve_smile2_y.setData(self.smile2_y_x, self.smile2_y)
        if self.sty == StockType.gz300:
            self.curve_forward_vol4.setData(self.forward_vol4)
            self.curve_forward_vol5.setData(self.forward_vol5)

        for fig in [self.fig_vix, self.fig_forward_vol, self.fig_skew]:
            fig.getAxis('bottom').setTicks([list(zip(range(len(self.tstr)), self.tstr))])
        self.fig_vix_tsir.getAxis('bottom').setTicks([list(zip(range(len(self.vix_tsir)), self.vix_tsir))])
        self.fig_vix_interpolate.getAxis('bottom').setTicks([list(zip(range(len(self.vix_tsir)), self.vix_tsir))])
        for fig in [self.fig_vix, self.fig_forward_vol, self.fig_smile1, self.fig_smile2, self.fig_vix_tsir, self.fig_vix_interpolate]:
            fig.getAxis('left').setTicks([[(value, '{:.2%}'.format(value)) for value in [0.25 + 0.02 * i for i in range(-50, 50)]], []])

        space = 4 * 60 * 60 / (freq_for_screen / 1000)
        for fig in [self.fig_vix, self.fig_forward_vol, self.fig_skew]:
            fig.setXRange(max = space, min = 0)

class update_screen_data(QThread):

    signal = pyqtSignal(tuple)

    def __init__(self, sty: StockType, parent=None):
        super(update_screen_data, self).__init__(parent)
        self.sty = sty
        if self.sty in [StockType.gz300, StockType.h300, StockType.s300]:
            self.fty = FutureType.IF
        elif self.sty == StockType.etf50:
            self.fty = FutureType.IH
        self.data: dict = {}
        self.myScreen: MyScreen

        self.data[sty] = OptData(sty)
        for mat in self.data[sty].matlist:
            self.data[sty].subscribe_init(mat)
        for fty in [FutureType.IF, FutureType.IH]:
            self.data[fty] = FutureData(fty)
        self.myScreen = MyScreen(sty)
        self.myScreen.show()
        self.signal.connect(self.myScreen.plot)
        
        self.smile_x: dict = {}
        self.smile_interpolate_x = [0.92, 0.94, 0.96, 0.98, 1, 1.02, 1.04, 1.06, 1.08]
        for mat in self.data[sty].matlist:
            self.smile_x[mat] = []
            for i in range(len(self.data[sty].OptionList[mat])):
                self.smile_x[mat].append(self.data[sty].OptionList[mat][i][0].K)

        self.ul_chg = 1
        self.ft_chg: dict = {}
        for fty in [FutureType.IF, FutureType.IH]:
            for mat in [Maturity.M1, Maturity.M2, Maturity.Q1, Maturity.Q2]:
                self.ft_chg[(fty, mat)] = 1

    def run(self):
        t = time.localtime(); str_year = str(t.tm_year); str_month = str(t.tm_mon); str_day = str(t.tm_mday)
        name = type_to_str[self.sty]
        csvname = f'{str_year}-{str_month}-{str_day}-{name}监控屏数据记录.csv'
        f_0 = open(csvname,'a',newline='')
        f_0_w=csv.writer(f_0)
        if self.sty in [StockType.etf50, StockType.h300, StockType.s300]:
            f_0_w.writerow(['Time', '合成期货', 'VIX (M1)', 'VIX (M2)', 'VIX (Q1)', 'VIX (Q2)', 'skew (M1)', 'skew (M2)', 'skew (Q1)', 'skew (Q2)'])
        elif self.sty == StockType.gz300:
            f_0_w.writerow(['Time', '合成期货', 'VIX (M1)', 'VIX (M2)', 'VIX (M3)', 'VIX (Q1)', 'VIX (Q2)', 'VIX (Q3)', 'skew (M1)', 'skew (M2)', 'skew (M3)', 'skew (Q1)', 'skew (Q2)', 'skew (Q3)'])
        f_0.flush()

        vix_max: dict = {}
        vix_min: dict = {}
        vix: dict = {}
        forward_vol: dict = {}
        skew: dict = {}
        pciv_call_posi: dict = {}
        pciv_put_posi: dict = {}
        pciv: dict = {}
        plot_period: bool = False
        all_written: bool  = False
        csv_read: bool = False
        for mat in self.data[self.sty].matlist:
            vix_max[mat] = -5
            vix_min[mat] = 5

        while True:
            t = time.localtime(); hour = t.tm_hour; _min = t.tm_min; sec = t.tm_sec

            if (hour == 9 and _min >= 30) or (hour == 10) or (hour == 11 and _min < 30) or (hour == 11 and _min == 30 and sec == 0) or (hour == 13) or (hour == 14) or (hour == 15 and _min < 2):
                plot_period = True
            else:
                plot_period = False
            
            time_wait_for_opt_all_written = time.time()
            while all_written == False:
                n_written = 0
                n_all_options = 0
                for mat in self.data[self.sty].matlist:
                    n_all_options += 2 * len(self.data[self.sty].OptionList[mat])
                    for i in range(len(self.data[self.sty].OptionList[mat])):
                        if self.data[self.sty].OptionList[mat][i][0].written == True:
                            n_written += 1
                        if self.data[self.sty].OptionList[mat][i][1].written == True:
                            n_written += 1
                if n_written == n_all_options or time.time() - time_wait_for_opt_all_written > 30:
                    all_written = True

            # hist_data_init
            if csv_read == False:
                def variable_readin():
                    csvname = f'./{name}-单屏_hist.csv'
                    f = open(csvname)
                    reader = csv.reader(f, delimiter = ',')
                    hist: dict = {}
                    def str_to_enum(l: list):
                        new_l = []
                        for i in l:
                            if str(i) in ['etf50', 'h300', 'gz300', 's300', 'IF', 'IH', 'M1', 'M2', 'M3', 'Q1', 'Q2', 'Q3']:
                                new_l.append(str_to_type[str(i)])
                            else:
                                new_l.append(str(i))
                        return new_l
                    for z, row in enumerate(reader):
                        for i in range(len(row)):
                            if row[i] == 'reading_end':
                                end_posi = i
                        if row[0][:3] == 'sgl':
                            hist[tuple(str_to_enum(row[:end_posi]))] = float(row[end_posi + 1])
                        elif row[0][:3] == 'seq':
                            hist[tuple(str_to_enum(row[:end_posi]))] = [float(x) for x in row[end_posi + 1:] if x != '']
                    f.close()
                    return hist

                exist = os.path.isfile(f'./{name}-单屏_hist.csv')
                if exist == False:
                    csvname = f'{name}-单屏_hist.csv'
                    f = open(csvname,'w',newline='')
                    f_w = csv.writer(f)
                    if self.sty in [StockType.etf50, StockType.h300, StockType.s300]:
                        f_w.writerow(['seq_ul_close', name, 'reading_end'] + [1] * 50)
                    elif self.sty == StockType.gz300:
                        f_w.writerow(['seq_close', 'IF', 'M1', 'reading_end'] + [1] * 50)
                    for mat in self.data[self.sty].matlist:
                        f_w.writerow(['sgl_vix', name, type_to_str[mat], 'l1', 'reading_end', 0.25])
                        f_w.writerow(['sgl_vix', name, type_to_str[mat], 'l2', 'reading_end', 0.25])
                        f_w.writerow(['sgl_skew', name, type_to_str[mat], 'l1', 'reading_end', 100])
                        f_w.writerow(['sgl_S', name, type_to_str[mat], 'l1', 'reading_end', 1])
                    for mat in self.data[self.sty].matlist[:-1]:
                        f_w.writerow(['sgl_forward_vol', name, type_to_str[mat], 'l1', 'reading_end', 0.25])
                    if self.sty in [StockType.etf50, StockType.h300, StockType.s300]:
                        f_w.writerow(['seq_ul_min', name, 'reading_end'] + [1] * 50)
                        f_w.writerow(['seq_ul_max', name, 'reading_end'] + [1] * 50)
                    elif self.sty == StockType.gz300:
                        f_w.writerow(['seq_min', 'IF', 'M1', 'reading_end'] + [1] * 50)
                        f_w.writerow(['seq_max', 'IF', 'M1', 'reading_end'] + [1] * 50)
                    x = [self.data[self.sty].OptionList[Maturity.M1][i][0].K for i in range(len(self.data[self.sty].OptionList[Maturity.M1]))]
                    f_w.writerow(['seq_smile_y_x', name, 'M1', 'reading_end'] + x)
                    f_w.writerow(['seq_smile_y', name, 'M1', 'reading_end'] + [0.25] * len(x))
                    x = [self.data[self.sty].OptionList[Maturity.M2][i][0].K for i in range(len(self.data[self.sty].OptionList[Maturity.M2]))]
                    f_w.writerow(['seq_smile_y_x', name, 'M2', 'reading_end'] + x)
                    f_w.writerow(['seq_smile_y', name, 'M2', 'reading_end'] + [0.25] * len(x))
                    f_w.writerow(['seq_vix', name, 'reading_end'] + [0.25] * 50)
                    f.flush()
                    f.close()
                hist = variable_readin()
                csv_read = True


            if all_written and csv_read and plot_period:

                change_mat_8 = False
                if (Mat[self.sty][Maturity.M1] - calendar.datetime.date(t.tm_year, t.tm_mon, t.tm_mday)).days <= 10:
                    change_mat_8 = True

                vix_yc_plus_01 = hist[('sgl_vix', self.sty, Maturity.M1, 'l1')] * (1 - change_mat_8) + hist[('sgl_vix', self.sty, Maturity.M2, 'l1')] * change_mat_8 + 0.01
                future_mean = np.mean([self.data[self.sty].OptionList[Maturity.M1][posi][0].P_yc - self.data[self.sty].OptionList[Maturity.M1][posi][1].P_yc + self.data[self.sty].OptionList[Maturity.M1][posi][0].K for posi in range(len(self.data[self.sty].OptionList[Maturity.M1]))])
                if self.sty == StockType.gz300:
                    [warning_upper, warning_lower] = [np.round(future_mean * (1 + sign * 0.9 * (vix_yc_plus_01 - 0.01) / 15.6), 1) for sign in [1, -1]]
                else:
                    [warning_upper, warning_lower] = [np.round(future_mean * (1 + sign * 0.9 * (vix_yc_plus_01 - 0.01) / 15.6), 3) for sign in [1, -1]]

                for mat in self.data[self.sty].matlist:
                    vix[mat] = self.data[self.sty].vix(mat)
                    if not ((hour == 9 and _min == 30) or (hour == 11 and _min == 29) or (hour == 11 and _min == 30) or (hour == 13 and _min == 0)):
                        vix_max[mat] = max(vix_max[mat], vix[mat]) * (vix[mat] < 1.4 * hist[('sgl_vix', self.sty, mat, 'l1')]) + vix_max[mat] * (vix[mat] >= 1.4 * hist[('sgl_vix', self.sty, mat, 'l1')])
                        vix_min[mat] = min(vix_min[mat], vix[mat]) * (vix[mat] > 0.6 * hist[('sgl_vix', self.sty, mat, 'l1')]) + vix_min[mat] * (vix[mat] <= 0.6 * hist[('sgl_vix', self.sty, mat, 'l1')])
                    skew[mat] = 100 - 10 * self.data[self.sty].skew_same_T(mat)
                    pciv_call_posi[mat] = np.argmin(abs(np.array([self.data[self.sty].OptionList[mat][i][0].delta() for i in range(len(self.data[self.sty].OptionList[mat]))]) - 0.25))
                    pciv_put_posi[mat] = np.argmin(abs(np.array([self.data[self.sty].OptionList[mat][i][1].delta() for i in range(len(self.data[self.sty].OptionList[mat]))]) - (-0.25)))
                    pciv[mat] = self.data[self.sty].OptionList[mat][pciv_put_posi[mat]][1].iv() - self.data[self.sty].OptionList[mat][pciv_call_posi[mat]][0].iv()

                content_upper = str((self.data[self.sty].S[Maturity.M1] > warning_upper) + 0)
                content_lower = str((self.data[self.sty].S[Maturity.M1] < warning_lower) + 0)
                warning_upper_str = str(warning_upper) + ', ' + content_upper
                warning_lower_str = str(warning_lower) + ', ' + content_lower

                if not change_mat_8:
                    if vix[Maturity.M1] > hist[('sgl_vix', self.sty, Maturity.M1, 'l1')] + 0.01 and hist[('sgl_vix', self.sty, Maturity.M1, 'l1')] > hist[('sgl_vix', self.sty, Maturity.M1, 'l2')] + 0.01:
                        vix_warning = 'WARNING'
                    else:
                        vix_warning = 'n/a'
                else:
                    if vix[Maturity.M2] > hist[('sgl_vix', self.sty, Maturity.M2, 'l1')] + 0.01 and hist[('sgl_vix', self.sty, Maturity.M2, 'l1')] > hist[('sgl_vix', self.sty, Maturity.M2, 'l2')] + 0.01:
                        vix_warning = 'WARNING'
                    else:
                        vix_warning = 'n/a'

                if self.sty in [StockType.etf50, StockType.h300, StockType.s300]:
                    vix_mat_lable = ['M1', 'M2', 'Q1', 'Q2']
                    ir = np.array(hist[('seq_ul_close', self.sty)][-18:] + [self.data[self.sty].ul_yc] + [self.data[self.sty].ul]) / np.array(hist[('seq_ul_close', self.sty)][-19:] + [self.data[self.sty].ul_yc]) - 1
                    price = hist[('seq_ul_close', self.sty)] + [self.data[self.sty].ul_yc]
                    price_min = hist[('seq_ul_min', self.sty)] + [self.data[self.sty].ul_lowest]
                    price_max = hist[('seq_ul_max', self.sty)] + [self.data[self.sty].ul_highest]
                elif self.sty == StockType.gz300:
                    vix_mat_lable = ['M1', 'M2', 'M3', 'Q1', 'Q2', 'Q3']
                    ir = np.array(hist[('seq_close', FutureType.IF, Maturity.M1)][-18:] + [self.data[FutureType.IF].P_yc[Maturity.M1]] + [self.data[FutureType.IF].P[Maturity.M1]]) / np.array(hist[('seq_close', FutureType.IF, Maturity.M1)][-19:] + [self.data[FutureType.IF].P_yc[Maturity.M1]]) - 1
                    price = hist[('seq_close', FutureType.IF, Maturity.M1)] + [self.data[FutureType.IF].P_yc[Maturity.M1]]
                    price_min = hist[('seq_min', FutureType.IF, Maturity.M1)] + [self.data[FutureType.IF].P_lowest[Maturity.M1]]
                    price_max = hist[('seq_max', FutureType.IF, Maturity.M1)] + [self.data[FutureType.IF].P_highest[Maturity.M1]]

                array = [0] * (len(ir) + 1)
                array[0] = np.std(ir, ddof=1) ** 2
                lb = 0.94
                for i in range(1, len(array)):
                    array[i] = lb * array[i - 1] + (1 - lb) * ir[i - 1] ** 2
                hv20 = np.sqrt(array[-1] * 244)

                trueRange_2 = [(max(price[i], price_min[i], price_max[i]) - min(price[i], price_min[i], price_max[i])) / price[i] for i in range(-10, 0)]
                trueRange_2_mean = np.mean(trueRange_2)
                trueRange_2_l1 = [(max(price[i], price_min[i], price_max[i]) - min(price[i], price_min[i], price_max[i])) / price[i] for i in range(-10 - 1, -1)]
                trueRange_2_l1_mean = np.mean(trueRange_2_l1)
                trueRange_2_l2 = [(max(price[i], price_min[i], price_max[i]) - min(price[i], price_min[i], price_max[i])) / price[i] for i in range(-10 - 2, -2)]
                trueRange_2_l2_mean = np.mean(trueRange_2_l2)
                trueRange_1 = [(max(price[i], price_min[i], price_max[i]) - min(price[i], price_min[i], price_max[i])) / price[i] for i in range(-5, 0)]
                trueRange_1_mean = np.mean(trueRange_1)
                trueRange_1_l1 = [(max(price[i], price_min[i], price_max[i]) - min(price[i], price_min[i], price_max[i])) / price[i] for i in range(-5 - 1, -1)]
                trueRange_1_l1_mean = np.mean(trueRange_1_l1)
                trueRange_1_l2 = [(max(price[i], price_min[i], price_max[i]) - min(price[i], price_min[i], price_max[i])) / price[i] for i in range(-5 - 2, -2)]
                trueRange_1_l2_mean = np.mean(trueRange_1_l2)
                trueRange_today = (max(price[-1], price_min[-1], price_max[-1]) - min(price[-1], price_min[-1], price_max[-1])) / price[-1]

                for i, mat in enumerate(self.data[self.sty].matlist[:-1]):
                    next_mat = self.data[self.sty].matlist[i+1]
                    forward_vol[mat] = (vix[next_mat] * self.data[self.sty].T[next_mat] - vix[mat] * self.data[self.sty].T[mat]) / (self.data[self.sty].T[next_mat] - self.data[self.sty].T[mat])

                smile_y = {}
                smile_cs = {}
                for mat in self.data[self.sty].matlist:
                    smile_y[mat] = []
                    for i in range(len(self.smile_x[mat])):
                        if self.data[self.sty].OptionList[mat][i][0].K < self.data[self.sty].S[mat]:
                            smile_y[mat].append(self.data[self.sty].OptionList[mat][i][1].iv())
                        elif self.data[self.sty].OptionList[mat][i][0].K == self.data[self.sty].S[mat]:
                            smile_y[mat].append(self.data[self.sty].OptionList[mat][i][0].iv_s(self.data[self.sty].OptionList[mat][i][0].midbidaskspread() - self.data[self.sty].OptionList[mat][i][1].midbidaskspread() + self.data[self.sty].OptionList[mat][i][0].K))
                        else:
                            smile_y[mat].append(self.data[self.sty].OptionList[mat][i][0].iv())
                    smile_cs[mat] = CubicSpline(self.smile_x[mat], smile_y[mat])

                vix105 = [smile_cs[mat](1.05 * self.data[self.sty].S[mat]) for mat in self.data[self.sty].matlist]
                vix95 = [smile_cs[mat](0.95 * self.data[self.sty].S[mat]) for mat in self.data[self.sty].matlist]
                not_zero = self.data[self.sty].OptionList[Maturity.M1][self.data[self.sty].posi[Maturity.M1]][0].gamma() * 0.01 * self.data[self.sty].ul ** 2 * self.data[self.sty].cm * 2
                m_gamma = 1000000 / (not_zero * (not_zero != 0) + 1 * (not_zero == 0))
                not_zero = self.data[self.sty].OptionList[Maturity.M1][self.data[self.sty].posi[Maturity.M1]][0].vega() * self.data[self.sty].cm * 0.01 * 2
                t_vega = 1000 / (not_zero * (not_zero != 0) + 1 * (not_zero == 0))
                atm_call_p = self.data[self.sty].OptionList[Maturity.M1][self.data[self.sty].posi[Maturity.M1]][0].P * self.data[self.sty].cm
                atm_put_p = self.data[self.sty].OptionList[Maturity.M1][self.data[self.sty].posi[Maturity.M1]][1].P * self.data[self.sty].cm
                self.data[self.sty].OptionList[Maturity.M1][self.data[self.sty].posi[Maturity.M1]][0]._deposit(self.data[self.sty].ul_yc)
                atm_call_deposit = self.data[self.sty].OptionList[Maturity.M1][self.data[self.sty].posi[Maturity.M1]][0].deposit
                self.data[self.sty].OptionList[Maturity.M1][self.data[self.sty].posi[Maturity.M1]][1]._deposit(self.data[self.sty].ul_yc)
                atm_put_deposit = self.data[self.sty].OptionList[Maturity.M1][self.data[self.sty].posi[Maturity.M1]][1].deposit

                def rd_pc(x: list):
                    if len(x) != 1:
                        return ['{:.2%}'.format(i) for i in x]
                    else:
                        return '{:.2%}'.format(x[0])

                table = []
                table.append(['', '警戒线 (高)', '警戒线 (低)', '日波预警', '昨收VIX+1%'])
                if self.sty == StockType.gz300:
                    table.append(['{:.1f}'.format(self.data[self.sty].ul)] + [warning_upper_str, warning_lower_str] + [vix_warning] + [(rd_pc([vix_yc_plus_01]), (vix_yc_plus_01 < vix[Maturity.M1]) * (1 - change_mat_8) + (vix_yc_plus_01 < vix[Maturity.M2]) * change_mat_8, (vix_yc_plus_01 < vix_max[Maturity.M1]) * (1 - change_mat_8) + (vix_yc_plus_01 < vix_max[Maturity.M2]) * change_mat_8)])
                else:
                    table.append(['{:.3f}'.format(self.data[self.sty].ul)] + [warning_upper_str, warning_lower_str] + [vix_warning] + [(rd_pc([vix_yc_plus_01]), (vix_yc_plus_01 < vix[Maturity.M1]) * (1 - change_mat_8) + (vix_yc_plus_01 < vix[Maturity.M2]) * change_mat_8, (vix_yc_plus_01 < vix_max[Maturity.M1]) * (1 - change_mat_8) + (vix_yc_plus_01 < vix_max[Maturity.M2]) * change_mat_8)])
                table.append([rd_pc([self.ul_chg])])
                table.append(vix_mat_lable)
                if self.sty == StockType.gz300:
                    table.append(['{:.1f}'.format(self.data[self.sty].S[mat]) for mat in self.data[self.sty].matlist])
                else:
                    table.append(['{:.3f}'.format(self.data[self.sty].S[mat]) for mat in self.data[self.sty].matlist])
                table.append(rd_pc([(self.data[self.sty].S[mat] - self.data[self.sty].ul) / self.data[self.sty].ul / self.data[self.sty].T[mat] for mat in self.data[self.sty].matlist]))
                if self.sty in [StockType.etf50, StockType.h300, StockType.s300]:
                    table.append(['{:.1f}'.format(self.data[self.fty].P[mat]) for mat in [Maturity.M1, Maturity.M2, Maturity.Q1, Maturity.Q2]])
                    table.append(rd_pc([self.ft_chg[(self.fty, mat)] for mat in [Maturity.M1, Maturity.M2, Maturity.Q1, Maturity.Q2]]) + ['HV'])
                elif self.sty == StockType.gz300:
                    if Mat[StockType.gz300][Maturity.M3].month == Mat[FutureType.IF][Maturity.Q1].month:
                        table.append(['{:.1f}'.format(self.data[self.fty].P[mat]) for mat in [Maturity.M1, Maturity.M2, Maturity.Q1, Maturity.Q2]])
                        table.append(rd_pc([self.ft_chg[(self.fty, mat)] for mat in [Maturity.M1, Maturity.M2]]) + rd_pc([self.ft_chg[(self.fty, mat)] for mat in [Maturity.Q1, Maturity.Q2]]) + ['', '', 'HV'])
                    else:
                        table.append(['{:.1f}'.format(self.data[self.fty].P[mat]) for mat in [Maturity.M1, Maturity.M2]] + [''] + ['{:.1f}'.format(self.data[self.fty].P[mat]) for mat in [Maturity.Q1, Maturity.Q2]])
                        table.append(rd_pc([self.ft_chg[(self.fty, mat)] for mat in [Maturity.M1, Maturity.M2]]) + [''] + rd_pc([self.ft_chg[(self.fty, mat)] for mat in [Maturity.Q1, Maturity.Q2]]) + ['', 'HV'])
                table.append(rd_pc([vix[mat] for mat in self.data[self.sty].matlist] + [hv20]))
                table.append(rd_pc([hist[('sgl_vix', self.sty, mat, 'l1')] for mat in self.data[self.sty].matlist]))
                table.append(rd_pc([vix_max[mat] for mat in self.data[self.sty].matlist]))
                table.append(rd_pc([vix_min[mat] for mat in self.data[self.sty].matlist]))
                table.append(rd_pc([forward_vol[mat] for mat in self.data[self.sty].matlist[:-1]]))
                table.append([rd_pc([len([0 for i in hist[('seq_vix', self.sty)] if i < vix[Maturity.M1]]) / len(hist[('seq_vix', self.sty)])])] + ['历史80分位'] + [rd_pc([np.percentile(hist[('seq_vix', self.sty)], 80)])] + ['TR80分位'] + [rd_pc([np.percentile(trueRange_2, 80)])] + ['昨ATR10'] + [rd_pc([trueRange_2_l1_mean])] + ['前ATR10'] + [rd_pc([trueRange_2_l2_mean])])
                table.append([rd_pc([vix[Maturity.M1] - vix[Maturity.M2]])] + ['', '', '今TR'] + [rd_pc([trueRange_today])] + ['昨ATR5'] + [rd_pc([trueRange_1_l1_mean])] + ['前ATR5'] + [rd_pc([trueRange_1_l2_mean])])
                table.append(['{:.1f}'.format(skew[Maturity.M1])] + ['次月skew'] + ['{:.1f}'.format(skew[Maturity.M2])] + ['M$gm组数'] + ['{:.0f}'.format(m_gamma)] + ['T$vg组数'] + ['{:.0f}'.format(t_vega)])
                table.append(['{:.1f}'.format(self.data[self.sty].skew(Maturity.M1, Maturity.M2))] + ['五档skew'] + ['{:.1f}'.format(100 - 10 * self.data[self.sty].skew_same_T_partial(Maturity.M1, 2))] + ['权利金 (万)'] + ['{:.1f}'.format(m_gamma * (atm_call_p + atm_put_p) / 10000)] + ['T$vg权利金'] + ['{:.1f}'.format(t_vega * (atm_call_p + atm_put_p) / 10000)])
                table.append([rd_pc([pciv[Maturity.M1]])] + ['0.25IV差'] + [rd_pc([pciv[Maturity.M2]])] + ['M$gm保证金'] + ['{:.1f}'.format(m_gamma * (atm_call_deposit + atm_put_deposit) / 10000)] + ['T$vg保证金'] + ['{:.1f}'.format(t_vega * (atm_call_deposit + atm_put_deposit) / 10000)])
                table.append(['{:.2%}'.format(pciv[Maturity.M1] / vix[Maturity.M1] * (1 - change_mat_8) + pciv[Maturity.M2] / vix[Maturity.M2] * change_mat_8)] + ['skew比率'] + ['{:.2%}'.format(pciv[Maturity.M2] / vix[Maturity.M2])])
                table.append([self.data[self.sty].OptionList[Maturity.M1][pciv_call_posi[Maturity.M1]][0].K] + ['call'] + [self.data[self.sty].OptionList[Maturity.M2][pciv_call_posi[Maturity.M2]][0].K])
                table.append([self.data[self.sty].OptionList[Maturity.M1][pciv_put_posi[Maturity.M1]][1].K] + ['put'] + [self.data[self.sty].OptionList[Maturity.M2][pciv_put_posi[Maturity.M2]][1].K])
                table.append(rd_pc(vix95))
                table.append(rd_pc(vix105))

                new_data = (table, [self.data[self.sty].S[Maturity.M1]] + [vix[mat] for mat in self.data[self.sty].matlist[0:4]], [forward_vol[mat] for mat in self.data[self.sty].matlist[:-1]], [skew[Maturity.M1], skew[Maturity.M2]], [self.smile_x[Maturity.M1], smile_y[Maturity.M1], hist[('seq_smile_y_x', self.sty, Maturity.M1)], hist[('seq_smile_y', self.sty, Maturity.M1)]], [self.smile_x[Maturity.M2], smile_y[Maturity.M2], hist[('seq_smile_y_x', self.sty, Maturity.M2)], hist[('seq_smile_y', self.sty, Maturity.M2)]], [[vix[mat] for mat in self.data[self.sty].matlist], [hist[('sgl_vix', self.sty, mat, 'l1')] for mat in self.data[self.sty].matlist], [vix_max[mat] for mat in self.data[self.sty].matlist], [vix_min[mat] for mat in self.data[self.sty].matlist]], [vix105, vix95])
                f_0_w.writerow([time.strftime('%Y/%m/%d %H:%M:%S', t), self.data[self.sty].S[Maturity.M1]] + [vix[mat] for mat in self.data[self.sty].matlist] + [skew[mat] for mat in self.data[self.sty].matlist])
                f_0.flush()
                self.signal.emit(new_data)
                self.msleep(freq_for_screen)

                if hour == 15 and _min == 1 and sec > 30:
                    csvname = f'{name}-单屏_hist.csv'
                    f = open(csvname,'w',newline='')
                    f_w=csv.writer(f)
                    if self.sty in [StockType.etf50, StockType.h300, StockType.s300]:
                        f_w.writerow(['seq_ul_close', name, 'reading_end'] + hist[('seq_ul_close', self.sty)][1:] + [self.data[self.sty].ul_yc])
                    elif self.sty == StockType.gz300:
                        f_w.writerow(['seq_close', 'IF', 'M1', 'reading_end'] + hist[('seq_close', FutureType.IF, Maturity.M1)][1:] + [self.data[FutureType.IF].P_yc[Maturity.M1]])
                    # 换月
                    if (Mat[self.sty][Maturity.M1] - calendar.datetime.date(t.tm_year, t.tm_mon, t.tm_mday)).days == 0:
                        curr_mat_mon = [Mat[self.sty][mat].month for mat in self.data[self.sty].matlist]
                        if abs(curr_mat_mon[len(self.data[self.sty].matlist) // 2] - curr_mat_mon[len(self.data[self.sty].matlist) // 2 - 1]) % 10 == 1:
                            next_mat_mon = [curr_mat_mon[i] % 12 + 1 for i in range(len(self.data[self.sty].matlist) // 2)] + [curr_mat_mon[i] % 12 + 3 for i in range(len(self.data[self.sty].matlist) // 2, len(self.data[self.sty].matlist), 1)]
                        else:
                            next_mat_mon = [curr_mat_mon[i] % 12 + 1 for i in range(len(self.data[self.sty].matlist) // 2)] + curr_mat_mon[len(self.data[self.sty].matlist) // 2 : ]
                    for i, mat in enumerate(self.data[self.sty].matlist):
                        if (Mat[self.sty][Maturity.M1] - calendar.datetime.date(t.tm_year, t.tm_mon, t.tm_mday)).days == 0:
                            if next_mat_mon[i] in curr_mat_mon:
                                mat0 = self.data[self.sty].matlist[curr_mat_mon.index(next_mat_mon[i])]
                                f_w.writerow(['sgl_vix', name, type_to_str[mat], 'l1', 'reading_end', vix[mat0]])
                                f_w.writerow(['sgl_vix', name, type_to_str[mat], 'l2', 'reading_end', hist[('sgl_vix', self.sty, mat0, 'l1')]])
                                f_w.writerow(['sgl_skew', name, type_to_str[mat], 'l1', 'reading_end', skew[mat0]])
                                f_w.writerow(['sgl_S', name, type_to_str[mat], 'l1', 'reading_end', self.data[self.sty].S[mat0]])
                            else:
                                f_w.writerow(['sgl_vix', name, type_to_str[mat], 'l1', 'reading_end', 0.25])
                                f_w.writerow(['sgl_vix', name, type_to_str[mat], 'l2', 'reading_end', 0.25])
                                f_w.writerow(['sgl_skew', name, type_to_str[mat], 'l1', 'reading_end', 100])
                                f_w.writerow(['sgl_S', name, type_to_str[mat], 'l1', 'reading_end', self.data[self.sty].S[mat]]) ##
                        else:
                            f_w.writerow(['sgl_vix', name, type_to_str[mat], 'l1', 'reading_end', vix[mat]])
                            f_w.writerow(['sgl_vix', name, type_to_str[mat], 'l2', 'reading_end', hist[('sgl_vix', self.sty, mat, 'l1')]])
                            f_w.writerow(['sgl_skew', name, type_to_str[mat], 'l1', 'reading_end', skew[mat]])
                            f_w.writerow(['sgl_S', name, type_to_str[mat], 'l1', 'reading_end', self.data[self.sty].S[mat]])
                    for i, mat in enumerate(self.data[self.sty].matlist[:-1]):
                        if (Mat[self.sty][Maturity.M1] - calendar.datetime.date(t.tm_year, t.tm_mon, t.tm_mday)).days == 0:
                            if next_mat_mon[i] in curr_mat_mon:
                                mat0 = self.data[self.sty].matlist[curr_mat_mon.index(next_mat_mon[i])]
                                if mat0 == self.data[self.sty].matlist[-1]:
                                    f_w.writerow(['sgl_forward_vol', name, type_to_str[mat], 'l1', 'reading_end', 0.25])
                                else:
                                    f_w.writerow(['sgl_forward_vol', name, type_to_str[mat], 'l1', 'reading_end', forward_vol[mat0]])
                            else:
                                f_w.writerow(['sgl_forward_vol', name, type_to_str[mat], 'l1', 'reading_end', 0.25])
                        else:
                            f_w.writerow(['sgl_forward_vol', name, type_to_str[mat], 'l1', 'reading_end', forward_vol[mat]])
                    if self.sty in [StockType.etf50, StockType.h300, StockType.s300]:
                        f_w.writerow(['seq_ul_min', name, 'reading_end'] + hist[('seq_ul_min', self.sty)][1:] + [self.data[self.sty].ul_lowest])
                        f_w.writerow(['seq_ul_max', name, 'reading_end'] + hist[('seq_ul_max', self.sty)][1:] + [self.data[self.sty].ul_highest])
                    elif self.sty == StockType.gz300:
                        f_w.writerow(['seq_min', 'IF', 'M1', 'reading_end'] + hist[('seq_min', FutureType.IF, Maturity.M1)][1:] + [self.data[FutureType.IF].P_lowest[Maturity.M1]])
                        f_w.writerow(['seq_max', 'IF', 'M1', 'reading_end'] + hist[('seq_max', FutureType.IF, Maturity.M1)][1:] + [self.data[FutureType.IF].P_highest[Maturity.M1]])
                    if (Mat[self.sty][Maturity.M1] - calendar.datetime.date(t.tm_year, t.tm_mon, t.tm_mday)).days == 0:
                        f_w.writerow(['seq_smile_y_x', name, 'M1', 'reading_end'] + self.smile_x[Maturity.M2])
                        f_w.writerow(['seq_smile_y', name, 'M1', 'reading_end'] + smile_y[Maturity.M2])
                        if self.sty == StockType.gz300:
                            f_w.writerow(['seq_smile_y_x', name, 'M2', 'reading_end'] + self.smile_x[Maturity.M3])
                            f_w.writerow(['seq_smile_y', name, 'M2', 'reading_end'] + smile_y[Maturity.M3])
                        else:
                            f_w.writerow(['seq_smile_y_x', name, 'M2', 'reading_end'] + self.smile_x[Maturity.M2])
                            f_w.writerow(['seq_smile_y', name, 'M2', 'reading_end'] + [0.25] * len(self.smile_x[Maturity.M2]))
                    else:
                        f_w.writerow(['seq_smile_y_x', name, 'M1', 'reading_end'] + self.smile_x[Maturity.M1])
                        f_w.writerow(['seq_smile_y', name, 'M1', 'reading_end'] + smile_y[Maturity.M1])
                        f_w.writerow(['seq_smile_y_x', name, 'M2', 'reading_end'] + self.smile_x[Maturity.M2])
                        f_w.writerow(['seq_smile_y', name, 'M2', 'reading_end'] + smile_y[Maturity.M2])
                    if not change_mat_8:
                        f_w.writerow(['seq_vix', name, 'reading_end'] + hist[('seq_vix', self.sty)] + [vix[Maturity.M1]])
                    else:
                        f_w.writerow(['seq_vix', name, 'reading_end'] + hist[('seq_vix', self.sty)] + [vix[Maturity.M2]])
                    f.flush()
                    f.close()

class CFtdcMdSpi_Screen(QThread):

    def __init__(self, sty: StockType, parent=None):
        super(CFtdcMdSpi_Screen, self).__init__(parent)
        self.obj = g_QuoteZMQ
        self.q_data = q_data
        self.sty = sty
        self.thread = update_screen_data(sty)
        self.thread.start()

        self.socket_sub = self.obj.context.socket(zmq.SUB)
        self.socket_sub.connect("tcp://127.0.0.1:%s" % self.q_data["SubPort"])
        self.socket_sub.setsockopt_string(zmq.SUBSCRIBE,"")

    def run(self):
        while(True):
            message = (self.socket_sub.recv()[:-1]).decode("utf-8")
            index =  re.search(":",message).span()[1]
            message = message[index:]
            message = json.loads(message)

            rt_data = {}

            if message["DataType"] == "REALTIME":
                QuoteID = message["Quote"]["Symbol"]

                rt_data['LastPrice'] = 0.001
                if not message["Quote"]["TradingPrice"] == "":
                    rt_data['LastPrice'] = float(message["Quote"]["TradingPrice"])

                for key in ["Bid", "Ask", "YClosedPrice", "HighPrice", "LowPrice", "Change", "ReferencePrice"]:
                    rt_data[key] = 0
                    if not message["Quote"][key] == "":
                        rt_data[key] = float(message["Quote"][key])

            elif message["DataType"] == "PING":
                self.obj.QuotePong(self.q_data["SessionKey"])
                continue
            
            else:
                continue


            if QuoteID[3] == 'O':

                #TC.O.SSE.510050.202007.C.2.8
                #TC.O.SSE.510300.202007.C.4
                #TC.O.SZSE.159919.202007.C.4
                #TC.O.CFFEX.IO.202007.P.4000

                if self.sty == StockType.gz300 and 'TC.O.CFFEX.IO' in QuoteID:
                    mat = self.thread.data[self.sty]._2005_to_Mat[QuoteID[16 : 20]]
                elif self.sty == StockType.h300 and 'TC.O.SSE.510300' in QuoteID:
                    mat = self.thread.data[self.sty]._2005_to_Mat[QuoteID[18 : 22]]
                elif self.sty == StockType.s300 and 'TC.O.SZSE.159919' in QuoteID:
                    mat = self.thread.data[self.sty]._2005_to_Mat[QuoteID[19 : 23]]
                elif self.sty == StockType.etf50 and 'TC.O.SSE.510050' in QuoteID:
                    mat = self.thread.data[self.sty]._2005_to_Mat[QuoteID[18 : 22]]
                else:
                    continue

                position = self.thread.data[self.sty].k_list[mat].index(float(QuoteID[last_C_P(QuoteID) : ]))
                if '.C.' in QuoteID:
                    se = 0
                elif '.P.' in QuoteID:
                    se = 1


                # update OptionList
                self.thread.data[self.sty].OptionList[mat][position][se].P = rt_data['LastPrice']
                self.thread.data[self.sty].OptionList[mat][position][se].bid = rt_data['Bid']
                self.thread.data[self.sty].OptionList[mat][position][se].ask = rt_data['Ask']
                self.thread.data[self.sty].OptionList[mat][position][se].settlement_price = rt_data['ReferencePrice']
                # update S, k0, posi
                self.thread.data[self.sty].S_k0_posi(mat)
                self.thread.data[self.sty].OptionList[mat][position][se].S = self.thread.data[self.sty].S[mat]
                # update time
                t = time.localtime()
                self.thread.data[self.sty].T[mat] = self.thread.data[self.sty].initT[mat] + ((15 - t.tm_hour - 1 - 1.5 * (t.tm_hour < 12)) * 60 * 60 + (60 - t.tm_min -1) * 60 + (60 - t.tm_sec) + 120) / (60 * 60 * 4 + 120) / 244
                self.thread.data[self.sty].OptionList[mat][position][se].T = self.thread.data[self.sty].T[mat]
                # written
                if self.thread.data[self.sty].OptionList[mat][position][se].written == False:
                    self.thread.data[self.sty].OptionList[mat][position][se].written = True
                # cb
                try:
                    if self.thread.data[self.sty].OptionList[mat][position][se].cb['if'] == False and float(message["Quote"]["Bid"]) == float(message["Quote"]["Ask"]):
                        self.thread.data[self.sty].OptionList[mat][position][se].cb['if'] = True
                        self.thread.data[self.sty].OptionList[mat][position][se].cb['start_time'] = time.time()
                except:
                    pass
                if self.thread.data[self.sty].OptionList[mat][position][se].cb['if'] == True and time.time() - self.thread.data[self.sty].OptionList[mat][position][se].cb['start_time'] >= 180:
                    self.thread.data[self.sty].OptionList[mat][position][se].cb['if'] = False


            # future
            elif QuoteID[3] == 'F':

                if 'IF' in QuoteID:
                    fty = FutureType.IF
                elif 'IH' in QuoteID:
                    fty = FutureType.IH
                else:
                    continue
                # mat
                mat = self.thread.data[fty]._2005_to_Mat[QuoteID[-4:]]
                self.thread.data[fty].P[mat] = rt_data['LastPrice']
                self.thread.data[fty].P_yc[mat] = rt_data['YClosedPrice']
                self.thread.data[fty].P_highest[mat] = rt_data['HighPrice']
                self.thread.data[fty].P_lowest[mat] = rt_data['LowPrice']
                self.thread.ft_chg[(fty, mat)] = rt_data['Change'] / (rt_data['LastPrice'] - rt_data['Change'])

            # underlying
            elif QuoteID[3] == 'S':

                if not QuoteID == ['TC.S.SSE.510050', 'TC.S.SSE.510300', 'TC.S.SZSE.159919', 'TC.S.SSE.000300'][[StockType.etf50, StockType.h300, StockType.s300, StockType.gz300].index(self.sty)]:
                   continue 
                self.thread.data[self.sty].ul = rt_data['LastPrice']
                self.thread.data[self.sty].ul_yc = rt_data['YClosedPrice']
                self.thread.data[self.sty].ul_highest = rt_data['HighPrice']
                self.thread.data[self.sty].ul_lowest = rt_data['LowPrice']
                self.thread.ul_chg = rt_data['Change'] / (rt_data['LastPrice'] - rt_data['Change'])


class sig_qobj(QObject):

    signal = pyqtSignal(float)

class music_for_cb(QObject):

    def __init__(self, parent=None):
        super(music_for_cb, self).__init__(parent)
        self.if_play: bool = False

    def play(self, play_time: float):
        musicPath = r"warning_music.mp3"
        pygame.mixer.init()#初始化
        track = pygame.mixer.music.load(musicPath)#加载音乐
        pygame.mixer.music.play()#播放 
        pygame.mixer.music.set_volume(0.5)
        time.sleep(play_time)#表示音频的长度
        pygame.mixer.music.stop()
        self.if_play = False

class MyMixedScreen(QMainWindow):

    def __init__(self,parent=None):
        super(MyMixedScreen, self).__init__(parent)
        self.resize(1900, 900)
        HorizontalHeaderLabels = ['', '50', '沪E', '300', '深E', '', 'VIX', '昨收', '最大值', '最小值', 'skew', '', '50', '沪E', '300', '深E', '350', '当月', '次月', '强弱对比', '300-沪E', '当月', '次月']
        self.ncol = len(HorizontalHeaderLabels)
        self.nrow = 10
        self.setWindowTitle('综合监控屏')
        self.myScreen = GraphicsLayoutWidget()
        self.setCentralWidget(self.myScreen)

        self.myTable = QTableWidget()
        self.myTable.setColumnCount(self.ncol)
        self.myTable.setRowCount(self.nrow)
        self.myTable.setHorizontalHeaderLabels(HorizontalHeaderLabels)
        self.myTable.setStyleSheet('QTableView{background-color: black; color: white;}''QTableCornerButton::section{background-color: black;}')
        self.myTable.horizontalHeader().setStyleSheet('QHeaderView::section{background-color: black; color: white;}''QHeaderView{background-color: black;}')
        self.myTable.horizontalHeader().setDefaultAlignment(Qt.AlignLeft)
        self.myTable.horizontalHeader().setFont(QFont('Times', 11, QFont.Bold))
        self.myTable.verticalHeader().setStyleSheet('QHeaderView::section{background-color: black; color: white;}''QHeaderView{background-color: black;}')
        for i in range(self.ncol):
            if i == 0:
                self.myTable.setColumnWidth(i, 115)
            elif i == 11:
                self.myTable.setColumnWidth(i, 135)
            elif i in [17, 18, 19, 20, 21]:
                self.myTable.setColumnWidth(i, 100)
            else:
                self.myTable.setColumnWidth(i, 70)
        for i in range(self.nrow):
            self.myTable.setRowHeight(i, 5)
        self.dock = QDockWidget()
        self.dock.setWidget(self.myTable)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock)

        # 换月
        t = time.localtime()
        self.change_mat_8 = {}
        for sty in [StockType.etf50, StockType.h300, StockType.gz300, StockType.s300]:
            self.change_mat_8[sty] = False
            if (Mat[sty][Maturity.M1] - calendar.datetime.date(t.tm_year, t.tm_mon, t.tm_mday)).days <= 10:
                self.change_mat_8[sty] = True

        self.mat_M12_vs = [Maturity.M1, Maturity.M2]
        self.mat_M12_vs_gz300 = [Maturity.M1, Maturity.M2]
        self.mat_350_vs = [Maturity.M1, Maturity.M2]
        if self.change_mat_8[StockType.etf50]:
            self.mat_350_vs = [Maturity.M2, None]
            if (Mat[StockType.etf50][Maturity.Q1] - Mat[StockType.etf50][Maturity.M2]).days < 45:
                self.mat_M12_vs = [Maturity.M2, Maturity.Q1]
                self.mat_350_vs = [Maturity.M2, Maturity.Q1]
        if self.change_mat_8[StockType.gz300]:
            self.mat_M12_vs_gz300 = [Maturity.M2, Maturity.M3]

        self.mat_300_vs = {StockType.gz300: [Maturity.M1, Maturity.M2], StockType.h300: [Maturity.M1, Maturity.M2]}
        if self.change_mat_8[StockType.gz300]:
            self.mat_300_vs = {StockType.gz300: [Maturity.M2, None], StockType.h300: [Maturity.M2, None]}
            if Mat[StockType.gz300][Maturity.M3].month == Mat[StockType.h300][Maturity.Q1].month:
                self.mat_300_vs = {StockType.gz300: [Maturity.M2, Maturity.M3], StockType.h300: [Maturity.M2, Maturity.Q1]}

        if Mat[StockType.gz300][Maturity.M1].month != Mat[StockType.etf50][Maturity.M1].month:
            self.mat_300_vs = {StockType.gz300: [Maturity.M1, None], StockType.h300: [Maturity.M2, None]}
            if Mat[StockType.gz300][Maturity.M2].month == Mat[StockType.h300][Maturity.Q1].month:
                self.mat_300_vs = {StockType.gz300: [Maturity.M1, Maturity.M2], StockType.h300: [Maturity.M2, Maturity.Q1]}

        # fig_M1_M2_vix_spread
        self.fig_M1_M2_vix_spread = PlotItem(); self.fig_M1_M2_vix_spread.setTitle('当月 & 次月 vix spread', color = 'w')
        self.myScreen.addItem(self.fig_M1_M2_vix_spread, row = None, col = None, rowspan = 1, colspan = 1)
        self.fig_M1_M2_vix_spread.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_M1_M2_vix_spread.legend.setParentItem(self.fig_M1_M2_vix_spread)
        self.curve_h300_vix_M1_M2 = self.fig_M1_M2_vix_spread.plot(pen = 'y', name = '沪E' + ' | ' + str(Mat[StockType.h300][self.mat_M12_vs[0]].month) + ', ' + str(Mat[StockType.h300][self.mat_M12_vs[1]].month))
        self.curve_etf50_vix_M1_M2 = self.fig_M1_M2_vix_spread.plot(pen = '#82CAFA', name = 'ETF50')

        # fig_300_vix_spread
        self.fig_300_vix_spread = PlotItem(); self.fig_300_vix_spread.setTitle('泛300 vix spread', color = 'w')
        self.myScreen.addItem(self.fig_300_vix_spread, row = None, col = None, rowspan = 1, colspan = 1)
        self.fig_300_vix_spread.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_300_vix_spread.legend.setParentItem(self.fig_300_vix_spread.vb)
        self.curve_300_vix_M1 = self.fig_300_vix_spread.plot(pen = 'y', name = '当月 300 - 沪E' + ' | 300：' + str(Mat[StockType.gz300][self.mat_300_vs[StockType.gz300][0]].month) + ', 沪E：' + str(Mat[StockType.h300][self.mat_300_vs[StockType.h300][0]].month))
        if self.mat_300_vs[StockType.gz300][1] == None:
            self.curve_300_vix_M2 = self.fig_300_vix_spread.plot(pen = '#82CAFA', name = '次月 300 - 沪E' + ' | 300：n.a., 沪E：n.a.')
        else:
            self.curve_300_vix_M2 = self.fig_300_vix_spread.plot(pen = '#82CAFA', name = '次月 300 - 沪E' + ' | 300：' + str(Mat[StockType.gz300][self.mat_300_vs[StockType.gz300][1]].month) + ', 沪E：' + str(Mat[StockType.h300][self.mat_300_vs[StockType.h300][1]].month))

        # fig_350_vix_spread
        self.fig_350_vix_spread = PlotItem(); self.fig_350_vix_spread.setTitle('350 vix spread', color = 'w')
        self.myScreen.addItem(self.fig_350_vix_spread, row = None, col = None, rowspan = 1, colspan = 1)
        self.fig_350_vix_spread.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_350_vix_spread.legend.setParentItem(self.fig_350_vix_spread.vb)
        self.curve_350_vix_M1 = self.fig_350_vix_spread.plot(pen = 'y', name = '当月 沪E - ETF50' + ' | 沪E：' + str(Mat[StockType.h300][self.mat_350_vs[0]].month) + ', ETF50：' + str(Mat[StockType.etf50][self.mat_350_vs[0]].month))
        if self.mat_350_vs[1] == None:
            self.curve_350_vix_M2 = self.fig_350_vix_spread.plot(pen = '#82CAFA', name = '次月 沪E - ETF50' + ' | 沪E：n.a., ETF50：n.a.')
        else:
            self.curve_350_vix_M2 = self.fig_350_vix_spread.plot(pen = '#82CAFA', name = '次月 沪E - ETF50' + ' | 沪E：' + str(Mat[StockType.h300][self.mat_350_vs[1]].month) + ', ETF50：' + str(Mat[StockType.etf50][self.mat_350_vs[1]].month))

        self.myScreen.nextRow()

        # fig_skew
        self.fig_skew = PlotItem(); self.fig_skew.setTitle('skew', color = 'w')
        self.myScreen.addItem(self.fig_skew, row = None, col = None, rowspan = 1, colspan = 1)
        self.fig_skew.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_skew.legend.setParentItem(self.fig_skew.vb)
        self.curve_skew_gz300_interpolate_M1 = self.fig_skew.plot(pen = 'y', symbolBrush = 'y', symbolPen = 'y', name = '300')
        self.curve_skew_h300_interpolate_M1 = self.fig_skew.plot(pen = '82CAFA', symbolBrush = '82CAFA', symbolPen = '82CAFA', name = '沪E')

        # fig_skew_M1
        self.fig_skew_M1 = PlotItem(); self.fig_skew_M1.setTitle('skew 当月', color = 'w')
        self.myScreen.addItem(self.fig_skew_M1, row = None, col = None, rowspan = 1, colspan = 1)
        self.fig_skew_M1.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_skew_M1.legend.setParentItem(self.fig_skew_M1.vb)
        self.curve_skew_etf50_M1 = self.fig_skew_M1.plot(pen = '#EC870E', name = '50')
        self.curve_skew_h300_M1 = self.fig_skew_M1.plot(pen = '#82CAFA', name = '沪E')
        self.curve_skew_gz300_M1 = self.fig_skew_M1.plot(pen = '#B7B7B7', name = '300')

        # fig_skew_M2
        self.fig_skew_M2 = PlotItem(); self.fig_skew_M2.setTitle('skew 次月', color = 'w')
        self.myScreen.addItem(self.fig_skew_M2, row = None, col = None, rowspan = 1, colspan = 1)
        self.fig_skew_M2.legend = LegendItem(offset = (0, 1), labelTextColor = 'w'); self.fig_skew_M2.legend.setParentItem(self.fig_skew_M2.vb)
        self.curve_skew_etf50_M2 = self.fig_skew_M2.plot(pen = '#EC870E', name = '50')
        self.curve_skew_h300_M2 = self.fig_skew_M2.plot(pen = '#82CAFA', name = '沪E')
        self.curve_skew_gz300_M2 = self.fig_skew_M2.plot(pen = '#B7B7B7', name = '300')

        for orientation in ['left', 'bottom']:
            for fig in [self.fig_M1_M2_vix_spread, self.fig_300_vix_spread, self.fig_350_vix_spread, self.fig_skew, self.fig_skew_M1, self.fig_skew_M2]:
                fig.getAxis(orientation).setPen('w')
                fig.getAxis(orientation).setTextPen('w')


        self.timeList = [['09:30', 0], ['10:00', 0], ['10:30', 0], ['11:00', 0], ['11:30', 0], ['13:00', 0], ['13:30', 0], ['14:00', 0], ['14:30', 0]]
        self.timeShow: bool = False
        self.first_plot: bool = True
        self.sig_qobj = sig_qobj()
        self.warning_music = music_for_cb()
        self.warning_music_thread = QThread()
        self.warning_music.moveToThread(self.warning_music_thread)
        self.sig_qobj.signal.connect(self.warning_music.play)
        self.warning_music_thread.start()

        self.tstr: list = []
        self.h300_vix_M1_M2: list = []
        self.etf50_vix_M1_M2: list = []
        self._300_vix_M1: list = []
        self._300_vix_M2: list = []
        self._350_vix_M1: list = []
        self._350_vix_M2: list = []
        self.skew_x: list = []
        self.skew_gz300_interpolate_M1: list = []
        self.skew_h300_interpolate_M1: list = []
        self.skew_etf50_M1: list = []
        self.skew_h300_M1: list = []
        self.skew_gz300_M1: list = []
        self.skew_etf50_M2: list = []
        self.skew_h300_M2: list = []
        self.skew_gz300_M2: list = []


    def plot(self, new_data: tuple):

        self.h300_vix_M1_M2.append(new_data[1][0])
        self.etf50_vix_M1_M2.append(new_data[1][1])
        self._300_vix_M1.append(new_data[2][0])
        self._300_vix_M2.append(new_data[2][1])
        self._350_vix_M1.append(new_data[3][0])
        self._350_vix_M2.append(new_data[3][1])
        self.skew_x = new_data[4][0]
        self.skew_gz300_interpolate_M1 = new_data[4][1]
        self.skew_h300_interpolate_M1 = new_data[4][2]
        self.skew_etf50_M1.append(new_data[5][0])
        self.skew_h300_M1.append(new_data[5][1])
        self.skew_gz300_M1.append(new_data[5][2])
        self.skew_etf50_M2.append(new_data[6][0])
        self.skew_h300_M2.append(new_data[6][1])
        self.skew_gz300_M2.append(new_data[6][2])

        # table
        for i in range(len(new_data[0])):
            for j in range(len(new_data[0][i])):
                if [j, i] in [[3, 17], [4, 17], [3, 21], [4, 21]] or i == 7 and not self.first_plot:
                    continue
                newItem = QTableWidgetItem(str(new_data[0][i][j])); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(j, i, newItem)

        if self.first_plot:
            self.first_plot = False

        color = [QBrush(QColor(255, 165, 0)), QBrush(QColor(135, 206, 255)), QBrush(QColor(245, 245, 245)), QBrush(QColor(144, 238, 144))]
        for i in range(5, 11):
            for j in range(8):
                newItem = self.myTable.item(j, i).clone(); newItem.setForeground(color[j % 4]); self.myTable.setItem(j, i, newItem)

        for i in range(4):
            if 'WARNING' in str(new_data[0][12 + i][0]):
                newItem = QTableWidgetItem(str(new_data[0][12 + i][0])[len('WARNING') : ]); newItem.setBackground(QBrush(QColor(220, 20, 60))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(0, 12 + i, newItem)
            else:
                newItem = QTableWidgetItem(str(new_data[0][12 + i][0])[len('n/a') : ]); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(0, 12 + i, newItem)
            if str(new_data[0][12 + i][1])[-1] == '1':
                newItem = QTableWidgetItem(str(new_data[0][12 + i][1])[:-3]); newItem.setBackground(QBrush(QColor(220, 20, 60))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(1, 12 + i, newItem)
            else:
                newItem = QTableWidgetItem(str(new_data[0][12 + i][1])[:-3]); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(1, 12 + i, newItem)

        num_of_cb = [[0 for _ in range(6)] for i in range(4)] ## [[0] * 6] * 4 你试试！！！
        play_time = 0
        for i in range(4):
            for j in range(3, 9, 1):
                if str(new_data[0][12 + i][j]) == '':
                    continue
                num_of_cb[i][j - 3] = 1
                if '|' in str(new_data[0][12 + i][j]):
                    play_time = max(play_time, float(str(new_data[0][12 + i][j])[new_data[0][12 + i][j].index('C') + 2 : new_data[0][12 + i][j].index('|') - 1]), float(str(new_data[0][12 + i][j])[new_data[0][12 + i][j].index('P') + 2 : ]))
                else:
                    if 'C' in str(new_data[0][12 + i][j]):
                        play_time = max(play_time, float(str(new_data[0][12 + i][j])[new_data[0][12 + i][j].index('C') + 2 : ]))
                    if 'P' in str(new_data[0][12 + i][j]):
                        play_time = max(play_time, float(str(new_data[0][12 + i][j])[new_data[0][12 + i][j].index('P') + 2 : ]))
                newItem = QTableWidgetItem(str(new_data[0][12 + i][j])); newItem.setBackground(QBrush(QColor(220, 20, 60))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(j, 12 + i, newItem)

        if sum([sum(num_of_cb[i]) for i in range(len(num_of_cb))]) > 0 and self.warning_music.if_play == False:
            self.sig_qobj.signal.emit(play_time)
            self.warning_music.if_play = True

        # 350
        if self.mat_350_vs[0] == Maturity.M2:
            if sum(num_of_cb[0][3:6]) > 0 or sum(num_of_cb[1][3:6]) > 0:
                newItem = QTableWidgetItem(str(new_data[0][17][0])); newItem.setBackground(QBrush(QColor(220, 20, 60))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(0, 17, newItem)
        elif self.mat_350_vs == [Maturity.M1, Maturity.M2]:
            if sum(num_of_cb[0][0:3]) > 0 or sum(num_of_cb[1][0:3]) > 0:
                newItem = QTableWidgetItem(str(new_data[0][17][0])); newItem.setBackground(QBrush(QColor(220, 20, 60))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(0, 17, newItem)
            if sum(num_of_cb[0][3:6]) > 0 or sum(num_of_cb[1][3:6]) > 0:
                newItem = QTableWidgetItem(str(new_data[0][18][0])); newItem.setBackground(QBrush(QColor(220, 20, 60))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(0, 18, newItem)

        # 300
        if self.mat_300_vs[StockType.h300][0] == Maturity.M2:
            if sum(num_of_cb[1][3:6]) > 0:
                newItem = QTableWidgetItem(str(new_data[0][21][0])); newItem.setBackground(QBrush(QColor(220, 20, 60))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(0, 21, newItem)
        elif self.mat_300_vs[StockType.h300] == [Maturity.M1, Maturity.M2]:
            if sum(num_of_cb[1][0:3]) > 0:
                newItem = QTableWidgetItem(str(new_data[0][21][0])); newItem.setBackground(QBrush(QColor(220, 20, 60))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(0, 21, newItem)
            if sum(num_of_cb[1][3:6]) > 0:
                newItem = QTableWidgetItem(str(new_data[0][22][0])); newItem.setBackground(QBrush(QColor(220, 20, 60))); newItem.setFont(QFont('Times', 11, QFont.Bold)); self.myTable.setItem(0, 22, newItem)

        # graph
        tstr_ = time.strftime('%H:%M:%S', time.localtime())
        if self.timeShow == False and self.tstr != []:
            self.tstr[-1] = ''
        if [tstr_[:-3], 0] in self.timeList:
            self.timeShow = True
            self.timeList[self.timeList.index([tstr_[:-3], 0])][1] = 1
        else:
            self.timeShow = False
        self.tstr.append(tstr_)

        for curve_data in [(self.curve_h300_vix_M1_M2, self.h300_vix_M1_M2), (self.curve_etf50_vix_M1_M2, self.etf50_vix_M1_M2), (self.curve_300_vix_M1, self._300_vix_M1), (self.curve_300_vix_M2, self._300_vix_M2), (self.curve_350_vix_M1, self._350_vix_M1), (self.curve_350_vix_M2, self._350_vix_M2), (self.curve_skew_gz300_interpolate_M1, self.skew_gz300_interpolate_M1), (self.curve_skew_h300_interpolate_M1, self.skew_h300_interpolate_M1), (self.curve_skew_etf50_M1, self.skew_etf50_M1), (self.curve_skew_h300_M1, self.skew_h300_M1), (self.curve_skew_gz300_M1, self.skew_gz300_M1), (self.curve_skew_etf50_M2, self.skew_etf50_M2), (self.curve_skew_h300_M2, self.skew_h300_M2), (self.curve_skew_gz300_M2, self.skew_gz300_M2)]:
            if curve_data[1][-1] == None:
                continue
            curve_data[0].setData(curve_data[1])

        for fig in [self.fig_M1_M2_vix_spread, self.fig_300_vix_spread, self.fig_350_vix_spread]:
            fig.getAxis('bottom').setTicks([list(zip(range(len(self.tstr)), self.tstr))])
            fig.getAxis('left').setTicks([[(value, '{:.2%}'.format(value)) for value in [0.001 * i for i in range(-200, 200)]], []])
        self.fig_skew_M1.getAxis('bottom').setTicks([list(zip(range(len(self.tstr)), self.tstr))])
        self.fig_skew_M2.getAxis('bottom').setTicks([list(zip(range(len(self.tstr)), self.tstr))])
        self.fig_skew.getAxis('bottom').setTicks([list(zip(range(len(self.skew_x)), self.skew_x))])
        self.fig_skew.getAxis('left').setTicks([[(value, '{:.2%}'.format(value)) for value in [0.25 + 0.001 * i for i in range(-200, 200)]], []])

        space = 4 * 60 * 60 / (freq_for_mixed_screen / 1000)
        for fig in [self.fig_M1_M2_vix_spread, self.fig_300_vix_spread, self.fig_350_vix_spread, self.fig_skew_M1, self.fig_skew_M2]:
            fig.setXRange(max = space, min = 0)

class update_mixed_screen_data(QThread):

    signal = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super(update_mixed_screen_data, self).__init__(parent)
        self.data: dict = {}
        self.myScreen: MyMixedScreen

        self.stylist = [StockType.etf50, StockType.h300, StockType.gz300, StockType.s300]

        for sty in self.stylist:
            self.data[sty] = OptData(sty)
            self.data[sty].subscribe_init(Maturity.M1)
            self.data[sty].subscribe_init(Maturity.M2)
        self.data[StockType.gz300].subscribe_init(Maturity.M3)
        self.data[StockType.h300].subscribe_init(Maturity.Q1)
        self.data[StockType.s300].subscribe_init(Maturity.Q1)
        self.data[StockType.etf50].subscribe_init(Maturity.Q1)

        for fty in [FutureType.IF, FutureType.IH]:
            self.data[fty] = FutureData(fty)

        self.myScreen = MyMixedScreen()
        self.myScreen.show()
        self.signal.connect(self.myScreen.plot)

        self.smile_gz300_x = []
        self.smile_h300_x = []
        self.smile_interpolate_x = [0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03]

        self.ul_chg: dict = {}
        for sty in self.stylist:
            self.ul_chg[sty] = 1
        self.ft_chg: dict = {}
        for fty in [FutureType.IF, FutureType.IH]:
            for mat in [Maturity.M1, Maturity.M2, Maturity.Q1, Maturity.Q2]:
                self.ft_chg[(fty, mat)] = 1

    def run(self):
        vix_max: dict = {}
        vix_min: dict = {}
        vix: dict = {}
        atm_bid: dict = {}
        atm_ask: dict = {}
        pciv: dict = {}
        atm_ask_bid: dict = {}
        atm_vol_ask_bid: dict = {}
        skew: dict = {}
        vix_warning: dict = {}
        cb: dict = {}
        hv20: dict = {}
        S_sts: dict = {}
        k0_str: dict = {}
        plot_period: bool = False
        all_written: bool = False
        csv_read: bool = False
        for sty in self.stylist:
            for mat in [Maturity.M1, Maturity.M2]:
                vix_max[(sty, mat)] = -5
                vix_min[(sty, mat)] = 5


        mat_M12_vs = self.myScreen.mat_M12_vs
        mat_M12_vs_gz300 = self.myScreen.mat_M12_vs_gz300
        mat_350_vs = self.myScreen.mat_350_vs
        mat_300_vs = self.myScreen.mat_300_vs
        change_mat_8 = self.myScreen.change_mat_8


        while True:
            t = time.localtime(); str_year = str(t.tm_year); str_month = str(t.tm_mon); str_day = str(t.tm_mday); hour = t.tm_hour; _min = t.tm_min; sec = t.tm_sec

            if (hour == 9 and _min >= 30) or (hour == 10) or (hour == 11 and _min < 30) or (hour == 11 and _min == 30 and sec == 0) or (hour == 13) or (hour == 14) or (hour == 15 and _min < 2):
                plot_period = True
            else:
                plot_period = False

            time_wait_for_opt_all_written = time.time()
            while all_written == False:
                n_written = 0
                n_all_options = 0
                for sty in self.stylist:
                    for mat in [Maturity.M1, Maturity.M2]:
                        n_all_options += 2 * len(self.data[sty].OptionList[mat])
                        for i in range(len(self.data[sty].OptionList[mat])):
                            if self.data[sty].OptionList[mat][i][0].written == True:
                                n_written += 1
                            if self.data[sty].OptionList[mat][i][1].written == True:
                                n_written += 1
                if n_written == n_all_options or time.time() - time_wait_for_opt_all_written > 30:
                    all_written = True

            # hist_data_init
            if csv_read == False:
                def variable_readin():
                    csvname = '综合屏_hist.csv'
                    f = open(csvname)
                    reader = csv.reader(f, delimiter = ',')
                    hist: dict = {}
                    def str_to_enum(l: list):
                        new_l = []
                        for i in l:
                            if str(i) in ['etf50', 'h300', 'gz300', 's300', 'IF', 'IH', 'M1', 'M2', 'M3', 'Q1', 'Q2', 'Q3']:
                                new_l.append(str_to_type[str(i)])
                            else:
                                new_l.append(str(i))
                        return new_l
                    for z, row in enumerate(reader):
                        for i in range(len(row)):
                            if row[i] == 'reading_end':
                                end_posi = i
                        if row[0][:3] == 'sgl':
                            hist[tuple(str_to_enum(row[:end_posi]))] = float(row[end_posi + 1])
                        elif row[0][:3] == 'seq':
                            hist[tuple(str_to_enum(row[:end_posi]))] = [float(x) for x in row[end_posi + 1:] if x != '']
                    f.close()
                    return hist

                exist = os.path.isfile('./综合屏_hist.csv')
                if exist == False:
                    csvname = '综合屏_hist.csv'
                    f = open(csvname,'w',newline='')
                    f_w = csv.writer(f)
                    f_w.writerow(['seq_ul_close', 'etf50', 'reading_end'] + [1] * 50)
                    f_w.writerow(['seq_ul_close', 'h300', 'reading_end'] + [1] * 50)
                    f_w.writerow(['seq_close', 'IF', 'M1', 'reading_end'] + [1] * 50)
                    for mat in ['M1', 'M2']:
                        for sty in ['etf50', 'h300', 'gz300', 's300']:
                            f_w.writerow(['sgl_vix', sty, mat, 'l1', 'reading_end', 0.25])
                            f_w.writerow(['sgl_vix', sty, mat, 'l2', 'reading_end', 0.25])
                            f_w.writerow(['sgl_skew', sty, mat, 'l1', 'reading_end', 100])
                    f.flush()
                    f.close()
                hist = variable_readin()
                csv_read = True


            if all_written and csv_read and plot_period:

                vix[(StockType.gz300, Maturity.M3)] = self.data[StockType.gz300].vix(Maturity.M3)
                vix[(StockType.h300, Maturity.Q1)] = self.data[StockType.h300].vix(Maturity.Q1)
                vix[(StockType.s300, Maturity.Q1)] = self.data[StockType.s300].vix(Maturity.Q1)
                vix[(StockType.etf50, Maturity.Q1)] = self.data[StockType.etf50].vix(Maturity.Q1)

                for sty in self.stylist:
                    for mat in [Maturity.M1, Maturity.M2]:
                        vix[(sty, mat)] = self.data[sty].vix(mat)
                        S_sts[(sty, mat)] = (self.data[sty].S[mat] - self.data[sty].ul) / self.data[sty].ul / self.data[sty].T[mat]
                        if not ((hour == 9 and _min == 30 and sec < 30) or (hour == 11 and _min == 29 and sec > 50) or (hour == 11 and _min == 30) or (hour == 13 and _min == 0 and sec < 10)):
                            vix_max[(sty, mat)] = max(vix_max[(sty, mat)], vix[(sty, mat)]) * (vix[(sty, mat)] < 1.4 * hist[('sgl_vix', sty, mat, 'l1')]) + vix_max[(sty, mat)] * (vix[(sty, mat)] >= 1.4 * hist[('sgl_vix', sty, mat, 'l1')])
                            vix_min[(sty, mat)] = min(vix_min[(sty, mat)], vix[(sty, mat)]) * (vix[(sty, mat)] > 0.6 * hist[('sgl_vix', sty, mat, 'l1')]) + vix_min[(sty, mat)] * (vix[(sty, mat)] <= 0.6 * hist[('sgl_vix', sty, mat, 'l1')])

                        for shift in ['out1', 'at', 'in1']:
                            content = ''
                            if shift[0:2] == 'ou':
                                n_shift = 1
                            elif shift[0:2] == 'at':
                                n_shift = 0
                            elif shift[0:2] == 'in':
                                n_shift = -1
                            posi_c = (self.data[sty].posi[mat] + n_shift) * (len(self.data[sty].OptionList[mat]) - 1 >= self.data[sty].posi[mat] + n_shift >= 0) + self.data[sty].posi[mat] * (1 - (len(self.data[sty].OptionList[mat]) - 1 >= self.data[sty].posi[mat] + n_shift >= 0))
                            posi_p = (self.data[sty].posi[mat] - n_shift) * (len(self.data[sty].OptionList[mat]) - 1 >= self.data[sty].posi[mat] - n_shift >= 0) + self.data[sty].posi[mat] * (1 - (len(self.data[sty].OptionList[mat]) - 1 >= self.data[sty].posi[mat] - n_shift >= 0))
                            if self.data[sty].OptionList[mat][posi_c][0].cb['if']:
                                content += str(self.data[sty].OptionList[mat][posi_c][0].K) + 'C ' + '{:.0f}'.format(177 - time.time() + self.data[sty].OptionList[mat][posi_c][0].cb['start_time'])
                            if self.data[sty].OptionList[mat][posi_p][1].cb['if']:
                                if len(content) == 0:
                                    plus = ''
                                else:
                                    plus = ' | '
                                content += plus + str(self.data[sty].OptionList[mat][posi_p][1].K) + 'P ' + '{:.0f}'.format(177 - time.time() + self.data[sty].OptionList[mat][posi_p][1].cb['start_time'])
                            if (hour == 14 and _min > 56) or hour == 15:
                                cb[(sty, mat, shift)] = ''
                            else:
                                cb[(sty, mat, shift)] = content
                            if sty == StockType.gz300:
                                cb[(sty, mat, shift)] = ''

                        for oty in [OptionType.C, OptionType.P]:
                            if oty == OptionType.C:
                                atm_bid[(sty, mat, oty)] = self.data[sty].OptionList[mat][self.data[sty].posi[mat]][0].iv_p(self.data[sty].OptionList[mat][self.data[sty].posi[mat]][0].bid)
                                atm_ask[(sty, mat, oty)] = self.data[sty].OptionList[mat][self.data[sty].posi[mat]][0].iv_p(self.data[sty].OptionList[mat][self.data[sty].posi[mat]][0].ask)
                            elif oty == OptionType.P:
                                atm_bid[(sty, mat, oty)] = self.data[sty].OptionList[mat][self.data[sty].posi[mat]][1].iv_p(self.data[sty].OptionList[mat][self.data[sty].posi[mat]][1].bid)
                                atm_ask[(sty, mat, oty)] = self.data[sty].OptionList[mat][self.data[sty].posi[mat]][1].iv_p(self.data[sty].OptionList[mat][self.data[sty].posi[mat]][1].ask)
                            atm_ask_bid[(sty, mat, oty)]  = atm_ask[(sty, mat, oty)] - atm_bid[(sty, mat, oty)]
                        atm_vol_ask_bid[(sty, mat)]  = (atm_ask_bid[(sty, mat, OptionType.C)] + atm_ask_bid[(sty, mat, OptionType.P)]) / 2
                        pciv[(sty, mat)] = self.data[sty].OptionList[mat][np.argmin(abs(np.array([self.data[sty].OptionList[mat][i][1].delta() for i in range(len(self.data[sty].OptionList[mat]))]) - (-0.25)))][1].iv() - self.data[sty].OptionList[mat][np.argmin(abs(np.array([self.data[sty].OptionList[mat][i][0].delta() for i in range(len(self.data[sty].OptionList[mat]))]) - 0.25))][0].iv()
                        skew[(sty, mat)] = 100 - 10 * self.data[sty].skew_same_T(mat)

                    mat = Maturity.M1
                    if change_mat_8[sty]:
                        mat = Maturity.M2
                    if vix[(sty, mat)] > hist[('sgl_vix', sty, mat, 'l1')] + 0.01 and hist[('sgl_vix', sty, mat, 'l1')] > hist[('sgl_vix', sty, mat, 'l2')] + 0.01:
                        vix_warning[sty] = 'WARNING' + '{:.2%}'.format(hist[('sgl_vix', sty, mat, 'l1')] + 0.01)
                    else:
                        vix_warning[sty] = 'n/a' + '{:.2%}'.format(hist[('sgl_vix', sty, mat, 'l1')] + 0.01)

                    if (len(self.data[sty].OptionList[Maturity.M1]) - 1 > self.data[sty].posi[Maturity.M1] > 0) == 1:
                        content = str((self.data[sty].S[Maturity.M1] >= self.data[sty].k0[Maturity.M1]) * (self.data[sty].S[Maturity.M1] - self.data[sty].k0[Maturity.M1] > (self.data[sty].OptionList[Maturity.M1][self.data[sty].posi[Maturity.M1] + 1][0].K - self.data[sty].k0[Maturity.M1]) / 2 * 0.9) + (self.data[sty].S[Maturity.M1] < self.data[sty].k0[Maturity.M1]) * (self.data[sty].k0[Maturity.M1] - self.data[sty].S[Maturity.M1] > (self.data[sty].k0[Maturity.M1] - self.data[sty].OptionList[Maturity.M1][self.data[sty].posi[Maturity.M1] - 1][0].K) / 2 * 0.9) + 0)
                    else:
                        content = str(0 + 0)
                    if sty in [StockType.etf50, StockType.h300, StockType.s300]:
                        k0_str[(sty, Maturity.M1)] = '{:.2f}'.format(self.data[sty].k0[Maturity.M1]) + ', ' + content
                    elif sty == StockType.gz300:
                        k0_str[(sty, Maturity.M1)] = '{:.0f}'.format(self.data[sty].k0[Maturity.M1]) + ', ' + content

                for sty in [StockType.etf50, StockType.h300, StockType.gz300]:
                    if sty in [StockType.etf50, StockType.h300]:
                        ir = np.array(hist[('seq_ul_close', sty)][-18:] + [self.data[sty].ul_yc] + [self.data[sty].ul]) / np.array(hist[('seq_ul_close', sty)][-19:] + [self.data[sty].ul_yc]) - 1
                    elif sty == StockType.gz300:
                        ir = np.array(hist[('seq_close', FutureType.IF, Maturity.M1)][-18:] + [self.data[FutureType.IF].P_yc[Maturity.M1]] + [self.data[FutureType.IF].P[Maturity.M1]]) / np.array(hist[('seq_close', FutureType.IF, Maturity.M1)][-19:] + [self.data[FutureType.IF].P_yc[Maturity.M1]]) - 1
                    array = [0 for _ in range(len(ir) + 1)]
                    array[0] = np.std(ir, ddof=1) ** 2
                    lb = 0.94
                    for i in range(1, len(array)):
                        array[i] = lb * array[i - 1] + (1 - lb) * ir[i - 1] ** 2
                    hv20[sty] = np.sqrt(array[-1] * 244)

                etf50_vix_M1_M2 = vix[(StockType.etf50, mat_M12_vs[0])] - vix[(StockType.etf50, mat_M12_vs[1])]
                h300_vix_M1_M2 = vix[(StockType.h300, mat_M12_vs[0])] - vix[(StockType.h300, mat_M12_vs[1])]
                gz300_vix_M1_M2 = vix[(StockType.gz300, mat_M12_vs_gz300[0])] - vix[(StockType.gz300, mat_M12_vs_gz300[1])]
                s300_vix_M1_M2 = vix[(StockType.s300, mat_M12_vs[0])] - vix[(StockType.s300, mat_M12_vs[1])]
                _350_vix_M1 = vix[(StockType.h300, mat_350_vs[0])] - vix[(StockType.etf50, mat_350_vs[0])]
                _350_atm_vol_M1 = self.data[StockType.h300].OptionList[mat_350_vs[0]][self.data[StockType.h300].posi[mat_350_vs[0]]][0].iv() - self.data[StockType.etf50].OptionList[mat_350_vs[0]][self.data[StockType.etf50].posi[mat_350_vs[0]]][0].iv()
                _350_vix_M2 = None
                _350_atm_vol_M2 = None
                if not mat_350_vs[1] == None: 
                    _350_vix_M2 = vix[(StockType.h300, mat_350_vs[1])] - vix[(StockType.etf50, mat_350_vs[1])]
                    _350_atm_vol_M2 = self.data[StockType.h300].OptionList[mat_350_vs[1]][self.data[StockType.h300].posi[mat_350_vs[1]]][0].iv() - self.data[StockType.etf50].OptionList[mat_350_vs[1]][self.data[StockType.etf50].posi[mat_350_vs[1]]][0].iv()

                _300_vix_M1 = vix[(StockType.gz300, mat_300_vs[StockType.gz300][0])] - vix[(StockType.h300, mat_300_vs[StockType.h300][0])]
                _300_vix_M2 = None
                _300_atm_vol_M1 = self.data[StockType.gz300].OptionList[mat_300_vs[StockType.gz300][0]][self.data[StockType.gz300].posi[mat_300_vs[StockType.gz300][0]]][0].iv() - self.data[StockType.h300].OptionList[mat_300_vs[StockType.h300][0]][self.data[StockType.h300].posi[mat_300_vs[StockType.h300][0]]][0].iv()
                _300_atm_vol_M2 = None
                if not mat_300_vs[StockType.gz300][1] == None:
                    _300_vix_M2 = vix[(StockType.gz300, mat_300_vs[StockType.gz300][1])] - vix[(StockType.h300, mat_300_vs[StockType.h300][1])]
                    _300_atm_vol_M2 = self.data[StockType.gz300].OptionList[mat_300_vs[StockType.gz300][1]][self.data[StockType.gz300].posi[mat_300_vs[StockType.gz300][1]]][0].iv() - self.data[StockType.h300].OptionList[mat_300_vs[StockType.h300][1]][self.data[StockType.h300].posi[mat_300_vs[StockType.h300][1]]][0].iv()

                smile_gz300_y = []; smile_h300_y = []; smile_gz300_interpolate_y = []; smile_h300_interpolate_y = []
                mat_gz300 = mat_300_vs[StockType.gz300][0]
                mat_h300 = mat_300_vs[StockType.h300][0]
                self.smile_gz300_x = [self.data[StockType.gz300].OptionList[mat_gz300][i][0].K for i in range(len(self.data[StockType.gz300].OptionList[mat_gz300]))]
                self.smile_h300_x = [self.data[StockType.h300].OptionList[mat_h300][i][0].K for i in range(len(self.data[StockType.h300].OptionList[mat_h300]))]
                for i in range(len(self.smile_gz300_x)):
                    if self.data[StockType.gz300].OptionList[mat_gz300][i][0].K < self.data[StockType.gz300].S[mat_gz300]:
                        smile_gz300_y.append(self.data[StockType.gz300].OptionList[mat_gz300][i][1].iv())
                    elif self.data[StockType.gz300].OptionList[mat_gz300][i][0].K == self.data[StockType.gz300].S[mat_gz300]:
                        smile_gz300_y.append(self.data[StockType.gz300].OptionList[mat_gz300][i][0].iv_s(self.data[StockType.gz300].OptionList[mat_gz300][i][0].midbidaskspread() - self.data[StockType.gz300].OptionList[mat_gz300][i][1].midbidaskspread() + self.data[StockType.gz300].OptionList[mat_gz300][i][0].K))
                    else:
                        smile_gz300_y.append(self.data[StockType.gz300].OptionList[mat_gz300][i][0].iv())
                for i in range(len(self.smile_h300_x)):
                    if self.data[StockType.h300].OptionList[mat_h300][i][0].K < self.data[StockType.h300].S[mat_h300]:
                        smile_h300_y.append(self.data[StockType.h300].OptionList[mat_h300][i][1].iv())
                    elif self.data[StockType.h300].OptionList[mat_h300][i][0].K == self.data[StockType.h300].S[mat_h300]:
                        smile_h300_y.append(self.data[StockType.h300].OptionList[mat_h300][i][0].iv_s(self.data[StockType.h300].OptionList[mat_h300][i][0].midbidaskspread() - self.data[StockType.h300].OptionList[mat_h300][i][1].midbidaskspread() + self.data[StockType.h300].OptionList[mat_h300][i][0].K))
                    else:
                        smile_h300_y.append(self.data[StockType.h300].OptionList[mat_h300][i][0].iv())
                smile_gz300_cs = CubicSpline(self.smile_gz300_x, smile_gz300_y)
                smile_h300_cs = CubicSpline(self.smile_h300_x, smile_h300_y)
                s_gz300 = self.data[StockType.gz300].S[mat_gz300]
                s_h300 = self.data[StockType.h300].S[mat_h300]
                for i in self.smile_interpolate_x:
                    smile_gz300_interpolate_y.append(smile_gz300_cs(i * s_gz300))
                    smile_h300_interpolate_y.append(smile_h300_cs(i * s_h300))

                def rd_pc(x: list):
                    if len(x) != 1:
                        return ['{:.2%}'.format(i) for i in x]
                    else:
                        return '{:.2%}'.format(x[0])

                table = []
                table.append(['标的价格',  '涨跌幅',  '合成期货', '平值行权价', '升贴水', '次月合成期货', '次月平值行权价', '次月升贴水', 'IH (M1)', 'IH (M2)'])
                table.append(['{:.3f}'.format(self.data[StockType.etf50].ul), rd_pc([self.ul_chg[StockType.etf50]]), '{:.3f}'.format(self.data[StockType.etf50].S[Maturity.M1]), '{:.2f}'.format(self.data[StockType.etf50].k0[Maturity.M1]), rd_pc([S_sts[(StockType.etf50, Maturity.M1)]]), '{:.3f}'.format(self.data[StockType.etf50].S[Maturity.M2]), '{:.2f}'.format(self.data[StockType.etf50].k0[Maturity.M2]), rd_pc([S_sts[(StockType.etf50, Maturity.M2)]])] + ['{:.1f}'.format(self.data[FutureType.IH].P[mat]) for mat in [Maturity.M1, Maturity.M2]])
                table.append(['{:.3f}'.format(self.data[StockType.h300].ul), rd_pc([self.ul_chg[StockType.h300]]), '{:.3f}'.format(self.data[StockType.h300].S[Maturity.M1]), '{:.2f}'.format(self.data[StockType.h300].k0[Maturity.M1]), rd_pc([S_sts[(StockType.h300, Maturity.M1)]]), '{:.3f}'.format(self.data[StockType.h300].S[Maturity.M2]), '{:.2f}'.format(self.data[StockType.h300].k0[Maturity.M2]), rd_pc([S_sts[(StockType.h300, Maturity.M2)]])] + rd_pc([self.ft_chg[(FutureType.IH, mat)] for mat in [Maturity.M1, Maturity.M2]]))
                table.append(['{:.1f}'.format(self.data[StockType.gz300].ul), rd_pc([self.ul_chg[StockType.gz300]]), '{:.1f}'.format(self.data[StockType.gz300].S[Maturity.M1]), '{:.0f}'.format(self.data[StockType.gz300].k0[Maturity.M1]), rd_pc([S_sts[(StockType.gz300, Maturity.M1)]]), '{:.1f}'.format(self.data[StockType.gz300].S[Maturity.M2]), '{:.0f}'.format(self.data[StockType.gz300].k0[Maturity.M2]), rd_pc([S_sts[(StockType.gz300, Maturity.M2)]])] + ['IF (M1)', 'IF (M2)'])
                table.append(['{:.3f}'.format(self.data[StockType.s300].ul), rd_pc([self.ul_chg[StockType.s300]]), '{:.3f}'.format(self.data[StockType.s300].S[Maturity.M1]), '{:.2f}'.format(self.data[StockType.s300].k0[Maturity.M1]), rd_pc([S_sts[(StockType.s300, Maturity.M1)]]), '{:.3f}'.format(self.data[StockType.s300].S[Maturity.M2]), '{:.2f}'.format(self.data[StockType.s300].k0[Maturity.M2]), rd_pc([S_sts[(StockType.s300, Maturity.M2)]])] + ['{:.1f}'.format(self.data[FutureType.IF].P[mat]) for mat in [Maturity.M1, Maturity.M2]])
                table.append(['50 (M1)', '沪E (M1)', '300 (M1)', '深E (M1)', '50 (M2)', '沪E (M2)', '300 (M2)', '深E (M2)'] + rd_pc([self.ft_chg[(FutureType.IF, mat)] for mat in [Maturity.M1, Maturity.M2]]))
                table.append(rd_pc([vix[(sty, Maturity.M1)] for sty in self.stylist] + [vix[(sty, Maturity.M2)] for sty in self.stylist]))
                table.append(rd_pc([hist[('sgl_vix', sty, Maturity.M1, 'l1')] for sty in self.stylist] + [hist[('sgl_vix', sty, Maturity.M2, 'l1')] for sty in self.stylist]))
                table.append(rd_pc([vix_max[(sty, Maturity.M1)] for sty in self.stylist] + [vix_max[(sty, Maturity.M2)] for sty in self.stylist]))
                table.append(rd_pc([vix_min[(sty, Maturity.M1)] for sty in self.stylist] + [vix_min[(sty, Maturity.M2)] for sty in self.stylist]))
                table.append(['{:.1f}'.format(skew[(sty, Maturity.M1)]) for sty in self.stylist] + ['{:.1f}'.format(skew[(sty, Maturity.M2)]) for sty in self.stylist])
                table.append(['日波预警', '行权价切换', '熔断合约与倒计时', '虚值一 (M1)', '平值 (M1)', '实值一 (M1)', '虚值一 (M2)', '平值 (M2)', '实值一 (M2)'])
                table.append([vix_warning[StockType.etf50], k0_str[(StockType.etf50, Maturity.M1)]] + [''] + [cb[(StockType.etf50, Maturity.M1, 'out1')], cb[(StockType.etf50, Maturity.M1, 'at')], cb[(StockType.etf50, Maturity.M1, 'in1')], cb[(StockType.etf50, Maturity.M2, 'out1')], cb[(StockType.etf50, Maturity.M2, 'at')], cb[(StockType.etf50, Maturity.M2, 'in1')]])
                table.append([vix_warning[StockType.h300], k0_str[(StockType.h300, Maturity.M1)]] + [''] + [cb[(StockType.h300, Maturity.M1, 'out1')], cb[(StockType.h300, Maturity.M1, 'at')], cb[(StockType.h300, Maturity.M1, 'in1')], cb[(StockType.h300, Maturity.M2, 'out1')], cb[(StockType.h300, Maturity.M2, 'at')], cb[(StockType.h300, Maturity.M2, 'in1')]])
                table.append([vix_warning[StockType.gz300], k0_str[(StockType.gz300, Maturity.M1)]] + [''] + [cb[(StockType.gz300, Maturity.M1, 'out1')], cb[(StockType.gz300, Maturity.M1, 'at')], cb[(StockType.gz300, Maturity.M1, 'in1')], cb[(StockType.gz300, Maturity.M2, 'out1')], cb[(StockType.gz300, Maturity.M2, 'at')], cb[(StockType.gz300, Maturity.M2, 'in1')]])
                table.append([vix_warning[StockType.s300], k0_str[(StockType.s300, Maturity.M1)]] + [''] + [cb[(StockType.s300, Maturity.M1, 'out1')], cb[(StockType.s300, Maturity.M1, 'at')], cb[(StockType.s300, Maturity.M1, 'in1')], cb[(StockType.s300, Maturity.M2, 'out1')], cb[(StockType.s300, Maturity.M2, 'at')], cb[(StockType.s300, Maturity.M2, 'in1')]])
                table.append(['VIX', 'ATM Vol', 'HV', 'Long', 'Short', '', '50', '沪E', '300', '深E'])
                table.append(rd_pc([_350_vix_M1, _350_atm_vol_M1, hv20[StockType.h300] - hv20[StockType.etf50]]) + ['', '', '近次月VIX差'] + rd_pc([etf50_vix_M1_M2, h300_vix_M1_M2, gz300_vix_M1_M2, s300_vix_M1_M2]))
                if _350_vix_M2 == None:
                    table.append(['n.a.', 'n.a.'] + ['', '', '', '当月买卖差'] + rd_pc([atm_vol_ask_bid[(sty, Maturity.M1)] for sty in self.stylist]))
                else:
                    table.append(rd_pc([_350_vix_M2, _350_atm_vol_M2]) + ['', '', '', '当月买卖差'] + rd_pc([atm_vol_ask_bid[(sty, Maturity.M1)] for sty in self.stylist]))
                table.append(['', '', '', '', '', '次月买卖差'] + rd_pc([atm_vol_ask_bid[(sty, Maturity.M2)] for sty in self.stylist]))
                table.append(['VIX', 'ATM Vol', 'HV', 'Long', 'Short', '当月PCIV差'] + rd_pc([pciv[(sty, Maturity.M1)] for sty in self.stylist]))
                table.append(rd_pc([_300_vix_M1, _300_atm_vol_M1, hv20[StockType.gz300] - hv20[StockType.h300]]) + ['', '', '次月PCIV差'] + rd_pc([pciv[(sty, Maturity.M2)] for sty in self.stylist]))
                if _300_vix_M2 == None:
                    table.append(['n.a.', 'n.a.'])
                else:
                    table.append(rd_pc([_300_vix_M2, _300_atm_vol_M2]))

                new_data = (table, [h300_vix_M1_M2, etf50_vix_M1_M2], [_300_vix_M1, _300_vix_M2], [_350_vix_M1, _350_vix_M2], [['{:.0f}'.format(i * s_gz300) + '|' + '{:.2f}'.format(i * s_h300) for i in self.smile_interpolate_x], smile_gz300_interpolate_y, smile_h300_interpolate_y], [skew[(sty, Maturity.M1)] for sty in self.stylist[:-1]], [skew[(sty, Maturity.M2)] for sty in self.stylist[:-1]])
                self.signal.emit(new_data)
                self.msleep(freq_for_mixed_screen)

                if hour == 15 and _min == 1 and sec > 30:
                    csvname = '综合屏_hist.csv'
                    f = open(csvname,'w',newline='')
                    f_w=csv.writer(f)
                    f_w.writerow(['seq_ul_close', 'etf50', 'reading_end'] + hist[('seq_ul_close', StockType.etf50)][1:] + [self.data[StockType.etf50].ul_yc])
                    f_w.writerow(['seq_ul_close', 'h300', 'reading_end'] + hist[('seq_ul_close', StockType.h300)][1:] + [self.data[StockType.h300].ul_yc])
                    f_w.writerow(['seq_close', 'IF', 'M1', 'reading_end'] + hist[('seq_close', FutureType.IF, Maturity.M1)][1:] + [self.data[FutureType.IF].P_yc[Maturity.M1]])
                    for mat in ['M1', 'M2']:
                        for sty in ['etf50', 'h300', 'gz300', 's300']:
                            if (Mat[str_to_type[sty]][Maturity.M1] - calendar.datetime.date(t.tm_year, t.tm_mon, t.tm_mday)).days == 0:
                                if sty != 'gz300':
                                    if mat == 'M1':
                                        f_w.writerow(['sgl_vix', sty, mat, 'l1', 'reading_end', vix[(str_to_type[sty], str_to_type['M2'])]])
                                        f_w.writerow(['sgl_vix', sty, mat, 'l2', 'reading_end', hist[('sgl_vix', str_to_type[sty], str_to_type['M2'], 'l1')]])
                                        f_w.writerow(['sgl_skew', sty, mat, 'l1', 'reading_end', skew[(str_to_type[sty], str_to_type['M2'])]])
                                    elif mat == 'M2':
                                        if Mat[StockType.h300][Maturity.M2].month % 12 + 1 == Mat[StockType.h300][Maturity.Q1].month:
                                            f_w.writerow(['sgl_vix', sty, mat, 'l1', 'reading_end', vix[(str_to_type[sty], str_to_type['Q1'])]])
                                            f_w.writerow(['sgl_vix', sty, mat, 'l2', 'reading_end', 0.25])
                                            f_w.writerow(['sgl_skew', sty, mat, 'l1', 'reading_end', 100 - 10 * self.data[str_to_type[sty]].skew_same_T(Maturity.Q1)])
                                        else:
                                            f_w.writerow(['sgl_vix', sty, mat, 'l1', 'reading_end', 0.25])
                                            f_w.writerow(['sgl_vix', sty, mat, 'l2', 'reading_end', 0.25])
                                            f_w.writerow(['sgl_skew', sty, mat, 'l1', 'reading_end', 100])
                                else:
                                    if mat == 'M1':
                                        f_w.writerow(['sgl_vix', sty, mat, 'l1', 'reading_end', vix[(str_to_type[sty], str_to_type['M2'])]])
                                        f_w.writerow(['sgl_vix', sty, mat, 'l2', 'reading_end', hist[('sgl_vix', str_to_type[sty], str_to_type['M2'], 'l1')]])
                                        f_w.writerow(['sgl_skew', sty, mat, 'l1', 'reading_end', skew[(str_to_type[sty], str_to_type['M2'])]])
                                    elif mat == 'M2':
                                        f_w.writerow(['sgl_vix', sty, mat, 'l1', 'reading_end', self.data[StockType.gz300].vix(Maturity.M3)])
                                        f_w.writerow(['sgl_vix', sty, mat, 'l2', 'reading_end', self.data[StockType.gz300].vix(Maturity.M3)]) ##
                                        f_w.writerow(['sgl_skew', sty, mat, 'l1', 'reading_end', 100 - 10 * self.data[StockType.gz300].skew_same_T(Maturity.M3)])
                            else:
                                f_w.writerow(['sgl_vix', sty, mat, 'l1', 'reading_end', vix[(str_to_type[sty], str_to_type[mat])]])
                                f_w.writerow(['sgl_vix', sty, mat, 'l2', 'reading_end', hist[('sgl_vix', str_to_type[sty], str_to_type[mat], 'l1')]])
                                f_w.writerow(['sgl_skew', sty, mat, 'l1', 'reading_end', skew[(str_to_type[sty], str_to_type[mat])]])
                    f.flush()
                    f.close()

class CFtdcMdSpi_Mixed_Screen(QThread):

    def __init__(self, parent=None):
        super(CFtdcMdSpi_Mixed_Screen, self).__init__(parent)
        self.obj = g_QuoteZMQ
        self.q_data = q_data
        self.thread = update_mixed_screen_data()
        self.thread.start()
        self.mat_2005_M1 = {}
        self.mat_2005_M2 = {}
        for sty in [StockType.gz300, StockType.etf50, StockType.h300, StockType.s300]:
            self.mat_2005_M1[sty] = self.thread.data[sty].Mat_to_2005[Maturity.M1]
            self.mat_2005_M2[sty] = self.thread.data[sty].Mat_to_2005[Maturity.M2]

        self.socket_sub = self.obj.context.socket(zmq.SUB)
        self.socket_sub.connect("tcp://127.0.0.1:%s" % self.q_data["SubPort"])
        self.socket_sub.setsockopt_string(zmq.SUBSCRIBE,"")

    def run(self):
        while(True):
            message = (self.socket_sub.recv()[:-1]).decode("utf-8")
            index =  re.search(":",message).span()[1]
            message = message[index:]
            message = json.loads(message)

            rt_data = {}

            if message["DataType"] == "REALTIME":
                QuoteID = message["Quote"]["Symbol"]

                rt_data['LastPrice'] = 0.001
                if not message["Quote"]["TradingPrice"] == "":
                    rt_data['LastPrice'] = float(message["Quote"]["TradingPrice"])

                for key in ["Bid", "Ask", "YClosedPrice", "HighPrice", "LowPrice", "Change"]:
                    rt_data[key] = 0
                    if not message["Quote"][key] == "":
                        rt_data[key] = float(message["Quote"][key])

            elif message["DataType"] == "PING":
                self.obj.QuotePong(self.q_data["SessionKey"])
                continue
            
            else:
                continue


            if QuoteID[3] == 'O':

                #TC.O.SSE.510050.202007.C.2.8
                #TC.O.SSE.510300.202007.C.4
                #TC.O.SZSE.159919.202007.C.4
                #TC.O.CFFEX.IO.202007.P.4000

                if 'TC.O.SSE.510050' in QuoteID:
                    sty = StockType.etf50
                    mat = self.thread.data[sty]._2005_to_Mat[QuoteID[18 : 22]]
                elif 'TC.O.SSE.510300' in QuoteID:
                    sty = StockType.h300
                    mat = self.thread.data[sty]._2005_to_Mat[QuoteID[18 : 22]]
                elif 'TC.O.SZSE.159919' in QuoteID:
                    sty = StockType.s300
                    mat = self.thread.data[sty]._2005_to_Mat[QuoteID[19 : 23]]
                elif 'TC.O.CFFEX.IO' in QuoteID:
                    sty = StockType.gz300
                    mat = self.thread.data[sty]._2005_to_Mat[QuoteID[16 : 20]]
                else:
                    continue

                if mat not in [Maturity.M1, Maturity.M2, Maturity.M3, Maturity.Q1] or (sty == StockType.gz300 and mat == Maturity.Q1):
                    continue

                position = self.thread.data[sty].k_list[mat].index(float(QuoteID[last_C_P(QuoteID) : ]))
                if '.C.' in QuoteID:
                    se = 0
                elif '.P.' in QuoteID:
                    se = 1


                # update OptionList
                self.thread.data[sty].OptionList[mat][position][se].P = rt_data['LastPrice']
                self.thread.data[sty].OptionList[mat][position][se].bid = rt_data['Bid']
                self.thread.data[sty].OptionList[mat][position][se].ask = rt_data['Ask']
                # update S, k0, posi
                self.thread.data[sty].S_k0_posi(mat)
                self.thread.data[sty].OptionList[mat][position][se].S = self.thread.data[sty].S[mat]
                # update time
                t = time.localtime()
                self.thread.data[sty].T[mat] = self.thread.data[sty].initT[mat] + ((15 - t.tm_hour - 1 - 1.5 * (t.tm_hour < 12)) * 60 * 60 + (60 - t.tm_min -1) * 60 + (60 - t.tm_sec) + 120) / (60 * 60 * 4 + 120) / 244
                self.thread.data[sty].OptionList[mat][position][se].T = self.thread.data[sty].T[mat]
                # written
                if self.thread.data[sty].OptionList[mat][position][se].written == False:
                    self.thread.data[sty].OptionList[mat][position][se].written = True
                # cb
                try:
                    if self.thread.data[sty].OptionList[mat][position][se].cb['if'] == False and float(message["Quote"]["Bid"]) == float(message["Quote"]["Ask"]):
                        self.thread.data[sty].OptionList[mat][position][se].cb['if'] = True
                        self.thread.data[sty].OptionList[mat][position][se].cb['start_time'] = time.time()
                except:
                    pass
                if self.thread.data[sty].OptionList[mat][position][se].cb['if'] == True and time.time() - self.thread.data[sty].OptionList[mat][position][se].cb['start_time'] >= 180:
                    self.thread.data[sty].OptionList[mat][position][se].cb['if'] = False


            # future
            elif QuoteID[3] == 'F':

                if 'IF' in QuoteID:
                    fty = FutureType.IF
                elif 'IH' in QuoteID:
                    fty = FutureType.IH
                else:
                    continue
                # mat
                mat = self.thread.data[fty]._2005_to_Mat[QuoteID[-4:]]
                self.thread.data[fty].P[mat] = rt_data['LastPrice']
                self.thread.data[fty].P_yc[mat] = rt_data['YClosedPrice']
                self.thread.data[fty].P_highest[mat] = rt_data['HighPrice']
                self.thread.data[fty].P_lowest[mat] = rt_data['LowPrice']
                self.thread.ft_chg[(fty, mat)] = rt_data['Change'] / (rt_data['LastPrice'] - rt_data['Change'])


            # underlying
            elif QuoteID[3] == 'S':

                sty = [StockType.etf50, StockType.h300, StockType.s300, StockType.gz300][['TC.S.SSE.510050', 'TC.S.SSE.510300', 'TC.S.SZSE.159919', 'TC.S.SSE.000300'].index(QuoteID)]
                self.thread.data[sty].ul = rt_data['LastPrice']
                self.thread.data[sty].ul_yc = rt_data['YClosedPrice']
                self.thread.data[sty].ul_highest = rt_data['HighPrice']
                self.thread.data[sty].ul_lowest = rt_data['LowPrice']
                self.thread.ul_chg[sty] = rt_data['Change'] / (rt_data['LastPrice'] - rt_data['Change'])


def last_C_P(string: str):
    num = len(string) - 1
    while (string[num] != 'C' and string[num] != 'P'):
        num -= 1
    return num + 2

def sub_all_options():
    global g_QuoteZMQ
    global g_QuoteSession
    global q_data
    g_QuoteZMQ = tcore_zmq()
    q_data = g_QuoteZMQ.quote_connect("51878") # 方正 51909，公版 51878

    if q_data["Success"] != "OK":
        print("[quote]connection failed")
        return
    g_QuoteSession = q_data["SessionKey"]
    
    global QuoteID
    QuoteID = []
    data = g_QuoteZMQ.QueryAllInstrumentInfo(g_QuoteSession, "Options")
    for i in range(len(data['Instruments']["Node"])):
        if data['Instruments']["Node"][i]['ENG'] == 'SSE(O)':
            for mat_classification in data['Instruments']["Node"][i]["Node"][0]["Node"][-4 : ]:
                for z in range(2):
                    QuoteID += mat_classification["Node"][z]['Contracts'] # etf50; z =1 for call; z=2 for put
                    Mat[StockType.etf50] += [calendar.datetime.date(int(x[0:4]), int(x[4:6]), int(x[-2:])) for x in mat_classification["Node"][z]['ExpirationDate']]
            for mat_classification in data['Instruments']["Node"][i]["Node"][1]["Node"][-4 : ]:
                for z in range(2):
                    QuoteID += mat_classification["Node"][z]['Contracts'] # h300
                    Mat[StockType.h300] += [calendar.datetime.date(int(x[0:4]), int(x[4:6]), int(x[-2:])) for x in mat_classification["Node"][z]['ExpirationDate']]
        if data['Instruments']["Node"][i]['ENG'] == 'SZSE(O)':
            for mat_classification in data['Instruments']["Node"][i]["Node"][0]["Node"][-4 : ]:
                for z in range(2):
                    QuoteID += mat_classification["Node"][z]['Contracts'] # s300
                    Mat[StockType.s300] += [calendar.datetime.date(int(x[0:4]), int(x[4:6]), int(x[-2:])) for x in mat_classification["Node"][z]['ExpirationDate']]
        if data['Instruments']["Node"][i]['ENG'] == 'CFFEX(O)':
            for mat_classification in data['Instruments']["Node"][i]["Node"][0]["Node"][-6 : ]:
                for z in range(2):
                    QuoteID += mat_classification["Node"][z]['Contracts'] # gz300
                    Mat[StockType.gz300] += [calendar.datetime.date(int(x[0:4]), int(x[4:6]), int(x[-2:])) for x in mat_classification["Node"][z]['ExpirationDate']]

    data = g_QuoteZMQ.QueryAllInstrumentInfo(g_QuoteSession, "Future")
    for i in range(len(data['Instruments']["Node"])):
        if data['Instruments']["Node"][i]['ENG'] == 'CFFEX':
            QuoteID += data['Instruments']["Node"][i]["Node"][2]['Contracts'][1:]
            QuoteID += data['Instruments']["Node"][i]["Node"][3]['Contracts'][1:]
            Mat[FutureType.IF] += [calendar.datetime.date(int(x[0:4]), int(x[4:6]), int(x[-2:])) for x in data['Instruments']["Node"][i]["Node"][2]['ExpirationDate'][1:]]
            Mat[FutureType.IH] += [calendar.datetime.date(int(x[0:4]), int(x[4:6]), int(x[-2:])) for x in data['Instruments']["Node"][i]["Node"][3]['ExpirationDate'][1:]]

    for sty in [StockType.etf50, StockType.h300, StockType.gz300, StockType.s300, FutureType.IF, FutureType.IH]:
        Mat[sty] = sorted(set(Mat[sty]))
        copy = Mat[sty].copy()
        Mat[sty] = {}
        if sty == StockType.gz300:
            for i, mat in enumerate([Maturity.M1, Maturity.M2, Maturity.M3, Maturity.Q1, Maturity.Q2, Maturity.Q3]):
                Mat[sty][mat] = copy[i]
        elif sty in [StockType.etf50, StockType.h300, StockType.s300, FutureType.IF, FutureType.IH]:
            for i, mat in enumerate([Maturity.M1, Maturity.M2, Maturity.Q1, Maturity.Q2]):
                Mat[sty][mat] = copy[i]

    # 删除adj合约
    delect_index_list = []
    for posi, i in enumerate(QuoteID):
        if 'TC.F' in i:
            continue
        k = float(i[last_C_P(i):])
        if 'CFFEX' in i:
            if len([0  for j in [Maturity.M1, Maturity.M2, Maturity.M3] if int(i[14 : 18]) == Mat[StockType.gz300][j].year and  int(i[18 : 20]) == Mat[StockType.gz300][j].month]) > 0:
                if not ((k <= 2500 and k % 25 == 0) or (k > 2500 and k <= 5000 and k % 50 == 0) or (k > 5000 and k <= 10000 and k % 100 == 0) or (k > 10000 and k % 200 == 0)):
                    delect_index_list.append(posi)
            else:
                if not ((k <= 2500 and k % 50 == 0) or (k > 2500 and k <= 5000 and k % 100 == 0) or (k > 5000 and k <= 10000 and k % 200 == 0) or (k > 10000 and k % 400 == 0)):
                    delect_index_list.append(posi)
        else:
            k *= 1000
            if not ((k <= 3000 and k % 50 == 0) or (k > 3000 and k <= 5000 and k % 100 == 0) or (k > 5000 and k <= 10000 and k % 250 == 0) or (k > 10000 and k <= 20000 and k % 500 == 0) or (k > 20000 and k <= 50000 and k % 1000 == 0) or (k > 50000 and k <= 100000 and k % 2500 == 0) or (k > 100000 and k % 5000 == 0)):
                delect_index_list.append(posi)
    for i in sorted(delect_index_list, reverse = True):
        QuoteID.pop(i)

    QuoteID += ['TC.S.SSE.510050', 'TC.S.SSE.510300', 'TC.S.SZSE.159919', 'TC.S.SSE.000300']
    print(QuoteID)

    quote_obj = {"Symbol":"ok", "SubDataType":"REALTIME"}
    for i in QuoteID:
        quote_obj["Symbol"] = i
        s_quote = g_QuoteZMQ.subquote(g_QuoteSession,quote_obj)
        

def screen():

    # choose underlying
    print('请选择品种：')
    print('1. 股指300')
    print('2. ETF50')
    print('3. 沪300')
    print('4. 深300')

    pz = int(input());
    if pz == 1:
        sty = StockType.gz300
    elif pz == 2:
        sty = StockType.etf50
    elif pz == 3:
        sty = StockType.h300
    elif pz == 4:
        sty = StockType.s300

    app = QApplication(sys.argv)
    myScreen = CFtdcMdSpi_Screen(sty)
    myScreen.start()
    app.exec()

def mixed_screen():

    app = QApplication(sys.argv)
    myMixedScreen = CFtdcMdSpi_Mixed_Screen()
    myMixedScreen.start()
    app.exec()


def main(): 

    print('请选择：')
    print('1. 单品种监控屏')
    print('2. 综合屏')

    pz = int(input());
    if pz == 1:
        screen()
    elif pz == 2:
        mixed_screen()
    

if __name__ == '__main__':
    sub_all_options()
    main()