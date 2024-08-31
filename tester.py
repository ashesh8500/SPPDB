import yfinance as yf
import yahooquery as yq
import pandas as pd
from pdb import set_trace


def main():
    qqq = yf.Ticker('QQQ')
    info = qqq.get_info()
    set_trace()

if __name__ == '__main__':
    main()
