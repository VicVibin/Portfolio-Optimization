import threading

import pandas as pd
import yfinance as yf

# in parallel

# use for loop

# each thread is spawened async
TICKERS = ["AAPL", "GOOGL", "META", "NFLX", "AMZN", "TSLA", "NVDA"]

GLOBAL_DATA = pd.DataFrame()

def parallel_thread(ticker_range):
    global GLOBAL_DATA
    tickers = TICKERS[ticker_range[0]:ticker_range[1]]
    data = yf.download(tickers, start="2018-01-01", end="2024-12-28", threads=False)["Adj Close"]
    data = pd.DataFrame(data)
    GLOBAL_DATA = GLOBAL_DATA.join(data)

MAX_WORKERS = 2#max(len(TICKERS) // 10, 1)


assert MAX_WORKERS < len(TICKERS)
STEP_SIZE = len(TICKERS) // MAX_WORKERS
threads = []
for i in range(0, len(TICKERS), STEP_SIZE):
    t = threading.Thread(target=parallel_thread, args=((i, i+STEP_SIZE),))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print(GLOBAL_DATA,"GLOBAL_DATA")
