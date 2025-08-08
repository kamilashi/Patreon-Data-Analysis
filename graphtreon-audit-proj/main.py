import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cloudscraper, re, json

URL = "https://graphtreon.com/creator/lazytarts"

scraper = cloudscraper.create_scraper()     
html     = scraper.get(URL, timeout=30).text
daily_patrons_pattern  = re.compile( r'dailyGraph_patronSeriesData\s*=\s*(\[[\s\S]*?\])\s*;', re.DOTALL | re.DOTALL)
daily_earnings_pattern  = re.compile( r'dailyGraph_earningsSeriesData\s*=\s*(\[[\s\S]*?\])\s*;', re.DOTALL| re.DOTALL)

daily_patrons_raw = daily_patrons_pattern.search(html)
daily_earnings_raw = daily_earnings_pattern.search(html)

daily_patrons_series = json.loads(daily_patrons_raw.group(1))
daily_earning_series = json.loads(daily_earnings_raw.group(1))

patrons_df = pd.DataFrame(daily_patrons_series, columns=["timestamp_ms", "paid_members"])
patrons_df["date"] = pd.to_datetime(patrons_df["timestamp_ms"], unit="ms", utc=True).dt.date

earnings_df = pd.DataFrame(daily_earning_series, columns=["timestamp_ms", "daily_earnings"])
earnings_df["date"] = pd.to_datetime(earnings_df["timestamp_ms"], unit="ms", utc=True).dt.date

print(patrons_df)

plt.figure(1)
plt.plot(patrons_df["date"], patrons_df["paid_members"])
plt.xlabel('date')
plt.ylabel('paid_members')
plt.title('All time daily patron counts')

plt.figure(2)
plt.plot(earnings_df["date"], earnings_df["daily_earnings"])
plt.xlabel('date')
plt.ylabel('daily_earnings')
plt.title('All time daily revenue')

plt.show()