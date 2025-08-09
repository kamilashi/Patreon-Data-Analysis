import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt, detrend
import cloudscraper, re, json


URL = "https://graphtreon.com/creator/lazytarts"
LASTNDAYS = 0 # if > 0, only those last N dates will be processed. if <= 0 all the dates will be processed

scraper = cloudscraper.create_scraper()     
html     = scraper.get(URL, timeout=30).text
daily_patrons_pattern  = re.compile( r'dailyGraph_patronSeriesData\s*=\s*(\[[\s\S]*?\])\s*;', re.DOTALL | re.DOTALL)
daily_earnings_pattern  = re.compile( r'dailyGraph_earningsSeriesData\s*=\s*(\[[\s\S]*?\])\s*;', re.DOTALL| re.DOTALL)

daily_patrons_raw = daily_patrons_pattern.search(html)
daily_earnings_raw = daily_earnings_pattern.search(html)

daily_patrons_series = json.loads(daily_patrons_raw.group(1))
daily_earning_series = json.loads(daily_earnings_raw.group(1))

patrons_and_earnings_df = pd.DataFrame(daily_patrons_series, columns=["timestamp_ms", "paid_members"])
patrons_and_earnings_df["daily_earnings"] = pd.Series([v[1] for v in daily_earning_series])
patrons_and_earnings_df["date"] = pd.to_datetime(patrons_and_earnings_df["timestamp_ms"], unit="ms", utc=True).dt.date

time_context = "all time"

if LASTNDAYS > 0:
    patrons_and_earnings_df = patrons_and_earnings_df.tail(LASTNDAYS)
    time_context = "the last " + str(LASTNDAYS) + " days"


#smooth_patrons = detrend(patrons_and_earnings_df["paid_members"])
trend_90d = patrons_and_earnings_df['paid_members'].rolling(window=90, center=False, min_periods=1).mean()
smooth_patrons = patrons_and_earnings_df['paid_members'] - trend_90d
smooth_patrons = smooth_patrons.rolling(window=30, center=True, min_periods=1).median()

sub_peaks_idx = find_peaks_cwt(smooth_patrons, widths=range(20,40), min_snr=0.3)      # tweak prominence / distance / height
sub_peaks_series = patrons_and_earnings_df["date"].iloc[sub_peaks_idx]

domain_length = patrons_and_earnings_df["date"].size
peak_freq = domain_length / sub_peaks_series.size

print("Length: " + str(domain_length))
print("Peaks in subs " + time_context + ":")
print(sub_peaks_series)
print("Ave yearly peak freq: " + time_context + ":")



figureNo = 1

plt.figure(figureNo)
plt.plot(patrons_and_earnings_df["date"], patrons_and_earnings_df["paid_members"])
plt.xlabel('date')
plt.ylabel('paid members')
plt.title('Daily patron counts in ' + time_context)
figureNo += 1

plt.figure(figureNo)
plt.plot(patrons_and_earnings_df["date"], trend_90d)
plt.xlabel('date')
plt.ylabel('paid members')
plt.title('Daily patron counts (global trend) in ' + time_context)
figureNo += 1

plt.figure(figureNo)
plt.plot(patrons_and_earnings_df["date"], smooth_patrons)
plt.xlabel('date')
plt.ylabel('paid members')
plt.title('Daily patron counts (detrended) in ' + time_context)
figureNo += 1

# patrons_and_earnings_df = patrons_and_earnings_df.set_index('date')
# monthly_ts = patrons_and_earnings_df['paid_members'].resample('M').mean()
# monthly_ts.plot(title='Monthly average paid members (time series)')

# plt.figure(2)
# plt.plot(patrons_and_earnings_df["date"], patrons_and_earnings_df["daily_earnings"])
# plt.xlabel('date')
# plt.ylabel('daily earnings')
# plt.title('Daily revenue in' + time_context)

plt.show()