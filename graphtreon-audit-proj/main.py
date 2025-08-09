import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import cloudscraper, re, json


URL = "https://graphtreon.com/creator/lazytarts"
LASTNDAYS = 365 # if > 0, only those last N dates will be processed. if <= 0 all the dates will be processed

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

# put the cutoff-frequency somewhere very low for the global trend
global_fc_norm = (1 / (365 / 0.5)) / (0.5) 
print("Global cutoff freq: " + str(global_fc_norm))
lp_b, lp_a = sig.butter(N=4, Wn = global_fc_norm)

# global trend
global_trend = sig.filtfilt(lp_b, lp_a, patrons_and_earnings_df['paid_members'])
patrons_detrended = patrons_and_earnings_df['paid_members'] - global_trend


seasonal_fc_norm = (1 / (35)) / (0.5) # monthly dips happen every 30 days, so we plug that into our new cutoff-frequency
print("Seasonal cutoff freq: " + str(seasonal_fc_norm))
lp_b, lp_a = sig.cheby2(N=6, rs=40, Wn = seasonal_fc_norm)

# montly dips
patrons_residual = sig.filtfilt(lp_b, lp_a, patrons_detrended)


sub_peaks_idx = sig.find_peaks_cwt(patrons_detrended, widths=range(20,40), min_snr=0.3)      
sub_peaks_series = patrons_and_earnings_df["date"].iloc[sub_peaks_idx]

domain_length = patrons_and_earnings_df["date"].size
peak_freq = domain_length / sub_peaks_series.size

print("Length: " + str(domain_length))
print("Peaks in subs " + time_context + ":")
print(sub_peaks_series)



figureNo = 1

plt.figure(figureNo)
plt.plot(patrons_and_earnings_df["date"], patrons_and_earnings_df["paid_members"])
plt.xlabel('date')
plt.ylabel('paid members')
plt.title('Daily patron counts in ' + time_context)
figureNo += 1

plt.figure(figureNo)
plt.plot(patrons_and_earnings_df["date"], global_trend)
plt.xlabel('date')
plt.ylabel('paid members')
plt.title('Daily patron counts (global trend) in ' + time_context)
figureNo += 1

plt.figure(figureNo)
plt.plot(patrons_and_earnings_df["date"], patrons_detrended)
plt.xlabel('date')
plt.ylabel('paid members')
plt.title('Daily patron counts (detrended) in ' + time_context)
figureNo += 1

plt.figure(figureNo)
plt.plot(patrons_and_earnings_df["date"], patrons_residual)
plt.xlabel('date')
plt.ylabel('paid members')
plt.title('Daily patron counts (residual) in ' + time_context)
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