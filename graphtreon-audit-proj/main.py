import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.ndimage import grey_closing
import cloudscraper, re, json


URL = "https://graphtreon.com/creator/lazytarts"
LASTNDAYS = 2 * 365 # if > 0, only those last N dates will be processed. if <= 0 all the dates will be processed

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

# montly dips
monthly_dip_window = 30
patrons_denoised = grey_closing(patrons_detrended, size=monthly_dip_window)

# peaks
peak_width = 4
peaks_idx_raw,_ = sig.find_peaks(patrons_denoised, width = peak_width)  

peaks_idx = []
peak_no = 0

seasonal_fc_norm = (1 / (35)) / (0.5) # monthly dips happen every 30 days, so we plug that into our new cutoff-frequency
print("Seasonal cutoff freq: " + str(seasonal_fc_norm))
lp_b, lp_a = sig.cheby2(N=6, rs=40, Wn = seasonal_fc_norm)

patrons_peaks = sig.filtfilt(lp_b, lp_a, patrons_denoised)

def get_slope(center, radius, data):
    left = center - radius
    right = center + radius 
    slope = (data[right] - data[left]) / (right - left)
    return slope

# potentially filter out start and end maxima and any stray ones in between
for peak_idx in peaks_idx_raw:
    peak_no += 1
    zero_slope_thrs = 0.1

    radius = int(peak_width / 2) 
    left = peak_idx - radius
    right = peak_idx + radius

    left_slope = get_slope(left, 1, patrons_peaks)
    right_slope = get_slope(right, 1, patrons_peaks)
    print("slopes around peak # " + str(peak_no) + ": " + str(left_slope) + " : " + str(right_slope))
    if np.sign(left_slope) != np.sign(right_slope):
        peaks_idx.append(peak_idx)

peak_dates = patrons_and_earnings_df["date"].iloc[peaks_idx_raw]

domain_length = patrons_and_earnings_df["date"].size

min_patrons = patrons_detrended.min()
max_patrons = patrons_detrended.max()

print("Length: " + str(domain_length))
print("Peaks in subs " + time_context + ":")
print(peak_dates)

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
plt.plot(patrons_and_earnings_df["date"], patrons_detrended, label='detrended', color='blue')
plt.plot(patrons_and_earnings_df["date"], patrons_denoised, label='de-noised, upper envelope', color='red')
#plt.plot(patrons_and_earnings_df["date"], patrons_peaks, label='smoothed peaks', color='orange')
plt.vlines(x = peak_dates.to_numpy(), ymin = min_patrons, ymax = max_patrons, color = 'green', label = 'release')
plt.xlabel('date')
plt.ylabel('paid members')
plt.title('Daily patron counts (with and without monthly dips) in ' + time_context)
figureNo += 1

plt.show()