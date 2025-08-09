import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.ndimage import grey_closing
import cloudscraper, re, json
import math


URL = "https://graphtreon.com/creator/lazytarts"
LASTNDAYS = 0#2 * 365 # if > 0, only those last N dates will be processed. if <= 0 all the dates will be processed

START_DATE = pd.to_datetime("2022-07-01")
END_DATE = pd.to_datetime("2024-05-01")

releases = []
with open("releases.json", "r") as f:
    releases = json.load(f)

releases_df = pd.DataFrame(releases, columns=["version", "date", "price"])
releases_df["date"] = pd.to_datetime(releases_df["date"])

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
patrons_and_earnings_df["date"] = pd.to_datetime(patrons_and_earnings_df["timestamp_ms"], unit="ms")

patrons_and_earnings_df["month"] = patrons_and_earnings_df["date"].dt.to_period("M") 

time_context = " in all time"

if LASTNDAYS > 0:
    START_DATE = patrons_and_earnings_df["date"].iloc[-LASTNDAYS]
    END_DATE = patrons_and_earnings_df["date"].iloc[-1]
    time_context = " in the last " + str(LASTNDAYS) + " days"

if START_DATE != END_DATE:
    mask = (patrons_and_earnings_df["date"] >= START_DATE) & (patrons_and_earnings_df["date"] <= END_DATE)
    patrons_and_earnings_df = patrons_and_earnings_df.loc[mask].reset_index(drop=True)

    mask = (releases_df["date"] >= START_DATE) & (releases_df["date"] <= END_DATE)
    releases_df = releases_df.loc[mask].reset_index(drop=True)

    time_context = " from " + str(START_DATE) + " to " + str(END_DATE)



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

peak_dates = patrons_and_earnings_df["date"].iloc[peaks_idx_raw]
print(peak_dates)

# slice up the data into inter-release segments
starts = peaks_idx_raw[:-1]
ends   = np.r_[starts[1:], peaks_idx_raw[len(peaks_idx_raw) - 1]]  

patrons_segments = [patrons_denoised[s:e] for s, e in zip(starts, ends)]

domain_length = patrons_and_earnings_df["date"].size

min_patrons = patrons_detrended.min()
max_patrons = patrons_detrended.max()

print("Length: " + str(domain_length))
#print("Peaks in subs " + time_context + ":")
#print(peak_dates)

figureNo = 1

plt.figure(figureNo)
plt.plot(patrons_and_earnings_df["date"], patrons_and_earnings_df["paid_members"])
plt.xlabel('date')
plt.ylabel('paid members')
plt.title('Daily patron counts' + time_context)
figureNo += 1

plt.figure(figureNo)
plt.plot(patrons_and_earnings_df["date"], global_trend)
plt.xlabel('date')
plt.ylabel('paid members')
plt.title('Daily patron counts (global trend)' + time_context)
figureNo += 1

price_colors = ['orange', 'red']

plt.figure(figureNo)
plt.plot(patrons_and_earnings_df["date"], patrons_detrended, label='detrended', color='blue')
plt.plot(patrons_and_earnings_df["date"], patrons_denoised, label='de-noised, upper envelope', color='orange')
#plt.plot(patrons_and_earnings_df["date"], patrons_peaks, label='smoothed peaks', color='orange')
plt.vlines(x = peak_dates.to_numpy(), ymin = min_patrons, ymax = max_patrons, color = 'green', linestyle='--', label = 'peaks')

release_by_price = releases_df.groupby("price")
color_idx = 0
for price, group in release_by_price:
    plt.vlines(x = group["date"], ymin = min_patrons, ymax = max_patrons, color = price_colors[color_idx], linestyle=':', label = str(price) + '$ release')
    color_idx += 1
    
plt.xlabel('date')
plt.ylabel('paid members')
plt.title('Daily patron counts (with and without monthly dips)' + time_context)
figureNo += 1

plt.figure(figureNo)
n_segments = len(patrons_segments)

n_cols = math.ceil(math.sqrt(n_segments))
n_rows = math.ceil(n_segments / n_cols)
subplot_no = 0


for segment in patrons_segments:
    index = subplot_no
    subplot_no += 1
    plt.subplot(n_rows, n_cols, subplot_no)
    plt.plot(patrons_and_earnings_df["date"].iloc[starts[index] : ends[index]], segment)
    plt.title('Idle window of ' + str(ends[index] - starts[index]) + " days")
    plt.xticks(rotation=90, fontweight='light',  fontsize='x-small')

plt.subplots_adjust(hspace= 0.25 * n_rows)

plt.show()