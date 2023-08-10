import pandas as pd
import datetime
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import Run_FFT
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm


def filter_signal(data, filter_type, low_f=None, high_f=None, notch_f=None, notch_quality_factor=30.0,
                  sample_f=None, filter_order=2):
    """Function that creates bandpass filter to ECG data.
    Required arguments:
    -data: 3-column array with each column containing one accelerometer axis
    -type: "lowpass", "highpass" or "bandpass"
    -low_f, high_f: filter cut-offs, Hz
    -sample_f: sampling frequency, Hz
    -filter_order: order of filter; integer
    """

    nyquist_freq = 0.5 * sample_f

    if filter_type == "lowpass":
        low = low_f / nyquist_freq
        b, a = butter(N=filter_order, Wn=low, btype="lowpass")
        # filtered_data = lfilter(b, a, data)
        filtered_data = filtfilt(b, a, x=data)

    if filter_type == "highpass":
        high = high_f / nyquist_freq

        b, a = butter(N=filter_order, Wn=high, btype="highpass")
        # filtered_data = lfilter(b, a, data)
        filtered_data = filtfilt(b, a, x=data)

    if filter_type == "bandpass":
        low = low_f / nyquist_freq
        high = high_f / nyquist_freq

        b, a = butter(N=filter_order, Wn=[low, high], btype="bandpass")
        # filtered_data = lfilter(b, a, data)
        filtered_data = filtfilt(b, a, x=data)

    if filter_type == 'notch':
        b, a = iirnotch(w0=notch_f, Q=notch_quality_factor, fs=sample_f)
        filtered_data = filtfilt(b, a, x=data)

    return filtered_data


def epoch_intensity(df_epoch, column, cutpoint_key, cutpoints_dict, author):

    df = df_epoch.copy()

    vals = []
    for i in df[column]:
        if i < cutpoints_dict[author + cutpoint_key][0]:
            vals.append('sedentary')
        if cutpoints_dict[author + cutpoint_key][0] <= i < cutpoints_dict[author + cutpoint_key][1]:
            vals.append('light')
        if i >= cutpoints_dict[author + cutpoint_key][1]:
            vals.append("moderate")

    return vals


def calculate_daily_activity(df_epoch, cutpoints_dict,
                             df_gait,
                             # df_act,
                             epoch_len=15, column='avm',
                             side="Non-dominant", author='Powell', ignore_sleep=True, ignore_nw=True):

    time_key = 'Timestamp' if 'Timestamp' in df_epoch.columns else 'start_time'

    days = sorted([i for i in set([i.date() for i in df_epoch[time_key]])])
    days = pd.date_range(start=days[0], end=days[-1] + datetime.timedelta(days=1), freq='1D')

    if ignore_sleep:
        df_epoch = df_epoch.loc[df_epoch['sleep_mask'] == 0]
    if ignore_nw:
        df_epoch = df_epoch.loc[df_epoch['nw_mask'] == 0]

    daily_vals = []
    for day1, day2 in zip(days[:], days[1:]):
        df_day = df_epoch.loc[df_epoch["Day"] == day1]

        sed = df_day.loc[df_day[column] < cutpoints_dict[author + side][0]]
        sed_mins = sed.shape[0] * epoch_len / 60

        light = df_day.loc[(df_day[column] >= cutpoints_dict[author + side][0]) &
                           (df_day[column] < cutpoints_dict[author + side][1])]
        light_mins = light.shape[0] * epoch_len / 60

        mod = df_day.loc[(df_day[column] >= cutpoints_dict[author + side][1]) &
                         (df_day[column] < cutpoints_dict[author + side][2])]
        mod_mins = mod.shape[0] * epoch_len / 60

        vig = df_day.loc[df_day[column] >= cutpoints_dict[author + side][2]]
        vig_mins = vig.shape[0] * epoch_len / 60

        df_gait_day = df_gait.loc[(day1 <= df_gait['start_timestamp']) & (df_gait['start_timestamp'] < day2)]

        n_steps = df_gait_day["step_count"].sum()
        walk_mins = df_gait_day["duration"].sum()/60

        avm = df_day['avm'].mean()

        daily_vals.append([day1, sed_mins, light_mins, mod_mins, vig_mins, n_steps, walk_mins, avm])

    df_daily = pd.DataFrame(daily_vals, columns=["Date", "Sed", "Light", "Mod", "Vig", "Steps", "MinutesWalking", "AVM"])
    df_daily["Active"] = df_daily["Light"] + df_daily["Mod"] + df_daily["Vig"]
    df_daily["MVPA"] = df_daily["Mod"] + df_daily["Vig"]
    # df_daily['avm'] = df_act['mean_avm']

    df_daily = pd.concat([df_daily, pd.Series({"Date": "TOTAL", "Sed": df_daily['Sed'].sum(),
                                               "Light": df_daily['Light'].sum(), "Mod": df_daily['Mod'].sum(),
                                               "Vig": df_daily['Vig'].sum(), "Steps": df_daily['Steps'].sum(),
                                               "MinutesWalking": df_daily['MinutesWalking'].sum(),
                                               "Active": df_daily['Active'].sum(), "MVPA": df_daily['MVPA'].sum(),
                                               "WristAVM": df_daily['AVM']})], ignore_index=True)

    return df_daily


def calculate_hand_dom(df_clin, wrist_file):

    if df_clin.shape[0] > 0:
        dom = df_clin.iloc[0]['Hand']

        try:
            if np.isnan(dom):
                print("Dominance data not given in ClinicalInsights file. Assuming right-handed.")
                dom = 'Right'
        except TypeError:
            pass

        # wear = df_clin.iloc[0]['Locations'].split(",")
        # wear = wear[0] if 'wrist' in wear[0] else wear[1]
        wear = wrist_file.split("_")[-1][0]

        # if dom.capitalize() in wear.capitalize():
        if dom.capitalize()[0] == wear.capitalize():
            dominant_wrist = True
        else:
            dominant_wrist = False

    if df_clin.shape[0] == 0:
        dom = 'Right'
        dominant_wrist = True
        print("No clinical data found. Assuming dominant limb data is present.")

    return dominant_wrist


def calculate_hand_dom2(subj,
                        site_code='SBH',
                        colls_csv_file="W:/NiMBaLWEAR/OND09/pipeline/collections.csv",
                        devices_csv_file="W:/NiMBaLWEAR/OND09/pipeline/devices.csv"):

    colls_csv = pd.read_csv(colls_csv_file)
    colls_csv_subj = colls_csv.loc[colls_csv['subject_id'] == subj]

    if colls_csv_subj.shape[0] == 0:
        colls_csv_subj = colls_csv.loc[colls_csv['subject_id'] == subj.split(site_code)[1]]

    devices_csv = pd.read_csv(devices_csv_file)
    devices_csv_subj = devices_csv.loc[devices_csv['subject_id'] == subj]

    if devices_csv_subj.shape[0] == 0:
        devices_csv_subj = devices_csv.loc[devices_csv['subject_id'] == subj.split(site_code)[1]]

    dom_hand = colls_csv_subj['dominant_hand'].iloc[0][0]

    dom_wrist_file = devices_csv_subj.loc[devices_csv_subj['device_location'] == f"{dom_hand}Wrist"]['file_name']

    if len(dom_wrist_file) == 0:
        hand_dom = False
        dom_wrist_file = devices_csv.loc[devices_csv['device_location'] == "LWrist"].iloc[0]['file_name'] if \
            dom_hand == 'R' else devices_csv.loc[devices_csv['device_location'] == "RWrist"].iloc[0]['file_name']

    if len(dom_wrist_file) == 1:
        hand_dom = True

    return hand_dom, colls_csv_subj


def print_medical_summary(df_clin):

    print("\nMedical: ")
    try:
        for i in df_clin['Medical'].iloc[0].split("."):
            print(f"-{i}")
    except (AttributeError, IndexError):
        print("-Nothing")

    print()


def print_activity_log(df_act_log):

    print("\nActivity log:")
    active = {'yes': 'active', 'no': 'inactive', 'unknown': 'unknown', "": "unknown"}

    for row in df_act_log.itertuples():
        try:
            print(
                f"-#{row.Index} ({active[row.active.strip()]}) | {pd.to_datetime(row.start_time).day_name()[:3]}. | "
                f"{row.start_time} ({row.duration} mins) - {row.activity}")

        except:
            try:
                print(
                    f"-#{row.Index} ({active[row.active.strip()]}) | {pd.to_datetime(row.start_time).day_name()[:3]}. | "
                    f"{row.start_time} ({row.duration} mins)- {row.activity}")
            except:
                print(f"-{row.Index} || Error.")


def print_quick_summary(subj, df_clin, df_daily, dominant, cutpoint_author, df_gaitbouts, df_epoch, walk_avm_str=""):

    print(f"\n=============== {subj} quick summary ===============")

    df_full = df_daily.loc[df_daily['Day_dur'] == 1440]

    try:
        print(
            f"\nSubject {subj}: {df_clin.iloc[0]['Age']} years old, cohort = {df_clin.iloc[0]['Cohort']}, "
            f"gait aid use = {df_clin.iloc[0]['GaitAids']}")
    except IndexError:
        print(
            f"\nSubject {subj}: UNKNOWN years old, cohort = UNKNOWN, gait aid use = UNKNOWN")

    print(f"-{cutpoint_author} cutpoints, {'dominant' if dominant else 'non-dominant'} wrist")

    df_gait = df_gaitbouts.loc[df_gaitbouts['duration'] >= 30]
    df_mvpa = df_epoch.loc[df_epoch['intensity'].isin(['moderate', 'vigorous'])]
    df_mvpa_gait = df_mvpa.loc[df_mvpa['cadence'] > 0]
    mvpa_has_gait = df_mvpa_gait.shape[0] * 100 / df_mvpa.shape[0]

    print("\nOverall:")
    print(f"-Steps: {df_full['Steps'].mean():.0f} +/- {df_full['Steps'].std():.0f} "
          f"({df_full['Steps'].min():.0f} - {df_full['Steps'].max():.0f}) per day")

    print(f"-Cadence: {df_gait['cadence'].mean():.1f} +/- {df_gait['cadence'].std():.1f} steps/min (>30-sec bouts)")

    print(f"-Light: {df_full[f'Light_{cutpoint_author}'].mean():.0f} +/- {df_full[f'Light_{cutpoint_author}'].std():.0f} "
          f"({df_full[f'Light_{cutpoint_author}'].min():.0f} - {df_full[f'Light_{cutpoint_author}'].max():.0f}) mins/day")

    print(f"-MVPA: {df_full[f'MVPA_{cutpoint_author}'].mean():.0f} +/- {df_full[f'MVPA_{cutpoint_author}'].std():.0f} "
          f"({df_full[f'MVPA_{cutpoint_author}'].min():.0f} - {df_full[f'MVPA_{cutpoint_author}'].max():.0f}) mins/day")
    print(f"    -Walking occurs in {mvpa_has_gait:.1f}% of MVPA epochs")

    active_mins = df_full[f'Active_{cutpoint_author}']
    print(f"-Active: {active_mins.mean():.0f} +- {active_mins.std():.0f} "
          f"({active_mins.min():.0f} - {active_mins.max():.0f}) mins/day")

    r_mvpa = scipy.stats.pearsonr(df_full[f"MVPA_{cutpoint_author}"], df_full['MinWalking'])
    r_active = scipy.stats.pearsonr(df_full[f"Active_{cutpoint_author}"], df_full['MinWalking'])

    print(f"-Walking: {df_full['MinWalking'].mean():.0f} +- {df_full['MinWalking'].std():.0f} "
          f"({df_full['MinWalking'].min():.0f} - {df_full['MinWalking'].max():.0f}) mins/day")

    print(f"-Walking/activity mapping: MVPA r = {r_mvpa[0]:.3f}; active r = {r_active[0]:.3f}")
    print(f"    -Long walking AVM: {walk_avm_str}")
    print(f"    -Daily AVM ~ walking minutes: r = {scipy.stats.pearsonr(df_daily['AVM'], df_daily['MinWalking'])[0]:.3f}")

    print("\nActivity log:")


def combine_df_daily(df_epoch, cutpoints, df_gait, epoch_len, hand_dom,
                     min_dur=1440, ignore_nw=True, ignore_sleep=True, end_time=None):

    fraysse = calculate_daily_activity(df_epoch=df_epoch, cutpoints_dict=cutpoints, df_gait=df_gait,
                                       epoch_len=epoch_len, column='avm',
                                       side='Dominant' if hand_dom else 'Non-dominant',
                                       author='Fraysse', ignore_sleep=ignore_sleep, ignore_nw=ignore_nw)

    powell = calculate_daily_activity(df_epoch=df_epoch, column='avm',
                                      df_gait=df_gait,
                                      cutpoints_dict=cutpoints, side='Dominant' if hand_dom else 'Non-dominant',
                                      author='Powell', ignore_sleep=ignore_sleep, ignore_nw=ignore_nw)

    df = pd.DataFrame({"Date": powell['Date'], "Sed_Powell": powell['Sed'], "Sed_Fraysse": fraysse['Sed'],
                       "Light_Powell": powell['Light'], "Light_Fraysse": fraysse['Light'],
                       "MVPA_Powell": powell['Mod'] + powell['Vig'], "MVPA_Fraysse": fraysse['Mod'],
                       "Steps": powell['Steps'], 'MinWalking': powell['MinutesWalking'],
                       'AVM': powell['AVM']})
    df['Active_Powell'] = df['Light_Powell'] + df['MVPA_Powell']
    df['Active_Fraysse'] = df['Light_Fraysse'] + df['MVPA_Fraysse']
    df = df.dropna()

    df['Date'] = [i.date() for i in df['Date']]
    df['Day_dur'] = [df_epoch.loc[df_epoch['Day'] == date].shape[0] * epoch_len / 60 for date in df['Date'].unique()]

    if end_time is not None:
        df = df.loc[df['Date'] < end_time]

    df = df.loc[df['Day_dur'] >= min_dur]

    return df


def calculated_logged_intensity(df_act_log, df_epoch, df_steps=None, epoch_len=15, hours_offset=0, quiet=True):
    sed = []
    light = []
    mvpa = []
    step_count = []

    df_act_log = df_act_log.copy()

    if not quiet:
        print("\nMeasured intensities of logged events:")

    epoch_key = 'Timestamp' if 'Timestamp' in df_epoch.columns else 'start_time'
    for row in df_act_log.itertuples():

        try:
            epoch = df_epoch.loc[(df_epoch[epoch_key] >= row.start_time + datetime.timedelta(hours=hours_offset)) &
                                 (df_epoch[epoch_key] <= row.start_time +
                                  datetime.timedelta(hours=hours_offset) + datetime.timedelta(seconds=row.duration * 60))]

            if df_steps is not None:
                steps = df_steps.loc[(df_steps['step_time'] >= row.start_time + datetime.timedelta(hours=hours_offset)) &
                                     (df_steps["step_time"] <= row.start_time +
                                      datetime.timedelta(hours=hours_offset) + datetime.timedelta(seconds=row.duration * 60))]
                step_count.append(steps.shape[0] * 2)

            if df_steps is None:
                step_count.append(None)

            vals = epoch['intensity'].value_counts()

            try:
                sed.append(vals['sedentary'] / (60 / epoch_len))
            except KeyError:
                sed.append(0)

            try:
                light.append(vals['light'] / (60 / epoch_len))
            except KeyError:
                light.append(0)

            try:
                mvpa.append(vals['moderate'] / (60 / epoch_len))
            except KeyError:
                mvpa.append(0)

        except TypeError:
            sed.append(None)
            light.append(None)
            mvpa.append(None)
            step_count.append(None)

    df_act_log['sed'] = sed
    df_act_log['sed_perc'] = df_act_log['sed'] * 100 / df_act_log['duration']
    df_act_log['light'] = light
    df_act_log['light_perc'] = df_act_log['light'] * 100 / df_act_log['duration']
    df_act_log['mvpa'] = mvpa
    df_act_log['mvpa_perc'] = df_act_log['mvpa'] * 100 / df_act_log['duration']

    for col in ['sed', 'sed_perc', 'light', 'light_perc', 'mvpa', 'mvpa_perc']:
        try:
            df_act_log[col] = [round(i, 1) for i in df_act_log[col]]
        except TypeError:
            pass

    if not quiet:
        for row in df_act_log.itertuples():
            if type(row.duration) is int:
                print(f"#{row.Index} {row.activity} || {row.start_time} ({row.duration} minutes) || sed={row.sed}, "
                      f"light={row.light}, mvpa={row.mvpa} minutes")

    df_act_log['steps'] = step_count

    return df_act_log


def calculate_logged_intensity_individual(start_timestamp, end_timestamp, df_epoch, epoch_len=15):

    start_timestamp = pd.to_datetime(start_timestamp)
    end_timestamp = pd.to_datetime(end_timestamp)

    epoch = df_epoch.loc[(df_epoch['Timestamp'] >= start_timestamp) &
                         (df_epoch["Timestamp"] <= end_timestamp)]

    duration = (end_timestamp - start_timestamp).total_seconds() / 60

    vals = epoch['intensity'].value_counts()

    try:
        sed = vals['sedentary'] / (60 / epoch_len)
    except KeyError:
        sed = 0

    try:
        light = vals['light'] / (60 / epoch_len)
    except KeyError:
        light = 0

    try:
        mvpa = vals['moderate'] / (60 / epoch_len)
    except KeyError:
        mvpa = 0

    sedp = sed * 100 / duration
    lightp = light * 100 / duration
    mvpap = mvpa * 100 / duration

    print(f"{start_timestamp} to {end_timestamp} ({duration:.1f} minutes) || sed={sed} ({sedp:.1f}%), "
          f"light={light} ({lightp:.1f}%), mvpa={mvpa} ({mvpap:.1f}%) minutes")


def freq_analysis(obj, ts, subj="", sample_rate=None, channel="", lowpass=None, highpass=None, n_secs=60, stft_mult=5, stft=False, data=None, show_plot=True):

    stamp = pd.to_datetime(ts)

    chn_idx = obj.get_signal_index(channel)

    sample_rate = sample_rate if sample_rate is not None else obj.signal_headers[chn_idx]['sample_rate']

    try:
        idx = Run_FFT.get_index_from_stamp(start=obj.header['start_datetime'], stamp=stamp,
                                           sample_rate=sample_rate)
    except KeyError:
        idx = Run_FFT.get_index_from_stamp(start=obj.header['startdate'], stamp=stamp,
                                           sample_rate=sample_rate)

    end_idx = int(idx + sample_rate * n_secs)

    if data is None:
        d = obj.signals[chn_idx][idx:end_idx]
    if data is not None:
        d = data

    if highpass is not None:
        d = filter_signal(data=d, sample_f=sample_rate, filter_type='highpass', high_f=highpass, filter_order=5)
    if lowpass is not None:
        d = filter_signal(data=d, sample_f=sample_rate, filter_type='lowpass', low_f=lowpass, filter_order=5)

    df_fft = None
    dom_f = None

    if not stft:
        fig, df_fft = Run_FFT.run_fft(data=d, sample_rate=sample_rate, remove_dc=False, show_plot=show_plot)
        if show_plot:
            fig.suptitle(f"OND09_{subj}: {channel}   ||   {stamp}")
            plt.tight_layout()

        dom_f = round(df_fft.loc[df_fft['power'] == df_fft['power'].max()]['freq'].iloc[0], 2)
        print(f"Dominant frequency is ~{dom_f}Hz")

    if stft:
        fig, f, t, Zxx = Run_FFT.plot_stft(data=d, sample_rate=sample_rate, nperseg_multiplier=stft_mult, plot_data=show_plot)
        if show_plot:
            fig.suptitle(f"OND09_{subj}: {channel}   ||   {stamp}")
            if lowpass is not None:
                fig.axes[1].set_ylim(fig.axes[1].get_ylim()[0], lowpass)

    return fig, df_fft, dom_f, [idx, end_idx]


def flag_sleep_epochs(df_epoch, df_sleep_alg):

    time_key = 'Timestamp' if 'Timestamp' in df_epoch.columns else 'start_time'

    sleep_mask = np.zeros(df_epoch.shape[0])

    start_time = df_epoch.iloc[0][time_key]
    epoch_len = (df_epoch.iloc[1][time_key] - start_time).total_seconds()

    for row in df_sleep_alg.itertuples():
        start_i = int(np.floor((row.start_time - start_time).total_seconds() / epoch_len))
        end_i = int(np.floor((row.end_time - start_time).total_seconds() / epoch_len))

        sleep_mask[start_i:end_i] = 1

    return sleep_mask


def flag_nonwear_epochs(df_epoch, df_nw):

    time_key = 'Timestamp' if 'Timestamp' in df_epoch.columns else 'start_time'

    nw_mask = np.zeros(df_epoch.shape[0])

    start_time = df_epoch.iloc[0][time_key]
    epoch_len = (df_epoch.iloc[1][time_key] - start_time).total_seconds()

    for row in df_nw.itertuples():
        start_i = int(np.floor((row.start_time - start_time).total_seconds() / epoch_len))
        end_i = int(np.floor((row.end_time - start_time).total_seconds() / epoch_len))

        nw_mask[start_i:end_i] = 1

    return nw_mask


def print_walking_intensity_summary(df_gait, df_epoch, cutpoints=(62.5, 92.5), min_dur=30):

    time_key = 'Timestamp' if 'Timestamp' in df_epoch.columns else 'start_time'

    df = df_gait.loc[df_gait['duration'] >= min_dur]
    n = df.shape[0]
    dur = df['duration'].sum()/60
    epoch_len = (df_epoch.iloc[1][time_key] - df_epoch.iloc[0][time_key]).total_seconds()
    print("\n====== WRIST INTENSITY DURING WALKING BOUTS =======")

    if n >= 1:
        avg_avm = []
        mvpa = 0
        light = 0
        sed = 0
        for bout in df.itertuples():

            df_e = df_epoch.loc[(df_epoch[time_key] >= bout.start_timestamp) &
                                (df_epoch[time_key] < bout.end_timestamp)]

            mvpa += list(df_e['intensity']).count('moderate') * epoch_len / 60
            light += list(df_e['intensity']).count('light') * epoch_len / 60
            sed += list(df_e['intensity']).count('sedentary') * epoch_len / 60

            for epoch in df_e.itertuples():
                avg_avm.append(epoch.avm)

        avg = np.mean(avg_avm)
        sd = np.std(avg_avm)

        if avg < cutpoints[0]:
            i = 'sedentary'
        if cutpoints[0] <= avg < cutpoints[1]:
            i = 'light'
        if cutpoints[1] <= avg:
            i = 'MVPA'

        print(f"-Wrist counts during gait bouts of at least {min_dur} seconds long:")
        print(f"     -{n} gait bouts found totalling {dur:.0f} minutes")
        print(f"     -Cadence of {df['cadence'].mean():.1f} +- {df['cadence'].std():.1f} steps/min")
        print(f"          -Expected intensity: {'MVPA' if df['cadence'].mean() >= 100 else 'light'}")
        print(f"     -{mvpa/dur*100:.1f}% MVPA, {light*100/dur:.1f}% light, and {sed*100/dur:.1f}% sedentary")
        print(f"     -Counts: {avg:.1f} +- {sd:.1f}")
        print(f"          -Falls into {i} intensity using cutpoints of {[round(j, 1) for j in cutpoints]}")

    if n == 0:
        print("-Not enough gait bouts found.")
        return 0

    return f"{avg:.0f} +- {sd:.0f}"


def check_intensity_window(df_epoch, start, end):

    print(f"\nActivity intensity summary from {start} to {end}:")
    time_key = 'Timestamp' if 'Timestamp' in df_epoch.columns else 'start_time'

    epoch_len = (df_epoch.iloc[1][time_key] - df_epoch.iloc[0][time_key]).total_seconds()
    d = df_epoch.loc[(df_epoch[time_key] >= start) & (df_epoch[time_key] <= end)]
    vals = d['intensity'].value_counts() / (60 / epoch_len)

    for i in vals.index:
        print(f"    -{i} = {vals.loc[i]} minutes ({vals.loc[i] * 100 / sum(vals):.1f}%)")


def calculate_window_cadence(start, stop, df_steps, ankle_obj=None, show_plot=False, axis='Gyroscope z'):

    if isinstance(stop, datetime.timedelta):
        stop = pd.to_datetime(start) + stop

    df = df_steps.loc[(df_steps['step_time'] >= pd.to_datetime(start)) &
                      (df_steps['step_time'] < pd.to_datetime(stop))].reset_index(drop=True)

    n_steps = df.shape[0]

    if n_steps >= 2:
        dt = (df['step_time'].iloc[-1] - df['step_time'].iloc[0]).total_seconds()

        mean_cad = 2 * n_steps / dt * 60
        print(f"\nMean cadence between {start} and {stop} is {mean_cad:.1f} spm")
        print(f"-{n_steps*2:.0f} steps in {dt:.0f} seconds/{dt/60:.1f} minutes")

    if n_steps < 2:
        print("\nNot enough steps to do anything.")

    if show_plot:
        start_key = 'startdate' if 'startdate' in ankle_obj.header.keys() else 'start_datetime'
        fig, ax = plt.subplots(1, figsize=(12, 5))
        start_idx = int((pd.to_datetime(start) - ankle_obj.header[start_key]).total_seconds() * ankle_obj.signal_headers[0]['sample_rate'])
        stop_idx = int((pd.to_datetime(stop) - ankle_obj.header[start_key]).total_seconds() * ankle_obj.signal_headers[0]['sample_rate'])

        ax.plot(ankle_obj.ts[start_idx:stop_idx], ankle_obj.signals[ankle_obj.get_signal_index(axis)][start_idx:stop_idx], color='black', zorder=0)

        ax.scatter(df['step_time'], ankle_obj.signals[ankle_obj.get_signal_index(axis)][df['step_idx']], color='red', zorder=1)


def epoch_cadence(epoch_timestamps, df_steps):

    epoch_len = (epoch_timestamps[1] - epoch_timestamps[0]).total_seconds()

    cads = []

    print(f"\nCalculating average cadence in {epoch_len}-second windows...")
    for stamp in tqdm(epoch_timestamps):
        df = df_steps.loc[(df_steps['step_time'] >= stamp) &
                          (df_steps['step_time'] < stamp + datetime.timedelta(seconds=epoch_len))]
        n_steps = df.shape[0]

        cad = n_steps / epoch_len * 60 * 2
        cads.append(cad)

    return cads


def adjust_timestamps(df, n_seconds=0, colnames=()):

    print(f"\nAdjusting {colnames} by {n_seconds} seconds...")

    for column in colnames:
        col = []
        for i in df[column]:
            try:
                col.append(i + datetime.timedelta(seconds=n_seconds))
            except:
                col.append(i)
        #df[column] = pd.to_datetime(df[column])
        #df[column] += timedelta(seconds=n_seconds)
        df[column] = col

    return df


def count_datapoints(data_objs=()):

    print("\nCounting datapoints...")
    datapoints = 0
    for obj in data_objs:
        for signal in obj.signals:
            datapoints += len(signal)

    s = f"{datapoints:,}".replace(",", " ")
    print(f"-Given data contains {s} datapoints")


def collection_summary(df_daily, wrist_obj=None, ankle_obj=None, chest_obj=None):

    data = []

    if wrist_obj is not None:
        data.append(['wrist', wrist_obj.ts[0], wrist_obj.ts[-1]])
    if wrist_obj is None:
        data.append(['wrist', None, None])

    if ankle_obj is not None:
        data.append(['ankle', ankle_obj.ts[0], ankle_obj.ts[-1]])
    if ankle_obj is None:
        data.append(['ankle', None, None])

    if chest_obj is not None:
        data.append(['chest', chest_obj.ts[0], chest_obj.ts[-1]])
    if chest_obj is None:
        data.append(['chest', None, None])

    data.append(['full_days', df_daily.iloc[0]['Date'], df_daily.iloc[-1]['Date']])

    return pd.DataFrame(data, columns=['source', 'start', 'end'])