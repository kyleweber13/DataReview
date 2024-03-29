import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from datetime import timedelta as timedelta
import numpy as np
import scipy.stats
import pandas as pd
from Analysis import filter_signal
xfmt = mdates.DateFormatter("%a\n%b-%d")
xfmt_raw = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")


def summary_plot(df_daily, author, subj, df_sleep):

    author = author.capitalize()

    df = df_daily[["Date", 'Steps', 'MinWalking', 'AVM']].copy()

    for col in ["Sed", "Light", "MVPA"]:
        df.loc[:, col] = df_daily[f"{col}_{author}"]

    df = df_daily.copy()

    fig, axes = plt.subplots(3, 3, figsize=(16, 9), sharex='all')
    plt.suptitle(f"OND09_{subj}")
    plt.subplots_adjust(left=.05, right=.975, wspace=.15)
    plt.rcParams['axes.labelsize'] = '8'

    axes[0][0].bar(df["Date"], df["Steps"], zorder=1,
                   edgecolor='black', color='dodgerblue', label='Steps', alpha=.75)
    axes[0][0].axhline(y=7000, color='green', linestyle='dashed', zorder=0)
    axes[0][0].set_title("Step Counts")

    axes[0][1].bar(df["Date"], df["MinWalking"], edgecolor='black', color='grey', alpha=.75)
    axes[0][1].set_ylim(0, )
    axes[0][1].set_title("Minutes Walking")

    axes[0][2].set_title("Sleep Hours")
    axes[0][2].bar(df_sleep["Date"], df_sleep["TST (h)"], color='navy', edgecolor='black', alpha=.75, zorder=1)
    axes[0][2].fill_between(x=[df.min()["Date"]+timedelta(hours=-12), df.max()["Date"]+timedelta(hours=12)],
                            y1=7, y2=9, color='green', alpha=.25)
    axes[0][2].set_ylim(0, 10)

    axes[1][2].bar(df["Date"], df[f"Sed_{author}"], edgecolor='black', color='silver', label='Sed', alpha=.75)
    axes[1][2].set_ylim(0, 1510)
    axes[1][2].set_yticks(np.arange(0, 1501, 250))
    axes[1][2].axhline(y=1440, color='black', linestyle='dashed')
    axes[1][2].set_title(f"Sedentary Minutes ({author})")

    axes[1][0].bar(df["Date"], df[f"Light_{author}"], edgecolor='black', color='forestgreen', label="Light", alpha=.75)
    axes[1][0].set_title(f"Light Minutes ({author})")

    axes[1][1].bar(df["Date"], df[f'MVPA_{author}'], edgecolor='black', color='orange', label="MVPA", alpha=.75)
    axes[1][1].set_title(f"MVPA Minutes ({author})")

    axes[2][0].plot(df["Date"], df['AVM'], color='purple',
                    marker='o', markerfacecolor='purple', markeredgecolor='black', label="AVM", alpha=.75)
    axes[2][0].set_title(f"AVM")

    axes[2][1].bar(df["Date"], df[f'Active_{author}'], edgecolor='black', color='gold', label="Active", alpha=.75)
    axes[2][1].set_title(f"Active Minutes ({author})")

    axes[2][2].scatter(df['Date'], 100 * df[f"Active_{author}"]/df['MinWalking'], label='Active', color='gold')
    axes[2][2].scatter(df['Date'], 100 * df[f"Light_{author}"]/df['MinWalking'], label='Light', color='forestgreen')
    axes[2][2].scatter(df['Date'], 100 * df[f"MVPA_{author}"]/df['MinWalking'], label='MVPA', color='orange')
    axes[2][2].legend()
    axes[2][2].axhline(y=100, color='black', linestyle='dashed')
    axes[2][2].set_ylabel("% of minutes walking")
    axes[2][2].set_title("Activity relative to minutes walking")

    # sets ylims for active minutes, light minutes, MVPA minutes, and walking minutes to the same scale
    max_active_ylim = max([axes[1][1].get_ylim()[1], axes[1][0].get_ylim()[1],
                           axes[0][1].get_ylim()[1], axes[2][1].get_ylim()[1]])
    axes[0][1].set_ylim(0, max_active_ylim)
    axes[1][0].set_ylim(0, max_active_ylim)
    axes[1][1].set_ylim(0, max_active_ylim)
    axes[2][1].set_ylim(0, max_active_ylim)

    axes[1][0].xaxis.set_major_formatter(xfmt)
    axes[1][1].xaxis.set_major_formatter(xfmt)
    axes[1][2].xaxis.set_major_formatter(xfmt)
    axes[-1][0].tick_params(axis='x', labelsize=8)
    axes[-1][1].tick_params(axis='x', labelsize=8)
    axes[-1][2].tick_params(axis='x', labelsize=8)

    plt.tight_layout()


def generate_scatter(df, x, y, label="", color='black', axes=None, grid=False):

    try:
        reg = scipy.stats.linregress(x=df[x], y=df[y])

        m = reg.slope
        b = reg.intercept
        r = scipy.stats.pearsonr(df[x], df[y])[0]
        p = reg.pvalue

        x_vals = np.linspace(df[x].min(), df[x].max(), 100)
        y_vals = [i*m + b for i in x_vals]

        if axes is None:
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.scatter(df[x], df[y], color=color)
            ax.plot(x_vals, y_vals, color=color, label=f"{label} (r={round(r, 3)})")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.legend()
            if grid:
                ax.grid()

        if axes is not None:
            axes.scatter(df[x], df[y], color=color)
            axes.plot(x_vals, y_vals, color=color, label=f"{label} (r={round(r, 3)})")
            axes.set_xlabel(x)
            axes.set_ylabel(y)
            axes.legend()

            if grid:
                axes.grid()

    except ValueError:
        pass


def gen_relationship_graph(daily_df, df_gait, author):

    author = author.capitalize()

    df = daily_df.copy()

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 3, 6)

    generate_scatter(df, "Steps", f"Active_{author}", axes=ax1)
    generate_scatter(df, "Steps", "MinWalking", axes=ax2)
    generate_scatter(df, "MinWalking", f"MVPA_{author}", label='MVPA', axes=ax3, color='orange')
    generate_scatter(df, "MinWalking", f"Active_{author}", label='Active', axes=ax3, color='dodgerblue', grid=True)

    ax3.plot(np.arange(max([ax3.get_xlim()[1], ax3.get_ylim()[1]])),
             np.arange(max([ax3.get_xlim()[1], ax3.get_ylim()[1]])), color='limegreen')
    generate_scatter(df, "Steps", f"Sed_{author}", axes=ax4)

    ax6.scatter(np.arange(1, df.shape[0]+1), df["MinWalking"]/df[f'Active_{author}'], color='dodgerblue', label='Active')
    ax6.axhline(np.mean(df["MinWalking"]/df[f'Active_{author}']), color='dodgerblue')
    ax6.scatter(np.arange(1, df.shape[0]+1), df["MinWalking"]/df[f'MVPA_{author}'], color='orange', label='MVPA')
    ax6.axhline(np.mean(df["MinWalking"]/df[f'MVPA_{author}']), color='orange')
    ax6.set_ylabel("Mins Walking / Activity")
    ax6.set_xlabel("Day number")
    ax6.legend()
    ax6.axhline(y=1, linestyle='dotted', color='limegreen')

    df_g = df_gait.loc[df_gait['duration'] >= 10]
    ax5.hist(df_g["cadence"], color='grey', edgecolor='black', alpha=.75,
             bins=np.arange(df_g['cadence'].min()*.9, df_g['cadence'].max()*1.1, 5))
    ax5.axvline(x=df_g["cadence"].median(), color='red', label='Median')
    ax5.set_xlabel("Cadence")
    ax5.set_yticks([])
    ax5.legend()

    plt.tight_layout()


def compare_cutpoints(df_daily):

    df = df_daily.copy()  # ignores

    fig, ax = plt.subplots(3, 4, sharex='col', figsize=(12, 8))

    # sedentary
    ax[0][0].bar(df['Date'], df['Sed_Powell'], color='grey', edgecolor='black')
    ax[0][0].axhline(df['Sed_Powell'].mean(), color='grey', linestyle='dashed')
    ax[0][0].set_ylabel("Minutes")
    ax[0][0].set_title("Sedentary (Powell)")
    ax[0][0].set_ylim(0, max([df['Sed_Powell'].max(), df['Sed_Fraysse'].max()]) * 1.05)

    ax[1][0].bar(df['Date'], df['Sed_Fraysse'], color='grey', edgecolor='black')
    ax[1][0].axhline(df['Sed_Fraysse'].mean(), color='grey', linestyle='dashed')
    ax[1][0].set_title("Sedentary (Fraysse)")
    ax[1][0].set_ylabel("Minutes")
    ax[1][0].set_ylim(0, max([df['Sed_Powell'].max(), df['Sed_Fraysse'].max()]) * 1.05)

    ax[2][0].bar(df['Date'], df['Sed_Fraysse'] / df['Sed_Powell'], color='grey', edgecolor='black')
    ax[2][0].axhline((df['Sed_Fraysse'] / df['Sed_Powell']).mean(), color='grey', linestyle='dashed')
    ax[2][0].set_title("Fraysse:Powell ratio")
    ax[2][0].set_ylabel("Ratio")

    # light
    ax[0][1].bar(df['Date'], df['Light_Powell'], color='green', edgecolor='black')
    ax[0][1].axhline(df['Light_Powell'].mean(), color='green', linestyle='dashed')
    ax[0][1].set_title("Light (Powell)")
    ax[0][1].set_ylim(0, max([df['Light_Powell'].max(), df['Light_Fraysse'].max()]) * 1.05)

    ax[1][1].bar(df['Date'], df['Light_Fraysse'], color='green', edgecolor='black')
    ax[1][1].axhline(df['Light_Fraysse'].mean(), color='green', linestyle='dashed')
    ax[1][1].set_title("Light (Fraysse)")
    ax[1][1].set_ylim(0, max([df['Light_Powell'].max(), df['Light_Fraysse'].max()]) * 1.05)

    ax[2][1].bar(df['Date'], df['Light_Fraysse'] / df['Light_Powell'], color='green', edgecolor='black')
    ax[2][1].axhline((df['Light_Fraysse'] / df['Light_Powell']).mean(), color='green', linestyle='dashed')
    ax[2][1].set_title("Fraysse:Powell ratio")

    # mvpa
    ax[0][2].bar(df['Date'], df['MVPA_Powell'], color='orange', edgecolor='black')
    ax[0][2].axhline(df['MVPA_Powell'].mean(), color='orange', linestyle='dashed')
    ax[0][2].set_title("MVPA (Powell)")
    ax[0][2].set_ylim(0, max([df['MVPA_Powell'].max(), df['MVPA_Fraysse'].max()]) * 1.05)

    ax[1][2].bar(df['Date'], df['MVPA_Fraysse'], color='orange', edgecolor='black')
    ax[1][2].axhline(df['MVPA_Fraysse'].mean(), color='orange', linestyle='dashed')
    ax[1][2].set_title("MVPA (Fraysse)")
    ax[1][2].set_ylim(0, max([df['MVPA_Powell'].max(), df['MVPA_Fraysse'].max()]) * 1.05)

    ax[2][2].bar(df['Date'], df['MVPA_Fraysse'] / df['MVPA_Powell'], color='orange', edgecolor='black')
    ax[2][2].axhline((df['MVPA_Fraysse'] / df['MVPA_Powell']).mean(), color='orange', linestyle='dashed')
    ax[2][2].set_title("Fraysse:Powell ratio")

    # light + mvpa
    ax[0][3].bar(df['Date'], df['Active_Powell'], color='dodgerblue', edgecolor='black')
    ax[0][3].axhline(df['Active_Powell'].mean(), color='dodgerblue', linestyle='dashed')
    ax[0][3].set_title("Active (Powell)")
    ax[0][3].set_ylim(0, max([df['Active_Powell'].max(), df['Active_Fraysse'].max()]) * 1.05)

    ax[1][3].bar(df['Date'], df['Active_Fraysse'], color='dodgerblue', edgecolor='black')
    ax[1][3].axhline(df['Active_Fraysse'].mean(), color='dodgerblue', linestyle='dashed')
    ax[1][3].set_title("Active (Fraysse)")
    ax[1][3].set_ylim(0, max([df['Active_Powell'].max(), df['Active_Fraysse'].max()]) * 1.05)

    ax[2][3].bar(df['Date'], df['Active_Fraysse'] / df['Active_Powell'], color='dodgerblue', edgecolor='black')
    ax[2][3].axhline((df['Active_Fraysse'] / df['Active_Powell']).mean(), color='dodgerblue', linestyle='dashed')
    ax[2][3].set_title("Fraysse:Powell ratio")

    for i in range(4):
        for tick in ax[2][i].get_xticklabels():
            tick.set_rotation(90)

    plt.tight_layout()


def plot_raw(ds_ratio=1, dominant=True, author='Powell', subj="", study_code='OND09', cutpoints=(), alpha=.5,
             df_hr=None, df_posture=None, df_epoch=None, df_sleep=None, df_sptw=None,
             df_steps=None, df_gait=None, df_act_log=None,
             df_ankle_sync=None, df_chest_sync=None,
             ankle_gyro=True, intensity_markers=False,
             shade_gait_bouts=False, min_gait_dur=15, mark_steps=False, bout_steps_only=True,
             wrist_gyro=False, highpass_accel=True,
             show_activity_log=False, shade_sleep_windows=False,
             wrist=None, wrist_nw=None,
             ankle=None, ankle_nw=None, ecg=None):
    """Function to plot tons of stuff. Basically plots all available data. Input items as None to skip if no
       specific argument exists.

        arguments:
        -subj: str for subject ID

        -analysis:
            -dominant: boolean whether wrist IMU worn on dominant wrist
            -author: "Powell" or "Fraysse" for which cutpoints to use
            -cutpoints: dictionary with keys corresponding to {Author}"Dominant" or {Author}{Non-dominant}

        -data:
            -tabular
                -df_hr: df with columnes 'timestamp' and 'rate' for HR
                -df_posture: df with posture bouts
                -df_epoch: df of epoched wrist AVM data
                -df_sleep_alg: df of sleep bouts
                -df_steps: df containing every step you take, every move you make. I'll be watching you.
                -df_gait: df of gait bouts
                -df_act_log: df of activity log
                -wrist_nw/ankle_nw: df of non-wear bouts
            -time series
                -wrist/ankle/ecg: objects containing raw data
                -wrist_gyro/ankle_gyro: boolean. If False, plots accelerometer data, gyroscope if True
                -highpass_accel: if True, runs .1Hz highpass filter on accelerometer data to remove gravity
        -Plotting options:
            -shade_gait_bouts: boolean whether to show gait bouts over ankle data in gold
            -min_gait_dur: minimum bout length in seconds to be included if shade_gait_bouts is True

            -mark_steps: triangular markers on y-axis ankle accelerometer/gyroscope if True
            -bout_steps_only: boolean whether to include all steps (True) or only steps in bouts of >= min_gait_dur

            -show_activity_log: boolean to show logged activities over wrist AVM data. Prints formatted log to console
            -shade_sleep_windows: boolean to show sleep windows in purple over wrist data

            -ds_ratio: downsample ratio. Plots every nth datapoint for all channels
            -alpha: transparency value for shading regions on graph
    """

    if wrist is not None:
        wrist_idx = {'ax': wrist.get_signal_index("Accelerometer x"),
                     'ay': wrist.get_signal_index("Accelerometer y"),
                     'az': wrist.get_signal_index("Accelerometer z"),
                     'gx': wrist.get_signal_index("Gyroscope x"),
                     'gy': wrist.get_signal_index("Gyroscope y"),
                     'gz': wrist.get_signal_index("Gyroscope z"),
                     'temp': wrist.get_signal_index("Temperature")}

    if ankle is not None:
        ankle_idx = {'ax': ankle.get_signal_index("Accelerometer x"),
                     'ay': ankle.get_signal_index("Accelerometer y"),
                     'az': ankle.get_signal_index("Accelerometer z"),
                     'gx': ankle.get_signal_index("Gyroscope x"),
                     'gy': ankle.get_signal_index("Gyroscope y"),
                     'gz': ankle.get_signal_index("Gyroscope z"),
                     'temp': ankle.get_signal_index("Temperature")}

    if ecg is not None:
        ecg_idx = {'ecg': ecg.ecg.get_signal_index("ECG"),
                   'temp': ecg.ecg.get_signal_index("Temperature")}

    """ --------- Plot set-up --------"""
    n_subplots = sum([int(i is not None) for i in [df_epoch, ecg, wrist, ankle, df_hr, df_posture]])

    if wrist is not None or ankle is not None:  # for temperature data
        n_subplots += 1

    if "cadence" in df_epoch.columns:
        n_subplots += 1

    gs = np.ones(n_subplots)

    fig, ax = plt.subplots(n_subplots, 1, sharex='col', figsize=(12, 8), gridspec_kw={'height_ratios': gs})

    plt.suptitle(f"{study_code}_{subj}")

    curr_plot = 0

    already_wrist_nw = False
    already_ankle_nw = False

    """------- Plots activity counts with cutpoints --------"""
    if df_epoch is not None and df_epoch.shape[0] >= 2:
        epoch_len = int((df_epoch.iloc[1]['start_time'] - df_epoch.iloc[0]['start_time']).total_seconds())
        ax[curr_plot].plot(df_epoch['start_time'], df_epoch['avm'], color='black', label=f'{epoch_len}s wrist', zorder=0)
        ax[curr_plot].set_ylabel("Wrist AVM (mG)")
        ax[curr_plot].axhline(0, color='grey', linestyle='dashed')

        if wrist is None and wrist_nw is not None:

            for row in wrist_nw.itertuples():
                if row.Index == 0:
                    ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha, linewidth=0,
                                          color='grey', label='Nonwear')
                ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha, linewidth=0,
                                      color='grey')

            if wrist is None:
                if df_sptw is not None and shade_sleep_windows:

                    for row in df_sptw.itertuples():
                        if row.Index == 0:
                            ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha, linewidth=0,
                                                  color='orchid', label='sptw')
                        # if row.Index != 0:
                        ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time,
                                              alpha=alpha if row.Index > 1 else 0, linewidth=0,
                                              color='orchid')

                if df_sleep is not None and shade_sleep_windows:

                    for row in df_sleep.itertuples():
                        if row.Index == 0:
                            ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha, linewidth=0,
                                                  color='purple', label='sleep')
                        # if row.Index != 0:
                        ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time,
                                              alpha=alpha if row.Index > 1 else 0, linewidth=0,
                                              color='purple')

                if (df_sptw is not None and shade_sleep_windows) or (
                        df_sleep is not None and shade_sleep_windows) or wrist_nw is not None:
                    ax[curr_plot].legend(loc='upper right')

            already_wrist_nw = True

        try:
            cp = cutpoints["{}{}".format(author.capitalize(), "Dominant" if dominant else "Non-dominant")]

            ax[curr_plot].axhline(cp[0], color='limegreen', linestyle='dashed', label=f'{author} light')
            ax[curr_plot].axhline(cp[1], color='orange', linestyle='dashed', label=f'{author} mvpa')

        except (KeyError, AttributeError):
            pass

        if show_activity_log:

            c = {'yes': 'lightskyblue', 'no': 'pink', 'unknown': 'darkorchid'}
            for row in df_act_log.itertuples():
                try:
                    df_activity = df_epoch.loc[(df_epoch['start_time'] >= row.start_time) &
                                               (df_epoch['start_time'] <= row.start_time + timedelta(minutes=row.duration))]
                    max_val = df_activity['avm'].max()

                    ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.start_time + timedelta(minutes=row.duration),
                                          alpha=alpha, linewidth=0, color=c[row.active])
                    ax[curr_plot].axvline(x=row.start_time, alpha=alpha, linewidth=.75, color=c[row.active])
                    ax[curr_plot].axvline(x=row.start_time + timedelta(minutes=row.duration),
                                          alpha=alpha, linewidth=.75, color=c[row.active])

                    ax[curr_plot].text(x=row.start_time + timedelta(minutes=row.duration/2),
                                       y=max_val * 1.1 if row.Index % 2 == 0 else max_val * 1.2,
                                       s=row.Index, color='red')

                except (TypeError, KeyError):
                    try:
                        ax[curr_plot].axvline(x=row.start_time, color='grey')
                        ax[curr_plot].text(x=row.start_time, y=max_val * 1.1 if row.Index % 2 == 0 else max_val * 1.2,
                                           s=row.Index, color='red')

                    except:
                        pass

        if intensity_markers:
            df_light = df_epoch.loc[df_epoch['intensity'] == 'light']
            df_mod = df_epoch.loc[df_epoch['intensity'] == 'moderate']

            ax[curr_plot].scatter(df_light['start_time'], df_light['avm'], color='limegreen', s=15, label='light', zorder=1)
            ax[curr_plot].scatter(df_mod['start_time'], df_mod['avm'], color='orange', s=15, label='mod', zorder=1)

        ax[curr_plot].legend(loc='lower right')

        curr_plot += 1

    """-------- Raw wrist data ----------"""
    if wrist is not None:
        wrist_prefix = "a" if not wrist_gyro else 'g'

        if not highpass_accel:
            ax[curr_plot].plot(wrist.ts[::ds_ratio], wrist.signals[wrist_idx[f'{wrist_prefix}x']][::ds_ratio],
                               color='black', label="{}Hz wrist".format(round(wrist.signal_headers[0]['sample_rate']/ds_ratio, 1)))
            ax[curr_plot].plot(wrist.ts[::ds_ratio], wrist.signals[wrist_idx[f'{wrist_prefix}y']][::ds_ratio],
                               color='red')
            ax[curr_plot].plot(wrist.ts[::ds_ratio], wrist.signals[wrist_idx[f'{wrist_prefix}z']][::ds_ratio],
                               color='dodgerblue')

        if highpass_accel:
            x = filter_signal(data=wrist.signals[wrist_idx['ax']],
                              sample_f=wrist.signal_headers[0]["sample_rate"],
                              filter_type='highpass', filter_order=3, high_f=.1)
            y = filter_signal(data=wrist.signals[wrist_idx['ay']],
                              sample_f=wrist.signal_headers[0]["sample_rate"],
                              filter_type='highpass', filter_order=3, high_f=.1)
            z = filter_signal(data=wrist.signals[wrist_idx['az']],
                              sample_f=wrist.signal_headers[0]["sample_rate"],
                              filter_type='highpass', filter_order=3, high_f=.1)

            ax[curr_plot].plot(wrist.ts[::ds_ratio], x[::ds_ratio], color='black',
                               label="{}Hz wrist".format(round(wrist.signal_headers[0]['sample_rate']/ds_ratio, 1)))
            ax[curr_plot].plot(wrist.ts[::ds_ratio], y[::ds_ratio], color='red')
            ax[curr_plot].plot(wrist.ts[::ds_ratio], z[::ds_ratio], color='dodgerblue')

        if wrist_nw is not None and not already_wrist_nw:

            for row in wrist_nw.itertuples():
                if row.Index == 0:
                    ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha, linewidth=0,
                                          color='grey', label='Nonwear')
                ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha, linewidth=0,
                                      color='grey')

        if df_sptw is not None and shade_sleep_windows:

            for row in df_sptw.itertuples():
                if row.Index == 0:
                    ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha, linewidth=0,
                                          color='orchid', label='sptw')
                # if row.Index != 0:
                ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha if row.Index > 1 else 0, linewidth=0,
                                      color='orchid')

        if df_sleep is not None and shade_sleep_windows:

            for row in df_sleep.itertuples():
                if row.Index == 0:
                    ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha, linewidth=0,
                                          color='purple', label='sleep')
                # if row.Index != 0:
                ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha if row.Index > 1 else 0, linewidth=0,
                                      color='purple')

        if (df_sptw is not None and shade_sleep_windows) or (df_sleep is not None and shade_sleep_windows) or wrist_nw is not None:
            ax[curr_plot].legend(loc='upper right')

        ax[curr_plot].legend(loc='lower right')
        ax[curr_plot].set_ylabel("Wrist Acc (G)" if not wrist_gyro else 'Wrist gyro (deg/sec)')

        curr_plot += 1

    """--------- Raw ankle data ---------"""
    if ankle is not None:
        ankle_prefix = "a" if not ankle_gyro else 'g'

        if not highpass_accel:
            ax[curr_plot].plot(ankle.ts[::ds_ratio], ankle.signals[ankle_idx[f'{ankle_prefix}x']][::ds_ratio],
                               color='black',
                               label="{}Hz ankle".format(round(ankle.signal_headers[0]['sample_rate']/ds_ratio, 1)))
            ax[curr_plot].plot(ankle.ts[::ds_ratio], ankle.signals[ankle_idx[f'{ankle_prefix}y']][::ds_ratio],
                               color='red')
            ax[curr_plot].plot(ankle.ts[::ds_ratio], ankle.signals[ankle_idx[f'{ankle_prefix}z']][::ds_ratio],
                               color='dodgerblue')

        if highpass_accel:
            x = filter_signal(data=ankle.signals[ankle_idx['ax']],
                              sample_f=ankle.signal_headers[0]["sample_rate"],
                              filter_type='highpass', filter_order=3, high_f=.1)
            y = filter_signal(data=ankle.signals[ankle_idx['ay']],
                              sample_f=ankle.signal_headers[0]["sample_rate"],
                              filter_type='highpass', filter_order=3, high_f=.1)
            z = filter_signal(data=ankle.signals[ankle_idx['az']],
                              sample_f=ankle.signal_headers[0]["sample_rate"],
                              filter_type='highpass', filter_order=3, high_f=.1)

            ax[curr_plot].plot(ankle.ts[::ds_ratio], x[::ds_ratio], color='black',
                               label="{}Hz ankle".format(round(ankle.signal_headers[0]['sample_rate']/ds_ratio, 1)))
            ax[curr_plot].plot(ankle.ts[::ds_ratio], y[::ds_ratio], color='red')
            ax[curr_plot].plot(ankle.ts[::ds_ratio], z[::ds_ratio], color='dodgerblue')

        if df_gait is not None and shade_gait_bouts:
            df_gait = df_gait.loc[df_gait['duration'] >= min_gait_dur].reset_index(drop=True)

            for row in df_gait.itertuples():
                if row.Index == 0:
                    ax[curr_plot].axvspan(xmin=row.start_timestamp, xmax=row.end_timestamp, alpha=alpha, linewidth=0,
                                          color='gold', label='GaitBouts')
                if row.Index != 0:
                    ax[curr_plot].axvspan(xmin=row.start_timestamp, xmax=row.end_timestamp, alpha=alpha, linewidth=0,
                                          color='gold')

        if ankle_nw is not None:
            for row in ankle_nw.itertuples():
                if row.Index == 0:
                    ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha, linewidth=0,
                                          color='grey', label='Nonwear')

                ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha, linewidth=0,
                                      color='grey')

            already_ankle_nw = True

        if df_ankle_sync is not None:
            for row in df_ankle_sync.itertuples():
                ax[curr_plot].axvspan(row.start_time, row.end_time, 0, 1, color='fuchsia', alpha=alpha)

                if row.Index != df_ankle_sync.index[0]:
                    ax[curr_plot].axvline(row.start_time, color='fuchsia', linestyle='dashed', lw=1.5)
                if row.Index == df_ankle_sync.index[0]:
                    ax[curr_plot].axvline(row.start_time, color='fuchsia', linestyle='dashed', lw=1.5, label='sync')

        if mark_steps and df_steps is not None:

            if bout_steps_only and df_gait is not None:
                steps = pd.DataFrame(columns=['step_time', 'step_idx'])
                gait_bouts = df_gait.loc[df_gait['duration'] >= min_gait_dur]

                for bout in gait_bouts.itertuples():
                    s = df_steps.loc[(df_steps['step_time'] >= bout.start_timestamp) &
                                     (df_steps['step_time'] <= bout.end_timestamp)]
                    steps = pd.concat(objs=[steps, s])

                ax[curr_plot].scatter(steps['step_time'],
                                      [ankle.signals[ankle_idx['ay' if not ankle_gyro else 'gz']][i] +
                                       2 if not ankle_gyro else 200 for i in steps['step_idx']],
                                      marker='v', color='limegreen', label='steps', s=25, zorder=1)

            if not bout_steps_only:

                ax[curr_plot].scatter(df_steps['step_time'],
                                      [ankle.signals[ankle_idx['ay' if not ankle_gyro else 'gz']][i] +
                                       2 if not ankle_gyro else 200 for i in df_steps['step_idx']],
                                      marker='v', color='limegreen', label='steps', s=25, zorder=1)

        if ankle_nw is not None or (df_gait is not None and shade_gait_bouts) or mark_steps:
            ax[curr_plot].legend(loc='lower right')

        ax[curr_plot].set_ylabel("Ankle acc (G)" if not ankle_gyro else 'Ankle gyro (deg/sec)')

        curr_plot += 1

    if 'cadence' in df_epoch.columns:
        ax[curr_plot].plot(df_epoch['start_time'], df_epoch['cadence'], color='black')
        ax[curr_plot].set_ylabel("Epoched Cadence")

        if not already_ankle_nw:
            for row in ankle_nw.itertuples():
                if row.Index == 0:
                    ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha, linewidth=0,
                                          color='grey', label='Nonwear')

                ax[curr_plot].axvspan(xmin=row.start_time, xmax=row.end_time, alpha=alpha, linewidth=0,
                                      color='grey')

        curr_plot += 1

    """ -------- temperature data ---------"""
    if wrist is not None:
        ax[curr_plot].plot(wrist.temp_ts, wrist.signals[wrist_idx['temp']], color='black', label='wrist')
    if ankle is not None:
        ax[curr_plot].plot(ankle.temp_ts, ankle.signals[ankle_idx['temp']], color='red', label='ankle')
    if ecg is not None:
        ax[curr_plot].plot(ecg.temp_ts, ecg.ecg.signals[ecg_idx['temp']], color='dodgerblue', label='ECG')

    if ankle is not None or wrist is not None or ecg is not None:
        ax[curr_plot].legend(loc='lower right')
        ax[curr_plot].set_ylabel("Temp. (deg. C)")
        curr_plot += 1

    # ECG-based data
    if ecg is not None:

        ax[curr_plot].plot(ecg.ts[::ds_ratio], ecg.filt[::ds_ratio], color='red', label='Filt. ECG')
        ax[curr_plot].legend(loc='lower right')
        ax[curr_plot].set_ylabel("Voltage")
        curr_plot += 1

    if df_chest_sync is not None:
        for row in df_chest_sync.itertuples():
            if row.Index != df_chest_sync.index[0]:
                ax[curr_plot].axvline(row.start_time, color='fuchsia', linestyle='dashed', lw=3)
            if row.Index == df_chest_sync.index[0]:
                ax[curr_plot].axvline(row.start_time, color='fuchsia', linestyle='dashed', lw=3, label='sync')

    if df_hr is not None:
        ax[curr_plot].plot(df_hr['timestamp'], df_hr['hr'], color='red')
        ax[curr_plot].set_ylabel("HR (bpm)")

        df_hr = df_hr.loc[[not np.isnan(i) for i in df_hr['hr']]]
        ax[curr_plot].set_yticks(np.arange(50, 25*np.ceil(max(df_hr['hr'])/25)+1, 25))
        ax[curr_plot].grid()

    ax[-1].xaxis.set_major_formatter(xfmt_raw)
    plt.tight_layout()
    plt.subplots_adjust(hspace=.05, top=.955, bottom=.06)

    return fig


def plot_gait_histogram(df_gait, col_name='duration', bins=None):

    fig, ax = plt.subplots(1, figsize=(10, 6))

    df_gait[col_name].plot.hist(bins=bins, ax=ax, color='grey', edgecolor='black')
    ax.set_xlabel(col_name)
    plt.tight_layout()

    return fig


def plot_activity(df_epoch, df_sleep, ankle_nw, wrist_nw):

    fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8))

    intensity_dict = {"sedentary": 0, 'light': 1, 'moderate': 2}

    ax[0].plot(df_epoch['start_time'], df_epoch['avm'], color='black')
    ax[0].set_ylim(0, )

    wrist_data = [intensity_dict[row.intensity] for row in df_epoch.itertuples()]
    ax[1].plot(df_epoch['start_time'], wrist_data, color='black')

    for row in df_sleep.itertuples():
        ax[1].axvspan(xmin=row.start_time, xmax=row.end_time, ymin=0, ymax=.5, color='purple', alpha=.25)
    for row in wrist_nw.itertuples():
        for subplot in range(2):
            ax[subplot].axvspan(xmin=row.start_time, xmax=row.end_time, ymin=0, ymax=.5, color='darkgrey', alpha=.75)
    for day in df_epoch['Day_Num'].unique():
        df_day = df_epoch.loc[df_epoch['Day_Num'] == day]
        for subplot in range(3):
            ax[subplot].axvspan(xmin=df_day.iloc[0]['Timestamp'], xmax=df_day.iloc[-1]['Timestamp'], ymin=.5, ymax=1,
                                color='dodgerblue' if day % 2 == 0 else 'red', alpha=.25)

    ax[1].set_ylim(0, 2)
    ax[1].set_yticks([0, 1, 2])
    ax[1].set_yticklabels(['sed', 'light', 'mod'])

    ax[2].plot(df_epoch['start_time'], df_epoch['cadence'], color='black')
    for row in ankle_nw.itertuples():
        ax[2].axvspan(xmin=row.start_time, xmax=row.end_time, ymin=0, ymax=.5, color='darkgrey', alpha=.75)
    ax[2].set_ylim(0, )
    ax[2].set_ylabel("Cadence")

    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S"))

    plt.tight_layout()

    return fig


def daily_cadence_description(df_gait, min_duration=15, plot=True):

    df = df_gait.loc[df_gait['duration'] >= min_duration]
    df['day'] = [i.date() for i in df['start_timestamp']]

    df_g = df.groupby('day')

    df_desc = df_g['cadence'].describe()[['count', 'mean', 'std', '50%']]
    df_desc_all = df['cadence'].describe()[['count', 'mean', 'std', '50%']]

    if plot:
        fig, ax = plt.subplots(len(df_g.groups), figsize=(8, 9), sharex='col')

        for i, group in enumerate(df_g.groups):
            df_g.get_group(group)['cadence'].plot.density(ax=ax[i],
                                                          color='black', label=f'n={len(df_g.get_group(group))}')
            ax[i].axvline(df_desc.loc[group]['50%'], color='red',
                          label=f"50%ile (day) = {df_desc.loc[group]['50%']:.1f}", linestyle='dashed')
            ax[i].axvline(df_desc_all['50%'], color='red',
                          label=f"50%ile (all) = {df_desc_all['50%']:.1f}")
            ax[i].set_ylabel(group)
            ax[i].set_yticks([])
            ax[i].set_ylim(0, )
            ax[i].legend(loc='lower right')
        ax[-1].set_xlim(25, 200)
        ax[-1].set_xlabel("Cadence (spm)")
        plt.tight_layout()
        plt.subplots_adjust(hspace=.025)

