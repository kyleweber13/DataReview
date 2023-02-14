import pandas as pd
import scipy.fft
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


def get_index_from_stamp(start, stamp, sample_rate):

    idx = (stamp - start).total_seconds() * sample_rate

    return int(idx)


def run_fft(data, sample_rate, remove_dc=True, highpass=None, show_plot=True):
    from Analysis import filter_signal

    if remove_dc:
        data_f = filter_signal(data=data, filter_type='highpass', filter_order=5, sample_f=sample_rate, high_f=.1)
    if highpass is not None:
        data_f = filter_signal(data=data, filter_type='highpass', filter_order=5, sample_f=sample_rate, high_f=highpass)

    if not remove_dc and highpass is None:
        data_f = data

    fft_raw = scipy.fft.fft(data_f)
    L = len(data_f)
    xf = np.linspace(0.0, 1.0 / (2.0 * (1 / sample_rate)), L // 2)

    fig = None

    if show_plot:
        fig, ax = plt.subplots(3 if remove_dc or highpass is not None else 2, figsize=(10, 6))
        plt.subplots_adjust(hspace=.3)

        ax[0].plot(np.arange(len(data))/sample_rate, data, color='red', label='Raw')
        ax[0].set_xlabel("Seconds")
        ax[0].legend()

        if remove_dc or highpass is not None:
            ax[1].plot(np.arange(len(data_f))/sample_rate, data_f, color='dodgerblue', label='Filtered')
            ax[1].set_xlabel("Seconds")
            ax[1].legend()

            ax[2].plot(xf, 2.0 / L / 2 * np.abs(fft_raw[0:L // 2]), color='red')
            ax[2].set_xlabel("Hz")
            ax[2].set_ylabel("Power")

        if not remove_dc and highpass is None:
            ax[1].plot(xf, 2.0 / L / 2 * np.abs(fft_raw[0:L // 2]), color='red')
            ax[1].set_xlabel("Hz")
            ax[1].set_ylabel("Power")

        plt.tight_layout()

    df_fft = pd.DataFrame({"freq": xf, "power": 2.0 / L / 2 * np.abs(fft_raw[0:L // 2])})
    power_sum = df_fft['power'].sum()
    df_fft['power_norm'] = df_fft['power'] / power_sum

    return fig, df_fft


def plot_stft(data, sample_rate, nperseg_multiplier=5, plot_data=True):

    freq_res = 1 / (sample_rate * nperseg_multiplier / sample_rate)

    f, t, Zxx = scipy.signal.stft(x=data, fs=sample_rate,
                                  nperseg=sample_rate * nperseg_multiplier, window='hamming')

    if plot_data:
        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 6))
        ax1.set_title("Data")
        ax2.set_title("STFT: {}-second and {} Hz resolution".format(nperseg_multiplier, round(freq_res, 5)))

        ax1.plot(np.arange(0, len(data)) / sample_rate, data, color='black')

        pcm = ax2.pcolormesh(t, f, np.abs(Zxx), cmap='turbo', shading='auto')

        cbaxes = fig.add_axes([.91, .11, .03, .35])
        cb = fig.colorbar(pcm, ax=ax2, cax=cbaxes)

        ax2.set_ylabel('Frequency [Hz]')
        ax2.set_xlabel('Seconds')

    return fig, f, t, Zxx


