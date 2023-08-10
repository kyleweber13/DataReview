import os
import matplotlib.pyplot as plt
import nimbalwear.activity
from DataImport import import_data, check_filenames
from Analysis import *
from Plotting import *
import pandas as pd
import Run_FFT
import neurokit2 as nk

if __name__ == "__main__":
    subjs = ['SBH0228']
    study_code = 'OND09'  # 'OND09'
    site_code = ''
    visit_num = '01'
    subj = subjs[0]
    cutpoint_age_cutoff = 60

    # root_dir = 'X:/CPSR/nimbalwear/'
    root_dir = "W:/prd/nimbalwear/"
    # root_dir = 'W:/NiMBaLWEAR/'

    # old_dir = 'X:/CPSR/nimbalwear/'
    old_dir = "W:/prd/nimbalwear/"

    edf_folder = f"{root_dir}{study_code}/wearables/device_edf_cropped/"

    file_dict = {
                 'clin_insights_file': "W:/OND09 (HANDDS-ONT)/handds_clinical_insights_new.xlsx",
                 'gait_file': root_dir + study_code + "/analytics/gait/bouts/{}_{}{}_{}_GAIT_BOUTS.csv",
                 'steps_file': root_dir + study_code + "/analytics/gait/steps/{}_{}{}_{}_GAIT_STEPS.csv",
                 'activity_log_file': "W:/OND09 (HANDDS-ONT)/Digitized logs/handds_activity_log.xlsx",
                 'sptw_file':  root_dir + f"{study_code}/analytics/sleep/sptw/{study_code}_{subj}_{visit_num}_SPTW.csv",
                 'sleep_bout_file': root_dir + f"{study_code}/analytics/sleep/bouts/{study_code}_{subj}_{visit_num}_SLEEP_BOUTS.csv",
                 'epoch_folder': root_dir + study_code + "/analytics/activity/epochs/",
                 'epoch_file': root_dir + f"{study_code}/analytics/activity/epochs/{study_code}_{site_code}{subj}_{visit_num}_ACTIVITY_EPOCHS.csv",
                 'edf_folder': edf_folder,
                 'cn_beat_file': f"W:/NiMBaLWEAR/OND09/analytics/ecg/CardiacNavigator/{study_code}_{subj}_Beats.csv",
                 'posture': root_dir + study_code + "/analytics/activity/posture/OND09_{}{}_{}_posture_bouts.csv",
                 'sync': f"{root_dir}{study_code}/analytics/sync/events/",
                 'devices': f"{old_dir}{study_code}/study/devices.csv",  # pipeline
                 'collections': f"{old_dir}{study_code}/study/collections.csv",  # pipeline
    }

    file_dict = check_filenames(file_dict=file_dict, subj=subj,
                                site_code='WTL',  # 'SBH'
                                imu_code='GNOR' if study_code == 'OND07' else 'AXV6',
                                nw_bouts_folder=f"{root_dir}{study_code}/analytics/nonwear/bouts_standard/",
                                data_review_df_folder="W:/OND09 (HANDDS-ONT)/Data Review/", visit_num=visit_num)

    cutpoints = {"PowellDominant": [51*1000/30/15, 68*1000/30/15, 142*1000/30/15],
                 "PowellNon-dominant": [47*1000/30/15, 64*1000/30/15, 157*1000/30/15],
                 'FraysseDominant': [62.5, 92.5, 10000],
                 'FraysseNon-dominant': [42.5, 98, 10000]}

    df_epoch, epoch_len, df_clin, df_posture, \
    df_act, df_sptw, df_sleep, df_gait, \
    df_steps, df_act_log, ankle_nw, wrist_nw, \
    ankle_sync, chest_sync, ankle, wrist = import_data(file_dict=file_dict, subj=subj, visit_num=visit_num,
                                                       site_code=site_code, study_code=study_code,
                                                       load_raw=False)

    df_sleep = df_sleep.loc[df_sleep['bout_detect'] == 't8a4']
    df_sleep.reset_index(drop=True, inplace=True)

    df_epoch.sort_values('start_time', inplace=True)

    try:
        start_date = df_epoch.loc[0]['Day']
        df_epoch.insert(loc=3, column='Day_Num', value=[(row.Day - start_date).days + 1 for row in df_epoch.itertuples()])
    except KeyError:
        pass

    try:
        end_timestamp = min([wrist.ts[-1], ankle.ts[-1]]).date()
    except AttributeError:
        end_timestamp = None

    hand_dom, df_coll_subj = calculate_hand_dom2(subj=subj,
                                                 colls_csv_file=file_dict['collections'],
                                                 devices_csv_file=file_dict['devices'])

    age = df_coll_subj.iloc[0]['age']
    use_author = 'Fraysse' if age >= cutpoint_age_cutoff else 'Powell'

    df_epoch['cadence'] = epoch_cadence(epoch_timestamps=df_epoch['start_time'], df_steps=df_steps)

    df_epoch['intensity'] = epoch_intensity(df_epoch=df_epoch, column='avm', cutpoints_dict=cutpoints,
                                            cutpoint_key='Dominant' if hand_dom else 'Non-dominant', author=use_author)
    df_epoch['sleep_mask'] = flag_sleep_epochs(df_epoch=df_epoch, df_sleep_alg=df_sleep)
    df_epoch['nw_mask'] = flag_nonwear_epochs(df_epoch=df_epoch, df_nw=wrist_nw)

    df_daily = combine_df_daily(ignore_sleep=True, ignore_nw=True, df_epoch=df_epoch, cutpoints=cutpoints,
                                df_gait=df_gait, epoch_len=15, hand_dom=hand_dom, end_time=end_timestamp, min_dur=1440)

    print()
    walk_avm_str = print_walking_intensity_summary(df_gait=df_gait, df_epoch=df_epoch, min_dur=30,
                                                   cutpoints=cutpoints[use_author + "Dominant" if hand_dom else
                                                   use_author + 'Non-dominant'])

    # gen_relationship_graph(daily_df=df_daily, df_gait=df_gait, author=use_author)

    # summary_plot(df_daily=df_daily, author=use_author, subj=subj, df_sleep=df_sleep)
    df_coll_summary = collection_summary(df_daily=df_daily, wrist_obj=wrist, ankle_obj=ankle, chest_obj=None)

    print("")
    fig = plot_raw(subj=subj, wrist=wrist, ankle=ankle, ecg=None,
                   highpass_accel=False, intensity_markers=True,
                   cutpoints=cutpoints, dominant=hand_dom, author=use_author, wrist_nw=wrist_nw, wrist_gyro=False,
                   ankle_gyro=True, ankle_nw=ankle_nw,
                   df_ankle_sync=ankle_sync, df_chest_sync=None,
                   df_epoch=df_epoch,
                   df_sleep=df_sleep, df_sptw=df_sptw, shade_sleep_windows=True,
                   df_gait=df_gait, df_act_log=df_act_log,
                   df_hr=None, df_posture=None, df_steps=df_steps,
                   shade_gait_bouts=True, min_gait_dur=15, mark_steps=True, bout_steps_only=True,
                   show_activity_log=True,
                   alpha=.35, ds_ratio=3)

    df_act_log = calculated_logged_intensity(epoch_len=15, df_epoch=df_epoch, df_act_log=df_act_log,
                                             hours_offset=0, df_steps=df_steps, quiet=True)

    print_activity_log(df_act_log)

    print_quick_summary(subj=subj, df_clin=df_clin, df_daily=df_daily, walk_avm_str=walk_avm_str,
                        cutpoint_author=use_author, dominant=hand_dom, df_gaitbouts=df_gait, df_epoch=df_epoch)

    print_medical_summary(df_clin=df_clin)

    # compare_cutpoints(df_daily=df_daily)

    # fft_fig, df_fft, dom_f, fft_idx = freq_analysis(obj=wrist, subj=subj, channel='Accelerometer x', ts='2023-02-06 6:44:00', lowpass=None, highpass=.1, sample_rate=None, n_secs=600, stft_mult=5, stft=False, show_plot=True)

    # calculate_window_cadence(start='2023-01-28 14:34:40', stop=timedelta(minutes=5), df_steps=df_steps, ankle_obj=ankle, show_plot=True, axis="Gyroscope y")
    # check_intensity_window(df_epoch=df_epoch, start=df_act_log.loc[5]['start_time'], end=df_act_log.loc[5]['start_time'] + timedelta(minutes=df_act_log.loc[5]['duration']))

    # calculate_logged_intensity_individual(start_timestamp='2023-01-24 12:16:00', end_timestamp='2023-01-24 12:45:25', df_epoch=df_epoch, epoch_len=15)

    # daily_cadence_description(df_gait=df_gait, min_duration=15, plot=True)
