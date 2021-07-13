from IPython.display import display, Markdown, Latex, HTML
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import _pickle as cPickle


def load(stage='raw', reapply=False, input_path='../Data', input_name='oasis3'):
    """[Returns the processed data for the Oasis-3 project.]
    For description of the different fields see https://naccdata.org/data-collection/forms-documentation/uds-2
    Parameters
    ----------
    stage : str, optional
        The type of data to return can be one of the following: 'raw','clean','bmi', by default 'raw'
        raw : downloads the raw data to a local folder and sotres it using the pkl format
        clean : removes empty features and includes a numeric column of time since admission 
                across the following datasets:
                "ADRC_ADRCCLINICALDATA" 
                'CNDA_PSYCHOMETRICSDATA',
                'FS_FSDATA',
                'UDS_A1SUBDEMODATA',
                'UDS_A5SUBHSTDATA',
                'UDS_B2HACHDATA',
                'UDS_B3UPDRSDATA',
                'UDS_B5BEHAVASDATA',
                'UDS_B6BEVGDSDATA',
                "subjects"
        bmi: extracts BMI measures from the "ADRC_ADRCCLINICALDATA" dataset 
    reapply : bool, optional
        [if True performs the preprocessing again otherwise it will
                just load a local pickle file stored in the last time the
                preprocessing was performed], by default False
    input_path : str, optional
        [defaults to '../Data' where to load and store intermediate files], by default '../Data'
    input_name : str, optional
        [defaults to 'oasis3' what suffix identifier to give the local
                pickles to simplify deletion.], by default 'oasis3'

    Returns
    -------
    Processed data per stage of preprocessing

    Raises
    ------
    Exception
        If a stage was not defined 
    """

    input_file = f'{input_path}/processed/{stage}_{input_name}.pkl'

    if not reapply and Path(input_file).exists():
        data = load_pickle(input_file)
    else:
        pipeline = {'raw':       download_data,
                    'clean':     clean_data,
                    'bmi':       process_bmi,
                    'thickness': tidy_freesurfer_thickness,
                    'volume':    tidy_freesurfer_volume}

        if stage in pipeline:
            data = pipeline[stage](input_file)
        else:
            raise Exception(f"Sorry, {stage} is not a valid pipeline stage")

    return data


def download_data(input_file):
    """[Download the summary tables for the oasis 3 dataset]

    Parameters
    ----------
    input_file : [str]
        [The local complete path to store the pkl file containing the raw dataset in a dictionary of pandas dataframes]

    Returns
    -------
    [dict]
        [raw datasets in pandas DataFrame format]
    """    
    data = {}
    for d in ["ADRC_ADRCCLINICALDATA",
            'CNDA_PSYCHOMETRICSDATA',
            'FS_FSDATA',
            'UDS_A1SUBDEMODATA',
            'UDS_A5SUBHSTDATA',
            'UDS_B2HACHDATA',
            'UDS_B3UPDRSDATA',
            'UDS_B5BEHAVASDATA',
            'UDS_B6BEVGDSDATA',
            "subjects"]:
        url = f'https://raw.githubusercontent.com/esoreq/Real_site/master/data/{d}.csv'
        data[d] = pd.read_csv(url)
    save_pickle(input_file, data)
    return data


def missing_profile(x):
    d = {}
    d['notnull'] = x.notnull().sum()
    d['isnull'] = x.isnull().sum()
    d['%missing'] = d['isnull']/x.shape[0]
    return pd.Series(d, index=d.keys())


def drop_missing_columns(df, thr):
    _df = df.apply(missing_profile).T
    columns_2_drop = _df[_df['%missing'] > thr]
    if not columns_2_drop.empty:
        df = df.drop(columns=columns_2_drop.index)
    return df, columns_2_drop


def clean_data(output_file, thr=0.9):
    data = load('raw')
    for k in data.keys():
        data[k] = drop_missing_columns(data[k], thr)
        data[k] = days_since_entry(data[k])
    save_pickle(output_file, data)
    return data


def days_since_entry(df):
    if len(df.iloc[0, 0].split('_')) > 1:
        df.insert(1, 'days_since_entry', df.iloc[:, 0].apply(
            lambda x: int(x.split('_')[-1][1:])))
    return df


def process_bmi(output_file):
    # load cleaned data
    df = load('clean')["ADRC_ADRCCLINICALDATA"]  # extract only task-relevant columns
    df = df[['Subject',
             'days_since_entry',
             'ageAtEntry',
             'height','weight']]
    # rename columns for simplicity
    # extract days_since_entry
    df['age'] = df['days_since_entry']/365.2425 + df['ageAtEntry']
    # transform from Imperial to metric
    df['height'] = df['height']*0.0254  # transfrom from inch to meters
    df['weight'] = df['weight']*0.453592  # transfrom from pounds to kg
    # calculate bmi
    df['bmi'] = df['weight']/(df['height']**2)
    # categorize bmi
    bins = [0, 18.5, 25, 30, 60]
    labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['bmi_cat'] = pd.cut(df['bmi'], bins, labels=labels)
    save_pickle(output_file, df)
    return df

def tidy_freesurfer_volume(output_file):
    df = load('clean')["FS_FSDATA"]  # extract only freesurfer data
    _df = df.loc[:, df.columns[df.columns.str.contains('_volume')]].div(
        df.IntraCranialVol, axis=0)
    _df = pd.concat([df[['Subject', 'days_since_entry', 'M/F',
                         'Hand', 'Race', 'Ethnicity']], _df], axis=1)
    save_pickle(output_file, _df)
    return _df

def tidy_freesurfer_thickness(output_file):
    df = load('clean')["FS_FSDATA"]  # extract only freesurfer data
    
    save_pickle(output_file, df)
    return df


def get_parent(filename):
    return Path(filename).absolute().parent


def save_pickle(output_name, data):
    if not Path(output_name).exists():
        Path(get_parent(output_name)).mkdir(parents=True, exist_ok=True)
    output_pickle = open(output_name, "wb")
    cPickle.dump(data, output_pickle)
    output_pickle.close()


def load_pickle(input_name):
    input_pickle = open(input_name, 'rb')
    data = cPickle.load(input_pickle)
    input_pickle.close()
    return data
