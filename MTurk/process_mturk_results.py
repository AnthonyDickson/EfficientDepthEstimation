from pathlib import Path
from urllib.parse import urlsplit

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plac
from scipy.stats import norm


def process_raw_data(csv_path):
    data = pd.read_csv(csv_path)

    rating_categories = ['Bad', 'Poor', 'Fair', 'Good', 'Excellent']
    rating_categorical = pd.Categorical(data['Answer.rating.label'], ordered=True, categories=rating_categories)

    data['Rating'] = rating_categorical
    data['Score'] = rating_categorical.codes + 1

    video_urls = data['Input.video_url'].map(lambda url: urlsplit(url).path)
    video_urls = video_urls.str.split(pat='/', expand=True)
    # url path is assumed to be in the format `/<model name>/<video name>.mp4
    model_names = video_urls[1]
    model_names = model_names.str.replace("reside", "hu")
    data['Model'] = model_names

    video_names = video_urls[2]
    frame_indices = video_names.map(lambda path: Path(path).stem)
    frame_indices = frame_indices.astype(int)
    data['Frame'] = frame_indices

    columns = ['WorkerId', 'WorkTimeInSeconds', 'Model', 'Frame', 'Rating', 'Score']
    data = data[columns]

    return data


def reject_workers(data, questionnaire_csv_paths):
    ids_from_questionnaire = set()

    for path in questionnaire_csv_paths:
        df = pd.read_csv(path)
        ids_from_questionnaire = ids_from_questionnaire.union(df['WorkerId'])

    num_tasks = data['WorkerId'].value_counts()
    worker_stats_std = data.groupby(['WorkerId']).std().sort_index()
    worker_stats_std['NumTasks'] = num_tasks
    worker_stats_mean = data.groupby(['WorkerId']).mean().sort_index()
    worker_stats_mean['NumTasks'] = num_tasks

    not_enough_answers = num_tasks < 180
    too_fast = worker_stats_mean['WorkTimeInSeconds'] < 5
    too_many_same_answers = (worker_stats_std['Score'] == 0.0) & (worker_stats_std['NumTasks'] > 5)

    rejection = pd.DataFrame()
    rejection['too_fast'] = too_fast
    rejection['not_enough_answers'] = not_enough_answers
    rejection['all_same_answers'] = too_many_same_answers

    rejection['did_not_complete_questionnaire'] = True
    rejection.loc[rejection.index.isin(ids_from_questionnaire), 'did_not_complete_questionnaire'] = False

    return rejection


def print_summary_stats(df, title):
    print(f"{title} Statistics:")
    print(f"\tMean: {df.mean():,.2f}")
    print(f"\tStd. Dev.: {df.std():,.2f}")
    print(f"\tMin.: {df.min():,.0f}")
    print(f"\tLower Quartile: {df.quantile(.25):,.2f}")
    print(f"\tMedian: {df.median():,.2f}")
    print(f"\tUpper Quartile: {df.quantile(.75):,.2f}")
    print(f"\tMax.: {df.max():,.0f}")


def analyse(data, questionnaire_csv_paths):
    reject_list = reject_workers(data, questionnaire_csv_paths)

    num_unique_workers = data['WorkerId'].nunique()
    num_tasks_completed_by_worker = data['WorkerId'].value_counts()
    num_one_task = (num_tasks_completed_by_worker == 1).sum()

    workers_to_reject = reject_list[reject_list['too_fast'] & reject_list['all_same_answers']].index
    num_tasks_to_reject = data['WorkerId'].isin(workers_to_reject).sum()

    print(f"Number of Unique Workers: {num_unique_workers:,d}")
    print(f"Number of Tasks Completed: {len(data):,d}")
    print(f"Num. One Task Completed: {num_one_task}")
    print(f"Rejection Stats (reason, count, rejection rate):")

    for column in reject_list:
        title = ' '.join(column.split('_')).capitalize()
        n = reject_list[column].sum()
        rate = n / num_unique_workers * 100
        print(f"\t{title}: {reject_list[column].sum():,d}/{num_unique_workers:,d} ({rate:.2f}%)")

    print(f"\tTasks that would be rejected: {num_tasks_to_reject:,d}/{len(data):,d}")

    print_summary_stats(num_tasks_completed_by_worker, 'Task Completion')
    print_summary_stats(data['Score'], 'Score')
    print_summary_stats(data['WorkTimeInSeconds'], 'Time To Answer')

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(16, 10))

    ax = axes[0, 0]
    sns.histplot(data, x='Score', discrete=True, ax=ax)
    ax.set_title('Distribution of Score')

    ax = axes[0, 1]
    sns.boxplot(data=data, y='Model', x='Score', ax=ax)
    ax.set_title('Distribution of Scores by Model')

    ax = axes[0, 2]
    sns.boxplot(data=data, y='Frame', x='Score', orient='h', ax=ax)
    ax.set_ylabel('Input Image ID')
    ax.set_title('Distribution of Scores by Input Image')

    ax = axes[1, 0]
    sns.histplot(num_tasks_completed_by_worker, binwidth=10, kde=True, ax=ax)
    ax.set_title('Distribution of Num. Tasks Completed by Workers')
    ax.set_xlabel('Num. Tasks Completed')

    ax = axes[1, 1]
    sns.histplot(data, x='WorkTimeInSeconds', binwidth=5, kde=True, ax=ax)
    ax.set_title('Distribution of Time to Complete Tasks')
    ax.set_xlabel('Time to Complete Task (s)')

    ax = axes[1, 2]
    reject_reasons_table = reject_list.sum().rename(
        index=lambda c: ' '.join(c.split('_')).capitalize()).reset_index().rename(
        columns={'index': 'reason', 0: 'count'})
    plots = sns.barplot(data=reject_reasons_table, x='reason', y='count', ax=ax)

    for bar in plots.patches:
        plots.annotate(f"{bar.get_height():,.0f} ({bar.get_height() / num_unique_workers * 100:,.1f}%)",
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=11, xytext=(0, 4),
                       textcoords='offset points')

    ax.set_title('Reasons for Rejecting Workers')
    ax.set_ylabel('Count')
    ax.set_xlabel('Reason')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)

    plt.tight_layout()
    plt.show()

    kappa, (lower, upper), p_value = fleiss_kappa(data)
    print(f"Fleiss Kappa:")
    print(f"\tKappa: {kappa:.3f}")
    print(f"\tConfidence Interval: [{lower:.3f}, {upper:.3f}]")
    print(f"\tP-Value: {p_value:.3f}")


def fleiss_kappa(data):
    data['Item'] = data['Model'] + '_' + data['Frame'].astype(str)
    num_workers = 20
    num_items = data['Model'].nunique() * data['Frame'].nunique()

    counts = dict()

    for item in data['Item']:
        counts[item] = data.loc[data['Item'] == item, 'Score'].value_counts().sort_index()

    counts_df = pd.DataFrame(counts).T.fillna(0.0)
    p = data['Score'].value_counts(normalize=True).sort_index()
    P = (np.square(counts_df).sum(axis=1) - num_workers) / (num_workers * (num_workers - 1))
    P_mean = P.sum() / num_items
    P_expected_mean = np.sum(np.square(p))
    kappa = (P_mean - P_expected_mean) / (1.0 - P_expected_mean)

    standard_error_per_cat = np.sqrt(2 / (num_items * num_workers * (num_workers - 1)))

    q = counts_df.sum(axis=0) / (num_workers * num_items)
    b = q * (1 - q)
    standard_error = standard_error_per_cat * np.sqrt(np.square(np.sum(b)) - np.sum(b * (1 - 2 * q))) / np.sum(b)
    z = kappa / standard_error
    p = 2 * (1.0 - norm.cdf(z))

    alpha = 0.05
    lower = kappa + standard_error * norm.ppf(alpha / 2)
    upper = kappa - standard_error * norm.ppf(alpha / 2)
    confidence_interval = (lower, upper)

    return kappa, confidence_interval, p


def main(rating_csv_path, *questionnaire_csv_paths):
    print("*" * 80)
    print("All Responses")
    print("*" * 80)

    data = process_raw_data(rating_csv_path)
    analyse(data, questionnaire_csv_paths)

    print("*" * 80)
    print("Answered Questionnaire Only")
    print("*" * 80)

    ids_from_questionnaire = set()

    for path in questionnaire_csv_paths:
        df = pd.read_csv(path)
        ids_from_questionnaire = ids_from_questionnaire.union(df['WorkerId'])

    data_questionnaire = data[data['WorkerId'].isin(ids_from_questionnaire)]
    analyse(data_questionnaire, questionnaire_csv_paths)


if __name__ == '__main__':
    plac.call(main)
