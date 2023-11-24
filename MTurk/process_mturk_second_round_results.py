import plac
import pandas as pd
from matplotlib import pyplot as plt


def main(csv_path):
    df = pd.read_csv(csv_path)
    print(df)

    gt = 'gt'
    models = ['reside_enb0-random_weights', 'flat', 'reside_enb0', 'reside_senet']
    videos = list(map(lambda x: f"{x:06d}", [0, 30, 66, 260]))
    num_tasks = len(models) * len(videos)
    scale_range = 7

    answer_prefix = 'Answer.'
    metadata = [col for col in df.columns if not col.startswith(answer_prefix)]
    demographics = [
        'age', 'agree-to-terms.on', 'gender.female',
        'gender.male', 'gender.other', 'good-vision.no', 'good-vision.yes'
    ]
    exit_survey = ['feedback-comments', 'feedback-realism']
    realism_answers = [f"{model}-{video}-realism" for model in models for video in videos]
    gt_realism_answers = [col for col in df.columns for video in videos if f"gt-{video}" in col and 'realism' in col]
    similarity_answers = [f"{gt}-{model}-{video}-similarity" for model in models for video in videos]

    similarity_scores = convert_to_scores(df, similarity_answers, scale_range, answer_prefix)
    realism_scores = convert_to_scores(df, realism_answers, scale_range, answer_prefix)

    gt_realism_scores = get_gt_realism_scores(df, videos)

    print("Mean Time (Minutes) per Task:\n",
          df[['WorkerId', 'WorkTimeInSeconds']].set_index('WorkerId') / 60 / num_tasks)

    plot_similarity_scores_by_rater(similarity_scores)

    return

def plot_similarity_scores_by_video(similarity_scores):
    pass

def plot_similarity_scores_by_rater(similarity_scores):
    similarity_scores_df = pd.DataFrame.from_dict(similarity_scores)

    ax = similarity_scores_df.T.boxplot()
    ax.set_title(f"Similarity Scores by Rater (N={len(similarity_scores_df.columns)})\n"
                 f"'These two videos are similar.'")

    ax.set_ylabel("Score")
    ax.set_yticklabels(['', 'Strongly \nDisagree (1)', 'Disagree (2)', 'Somewhat \nDisagree(3) ', 'Neutral (4)',
                        'Somewhat \nAgree (5)', 'Agree (6)', 'Strongly \nAgree (7)', ''], rotation=30)
    ax.set_ylim(bottom=-.3, top=6.3)
    ax.set_xlabel("Rater")
    ax.set_xticklabels(range(1, len(similarity_scores_df) + 1))

    ax.grid(axis='x')

    plt.tight_layout()
    plt.show()


def convert_to_scores(df, columns, scale_range, answer_prefix):
    output = dict()

    for col in columns:
        data = None

        for i in range(scale_range):
            full_col = f"{answer_prefix}{col}.{i + 1}"

            if data is None:
                data = df[full_col].copy(deep=True)
            data[df[full_col]] = i

        output[col] = data

    return output


def get_gt_realism_scores(df, videos):
    gt_realism_data = dict()
    cols_to_remove = []

    for col in df.columns:
        for video in videos:
            task_id = f"gt-{video}"

            if task_id in col and 'realism' in col:
                rating = col[-1]
                gt_id = col.replace(f"Answer.{task_id}-", '').split('-')[0]
                scores = df[col].copy() * int(rating)

                dest_col = f"{task_id}-realism-{gt_id}"

                if dest_col not in gt_realism_data:
                    gt_realism_data[dest_col] = scores
                else:
                    gt_realism_data[dest_col] = gt_realism_data[dest_col].add(scores, fill_value=0)

                cols_to_remove.append(col)

    gt_realism_scores = dict()

    for key in gt_realism_data:
        # This line gets rid of the unique id at the end
        task_id = '-'.join(key.split('-')[:-1])

        if task_id in gt_realism_scores:
            gt_realism_scores[task_id] = gt_realism_scores[task_id].append(gt_realism_data[key])
        else:
            gt_realism_scores[task_id] = gt_realism_data[key].copy()

    return gt_realism_scores, cols_to_remove


if __name__ == '__main__':
    plac.call(main)
