import datetime
import os
import shutil
import subprocess
import time
from typing import Sequence, Optional

import jinja2
import numpy as np
import pandas as pd
import plac
from PIL import Image


def log(msg):
    now = datetime.datetime.now()
    print(f"[{now}] {msg}")


@plac.annotations(
    benchmark_path=plac.Annotation(type=str, kind='option'),
    nyu_dataset_path=plac.Annotation(type=str, kind='option'),
    selection_interval=plac.Annotation(type=int, kind='option'),
    max_videos=plac.Annotation(type=int, kind='option'),
    output_path=plac.Annotation(type=str, kind='option'),
    video_selection=plac.Annotation(type=int, kind='option'),
)
def main(benchmark_path='benchmark/nyu', nyu_dataset_path='data/datasets/nyuv2/',
         output_path='benchmark/study_material',
         selection_interval=30, max_videos=20, *video_selection: Optional[Sequence[int]]):
    """

    Args:
        benchmark_path: The path to the benchmark outputs.
        nyu_dataset_path: Path to the NYU dataset.
        output_path: Path to output study material to.
        selection_interval: The interval to sample inputs from the NYU dataset. No effect if `video_selection` is set.
        max_videos: The maximum number of videos to generate. No effect if `video_selection` is set.
        *video_selection: Choose which test images from the NYU dataset to use by index.
    """
    start = time.time()

    log("Params:")
    log(f"\tbenchmark_path={benchmark_path}")
    log(f"\tnyu_dataset_path={nyu_dataset_path}")
    log(f"\tselection_interval={selection_interval}")
    log(f"\toutput_path={output_path}")
    print()

    # TODO: Update script to work with pre-existing rendered videos.
    # 1. Copy videos to new folder, using the indices from the NYU csv file to select videos by file name.
    # 2. Pair off each model with the ground truth and make folders for each of these.
    # 3. Generate side-by-side videos that combine the videos from each of these pairs.
    # 4. Generate Amazon S3 URLs
    # 5. Generate MTurk template using URLs
    # Generate videos for latin square study design:
    # a. Do steps 3-4 for a single latin square
    # b. Do step 5 but code it s.t. a 'seed' value can be used to select the correct set of videos.
    #   This could be done by simply creating a set of videos per participant, and storing each set in its own S3 folder
    #   e.g. bucket.s3.com/<index>/<video_name>.avi.
    source_path = os.path.join(output_path, 'source')
    pairs_output_path = os.path.join(output_path, 'pairs')

    log("Loading NYU CSV file")
    nyu_test_csv = os.path.join(nyu_dataset_path, 'nyu2_test.csv')
    nyu_test_files = pd.read_csv(nyu_test_csv, header=None)
    if video_selection and len(video_selection) > 0:
        selected_nyu_files = nyu_test_files.iloc[list(video_selection)]
    else:
        selected_nyu_files = nyu_test_files.iloc[::selection_interval, :][:max_videos]
    print()

    benchmark_models = os.listdir(benchmark_path)
    benchmark_models = filter(lambda model: os.path.isdir(os.path.join(benchmark_path, model)), benchmark_models)
    benchmark_models = filter(lambda model: model != 'ground_truth', benchmark_models)
    benchmark_models = list(benchmark_models)

    model_selection = ['reside_enb0-random_weights', 'flat', 'reside_enb0', 'reside_senet']

    benchmark_models = list(set(model_selection).intersection(benchmark_models))

    log(f"Found {len(benchmark_models)} benchmark models: {', '.join(benchmark_models)}.")
    print()

    if os.path.isdir(source_path) and os.path.isdir(pairs_output_path):
        log(f"Cached results found in {source_path} and {pairs_output_path}.")
        print()
    else:
        paths_to_create = [
            source_path,
            pairs_output_path
        ]

        log(f"Source output path is {source_path}.")

        log("Collecting src/dst paths for ground truth dataset...")
        source_copy_jobs = []
        paths_to_create.append(os.path.join(source_path, 'ground_truth'))

        for i in selected_nyu_files.index:
            video_input = os.path.join(benchmark_path, 'ground_truth', 'video', f'{i:06d}.avi')
            video_output = os.path.join(source_path, 'ground_truth', f'{i:06d}.avi')
            source_copy_jobs.append((video_input, video_output))

        log(f"Found {len(source_copy_jobs)} videos.")

        print()
        log(f"Collecting src/dst paths for estimated depth maps...")

        for model in benchmark_models:
            model_path = os.path.join(benchmark_path, model, 'rendered_images', 'video')

            model_output_path = os.path.join(source_path, model)
            paths_to_create.append(model_output_path)

            for i in selected_nyu_files.index:
                filename = f'{i:06d}.avi'
                video_input = os.path.join(model_path, filename)
                video_output = os.path.join(model_output_path, filename)
                source_copy_jobs.append((video_input, video_output))

        print()
        log(f"Collecting src/dst paths and ffmpeg jobs for paired videos...")
        ffmpeg_jobs = []
        directory = []

        default_ffmpeg_options = ["-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "3.0"]

        for model in benchmark_models:
            pair_output_path = os.path.join(pairs_output_path, f"gt-{model}")
            pair_reverse_output_path = os.path.join(pairs_output_path, f"{model}-gt")

            paths_to_create.append(pair_output_path)
            paths_to_create.append(pair_reverse_output_path)

            for i in selected_nyu_files.index:
                filename = f'{i:06d}.avi'
                gt_video_path = os.path.join(source_path, 'ground_truth', filename)
                model_video_path = os.path.join(source_path, model, filename)

                output_filename = f'{i:06d}.mp4'
                pair_video_path = os.path.join(pair_output_path, output_filename)
                pair_reverse_video_path = os.path.join(pair_reverse_output_path, output_filename)

                ffmpeg_cmd = ["ffmpeg", "-i", gt_video_path, "-i", model_video_path, "-filter_complex", "hstack",
                              *default_ffmpeg_options, pair_video_path]
                ffmpeg_jobs.append(ffmpeg_cmd)

                ffmpeg_reverse_cmd = ["ffmpeg", "-i", model_video_path, "-i", gt_video_path, "-filter_complex", "hstack",
                                      *default_ffmpeg_options, pair_reverse_video_path]
                ffmpeg_jobs.append(ffmpeg_reverse_cmd)

                directory.append((i, model, pair_video_path, pair_reverse_video_path))

        print()
        log(f"Creating {len(paths_to_create)} folders...")

        for path in paths_to_create:
            log(f"Creating path {path}...")
            os.makedirs(path)

        print()
        log(f"Copying source videos from {benchmark_path} to {output_path}...")

        for i, (src, dst) in enumerate(source_copy_jobs):
            log(f"[{i + 1:03,d}/{len(source_copy_jobs):03,d}] Copy {src} -> {dst}")
            shutil.copyfile(src, dst)

        print()
        log("Running FFMPEG jobs...")

        for i, cmd_parts in enumerate(ffmpeg_jobs):
            log(f"[{i + 1:03,d}/{len(ffmpeg_jobs):03,d}] {' '.join(cmd_parts)}")
            subprocess.run(cmd_parts)

    video_names = sorted(map(lambda frame_number: f"{frame_number:06d}", selected_nyu_files.index))
    video_file_ext = ".mp4"
    pairs = list(map(lambda model_name: ['gt', model_name], benchmark_models))
    s3_base_url = "https://dl3d.s3-us-west-1.amazonaws.com/pair_videos"

    template_path = "tools/template.html"

    with open(template_path, 'r') as f:
        template = jinja2.Template(f.read())

    rendered_template = template.render(video_names=video_names, video_file_ext=video_file_ext, pairs=pairs, s3_base_url=s3_base_url)

    print(rendered_template)

    rendered_template_path = os.path.join(output_path, 'template.html')

    with open(rendered_template_path, 'w') as f:
        f.write(rendered_template)

    print(f"Instructions:")
    print(f"1. Upload the contents of the folder {pairs_output_path} to the following URL: {s3_base_url}.")
    print(f"2. Create a survey on MTurk.")
    print(f"3. Copy and paste into the design area the HTML code from the following file: {rendered_template_path}.")

    log(f"Done in {time.time() - start:,.02f}s.")


if __name__ == '__main__':
    plac.call(main)
