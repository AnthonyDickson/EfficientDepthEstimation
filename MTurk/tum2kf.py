import argparse
import datetime
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
# Convert dataset into the format that Kinect Fusion and related methods require (everything in one directory >:| ).
from scipy.spatial.transform import Rotation


def log(message, end='\n', file=sys.stdout):
    print("[%s] %s" % (datetime.datetime.now(), message), file=file, end=end)


class FrameSampler:
    """
    Samples a subset of frames.
    """

    def __init__(self, start=0, stop=-1, step=1, fps=30.0, stop_is_inclusive=False):
        """
        :param start: The index of the first frame to sample.
        :param stop: The index of the last frame to index. Setting this to `-1` is equivalent to setting it to the index
            of the last frame.
        :param step: The gap between each of the selected frame indices.
        :param fps: The frame rate of the video that is being sampled. Important if you want to sample frames based on
            time based figures.
        :param stop_is_inclusive: Whether to sample frames as an open range (`stop` is not included) or a closed range
            (`stop` is included).
        """
        self.start = start
        self.stop = stop
        self.step = step
        self.fps = fps

        self.stop_is_inclusive = stop_is_inclusive

    def __repr__(self):
        kv_pairs = map(lambda kv: "%s=%s" % kv, self.__dict__.items())

        return "<%s(%s)>" % (self.__class__.__name__, ', '.join(kv_pairs))

    def frame_range(self, start, stop=-1):
        """
        Select a range of frames.

        :param start: The index of the first frame to sample (inclusive).
        :param stop: The index of the last frame to sample (inclusive only if `stop_is_inclusive` is set to `True`).
        :return: A new FrameSampler with the new frame range.
        """
        options = dict(self.__dict__)
        options.update(start=start, stop=stop)

        return FrameSampler(**options)

    def frame_interval(self, step):
        """
        Choose the frequency at which frames are sampled.

        :param step: The integer gap between sampled frames.
        :return: A new FrameSampler with the new sampling frequency.
        """
        options = dict(self.__dict__)
        options.update(step=step)

        return FrameSampler(**options)

    def time_range(self, start, stop=None):
        """
        Select a range of frames based on time.

        :param start: The time of the first frame to sample (in seconds, inclusive).
        :param stop: The time of the last frame to sample (in seconds, inclusive only if `stop_is_inclusive` is set to
            `True`).
        :return: A new FrameSampler with the new frame range.
        """
        options = dict(self.__dict__)

        start_frame = int(start * self.fps)

        if stop:
            stop_frame = int(stop * self.fps)
        else:
            stop_frame = -1

        options.update(start=start_frame, stop=stop_frame)

        return FrameSampler(**options)

    def time_interval(self, step):
        """
        Choose the frequency at which frames are sampled.

        :param step: The time (in seconds) between sampled frames.
        :return: A new FrameSampler with the new sampling frequency.
        """
        options = dict(self.__dict__)

        frame_step = int(step * self.fps)

        options.update(step=frame_step)

        return FrameSampler(**options)

    def choose(self, frames):
        """
        Choose frames based on the sampling range and frequency defined in this object.

        :param frames: The frames to sample from.
        :return: The subset of sampled frames.
        """
        num_frames = len(frames)

        if self.stop < 0:
            stop = num_frames
        else:
            stop = self.stop

        if self.stop_is_inclusive:
            stop += self.step

        return frames[self.start:stop:self.step]


import open3d as o3d


class TUMDataLoader:
    """
    Loads image, depth and pose data from a TUM formatted dataset.
    """
    # The below values are fixed and common to all subsets of the TUM dataset.
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    width = 640
    height = 480
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    fps = 30.0
    frame_time = 1.0 / fps

    def __init__(self, base_dir, is_16_bit=True,
                 pose_path="groundtruth.txt", rgb_files_path="rgb.txt",
                 depth_map_files_path="depth.txt"):
        """
        :param base_dir: The path to folder containing the dataset.
        :param is_16_bit: Whether the images are stored with 16-bit values or 32-bit values.
        :param pose_path: The name/path of the file that contains the camera pose information.
        :param rgb_files_path: The name/path of the file that contains the mapping of timestamps to image file paths.
        :param depth_map_files_path: The name/path of the file that contains the mapping of timestamps to depth map paths.
        """
        self.base_dir = Path(base_dir)
        self.pose_path = Path(os.path.join(base_dir, str(Path(pose_path))))
        self.rgb_files_path = Path(os.path.join(base_dir, str(Path(rgb_files_path))))
        self.depth_map_files_path = Path(os.path.join(base_dir, str(Path(depth_map_files_path))))

        self.is_16_bit = is_16_bit
        # The depth maps need to be divided by 5000 for the 16-bit PNG files
        # or 1.0 (i.e. no effect) for the 32-bit float images in the ROS bag files
        self.depth_scale_factor = 1.0 / 5000.0 if is_16_bit else 1.0

        self.synced_frame_data = None
        self.frames = None
        self.depth_maps = None
        self.poses = None

        self._validate_dataset()

    def _validate_dataset(self):
        """
        Check whether the dataset is valid and the expected files are present.

        :raises RuntimeError if there are any issues with the dataset.
        """
        if not self.base_dir.is_dir() or not self.base_dir.exists():
            raise RuntimeError(
                "The following path either does not exist, could not be read or is not a folder: %s." % self.base_dir)

        for path in (self.pose_path, self.rgb_files_path, self.depth_map_files_path):
            if not path.exists() or not path.is_file():
                raise RuntimeError("The following file either does not exist or could not be read: %s." % path)

    @property
    def num_frames(self):
        return len(self.frames) if self.frames is not None else 0

    @property
    def camera_matrix(self):
        return TUMDataLoader.intrinsic.intrinsic_matrix.copy()

    def _get_synced_frame_data(self):
        """
        Get the set of matching frames.

        The TUM dataset is created with a Kinect sensor.
        The colour images and depth maps given by this sensor are not synchronised and as such the timestamps never
        perfectly match.
        Therefore, we need to associate the frames with the closest timestamps to get the best set of frame pairs.

        :return: A list of 2-tuples each containing the paths to a colour image and depth map.
        """

        def load_timestamps_and_paths(list_path):
            timestamps = []
            data = []

            with open(str(list_path), 'r') as f:
                for line in f:
                    line = line.strip()

                    if line.startswith('#'):
                        continue

                    parts = line.split(' ')
                    timestamp = float(parts[0])
                    data_parts = parts[1:]

                    timestamps.append(timestamp)
                    data.append(data_parts)

            timestamps = np.array(timestamps)
            data = np.array(data)

            return timestamps, data

        image_timestamps, image_paths = load_timestamps_and_paths(self.rgb_files_path)
        depth_map_timestamps, depth_map_paths = load_timestamps_and_paths(self.depth_map_files_path)
        trajectory_timestamps, trajectory_data = load_timestamps_and_paths(self.pose_path)

        def get_match_indices(query, target):
            # This creates a M x N matrix of the difference between each of the image and depth map timestamp pairs
            # where M is the number of images and N is the number of depth maps.
            timestamp_deltas = np.abs(query.reshape(-1, 1) - target.reshape(1, -1))
            # There are more images than depth maps. So what we need is a 1:1 mapping from depth maps to images.
            # Taking argmin along the columns (axis=0) gives us index of the closest image timestamp for each
            # depth map timestamp.
            corresponding_indices = timestamp_deltas.argmin(axis=0)

            return corresponding_indices

        # Select the matching images.
        image_indices = get_match_indices(image_timestamps, depth_map_timestamps)
        image_filenames_subset = image_paths[image_indices]
        # data loaded by `load_timestamps_and_paths(...)` gives data as a 2d array (in this case a column vector),
        # but we want the paths as a 1d array.
        image_filenames_subset = image_filenames_subset.flatten()
        # Convert paths to Path objects to ensure cross compatibility between operating systems.
        image_filenames_subset = map(Path, image_filenames_subset)

        depth_map_subset = depth_map_paths.flatten()
        depth_map_subset = map(Path, depth_map_subset)

        # Select the matching trajectory readings.
        trajectory_indices = get_match_indices(trajectory_timestamps, depth_map_timestamps)
        trajectory_subset = trajectory_data[trajectory_indices]

        def process_trajectory_datum(datum):
            tx, ty, tz, qx, qy, qz, qw = map(float, datum)
            r = Rotation.from_quat((qx, qy, qz, qw)).as_rotvec().reshape((-1, 1))
            t = np.array([tx, ty, tz]).reshape((-1, 1))
            pose = np.vstack((r, t))

            return pose

        trajectory_subset = np.array(list(map(process_trajectory_datum, trajectory_subset)))

        # Rearrange pairs into the shape (N, 3) where N is the number of image and depth map pairs.
        synced_frame_data = list(zip(image_filenames_subset, depth_map_subset, trajectory_subset))

        return synced_frame_data

    def load(self, frame_sampler=FrameSampler()):
        """
        Load the data.

        :param frame_sampler: The frame sampler which chooses which frames to keep or discard.
        :return: A 4-tuple containing the frames, depth maps, camera parameters and camera poses.
        """
        log("Getting synced frame data...")
        self.synced_frame_data = self._get_synced_frame_data()

        frames = []
        depth_maps = []
        poses = []

        selected_frame_data = frame_sampler.choose(self.synced_frame_data)
        num_selected_frames = len(selected_frame_data)
        log("Selected %d frames." % num_selected_frames)

        log("Loading dataset...")

        for i, (image_path, depth_map_path, pose_data) in enumerate(selected_frame_data):
            frame_path = os.path.join(*map(str, (self.base_dir, image_path)))
            frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            frames.append(frame)

            raw_depth_map = cv2.imread(os.path.join(*map(str, (self.base_dir, depth_map_path))), cv2.IMREAD_ANYDEPTH)
            depth_map = self.depth_scale_factor * raw_depth_map
            depth_map = depth_map.astype(np.float32)
            depth_maps.append(depth_map)

            poses.append(pose_data)

            log("[%d/%d] Loading Dataset...\r" % (i + 1, num_selected_frames), end='')

        print()

        self.frames = np.array(frames)
        self.depth_maps = np.array(depth_maps)
        # TODO: Change pose data format to [t_1, r_1, t_2, r_2, ..., t_n, r_n] to [t_1, t_2, ..., t_n, r_1, r_2, ..., r_n].
        self.poses = np.vstack(poses).squeeze()

        return self

    def get_info(self):
        image_resolution = "%dx%d" % (
            self.frames[0].shape[1], self.frames[0].shape[0]) if self.frames is not None else 'N/A'
        depth_map_resolution = "%dx%d" % (
            self.depth_maps[0].shape[1], self.depth_maps[0].shape[0]) if self.frames is not None else 'N/A'

        lines = []
        # lines = [
        #     f"Dataset Info:",
        #     f"\tPath: {self.base_folder}",
        #     f"\tTrajectory Data Path: {self.pose_path}",
        #     f"\tRGB Frame List Path: {self.rgb_files_path}",
        #     f"\tDepth Map List Path: {self.depth_map_files_path}",
        #     f"",
        #     f"\tTotal Num. Frames: {self.num_frames}",
        #     f"\tImage Resolution: {image_resolution}",
        #     f"\tDepth Map Resolution: {depth_map_resolution}",
        #     f"\tIs 16-bit: {self.is_16_bit}",
        #     f"\tDepth Scale: {self.depth_scale_factor:.4f}",
        # ]

        return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', default='/path/to/source/dataset')
    parser.add_argument('--output_path', default='/path/to/save/output')
    args = parser.parse_args()

    data_loader = TUMDataLoader(args.base_path).load()

    base_path = os.path.abspath(args.base_path)
    output_path = os.path.abspath(args.output_path)

    cam_intr = data_loader.camera_matrix
    camera_trajectory = data_loader.poses.reshape((-1, 6))
    color_frames = data_loader.frames
    depth_frames = data_loader.depth_maps

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, (color, depth, pose) in enumerate(zip(color_frames, depth_frames, camera_trajectory)):
        frame_name = "frame-{:06d}".format(i)
        color_output_path = os.path.join(output_path, "{}.color.jpg".format(frame_name))
        depth_output_path = os.path.join(output_path, "{}.depth.png".format(frame_name))
        pose_output_path = os.path.join(output_path, "{}.pose.txt".format(frame_name))

        depth_16bit = (1000 * depth).astype(np.uint16)
        pose_mat = np.eye(4, dtype=np.float32)
        pose_mat[0:3, 0:3] = cv2.Rodrigues(pose[:3])[0]
        pose_mat[0:3, -1] = pose[-3:].reshape((1, -1))

        imageio.imwrite(color_output_path, color)
        imageio.imwrite(depth_output_path, depth_16bit)
        np.savetxt(pose_output_path, pose_mat)
        print("Saved data for frame {:06d}...".format(i))

    info_txt = f"""m_versionNumber = 4
    m_sensorName = UNREAL
    m_colorWidth = 640
    m_colorHeight = 480
    m_depthWidth = 640
    m_depthHeight = 480
    m_depthShift = 1000
    m_calibrationColorIntrinsic = {' '.join(map(str, cam_intr.astype(np.int).ravel()))} 
    m_calibrationColorExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 
    m_calibrationDepthIntrinsic = {' '.join(map(str, cam_intr.astype(np.int).ravel()))} 
    m_calibrationDepthExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 
    m_frames.size = {data_loader.num_frames}
    """

    with open(os.path.join(output_path, "info.txt"), 'w') as f:
        f.writelines(info_txt)
