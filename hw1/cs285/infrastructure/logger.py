import os
from tensorboardX import SummaryWriter
import numpy as np

class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_image(self, image, name, step):
        assert(len(image.shape) == 3)  # [C, H, W]
        self._summ_writer.add_image('{}'.format(name), image, step)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

    def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):
        # Keep only the number of videos we want to
        max_videos_to_save = np.min([max_videos_to_save, len(paths)])
        paths = paths[:max_videos_to_save]

        # Reshape the rollouts into image frames
        videos = [np.transpose(p['image_obs'][:, 0], [0, 3, 1, 2]) for p in paths]

        # Calculate max rollout length
        max_length = max(map(len, videos))

        # Pad rollouts to all be same length
        for i in range(len(videos)):
            v = videos[i]

            # If the video is already the max length, do nothing
            if len(v) >= max_length:
                continue

            # We extend the video by repeating the last frame
            last_frame = v[-1]

            # Calculate the number of frames we need to pad to match the max video length
            n_missing_frames = max_length - v.shape[0]

            # Create the repeated video frames
            padding = np.tile([last_frame], (n_missing_frames, 1, 1, 1))

            # Concatenate the video and the repeated frames together
            videos[i] = np.concatenate([v, padding], 0)

        # Concatenate all videos together
        videos = np.stack(videos, 0)

        # Log videos to tensorboard event file
        self.log_video(videos, video_title, step, fps=fps)

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_graph(self, array, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        im = plot_graph(array)
        self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def flush(self):
        self._summ_writer.flush()




