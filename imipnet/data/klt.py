import bisect
from typing import Sequence, Optional, Tuple

import cv2
import numpy as np
import scipy.spatial.distance
import sklearn.neighbors

from .pairs import CorrespondencePair


class SequenceOverlap:
    def __init__(self: 'SequenceOverlap', initial_num_keypoints: int):
        self.tracked_ids_per_frame = [
            np.array(range(initial_num_keypoints))
        ]
        self._next_keypoint_id = initial_num_keypoints

    def add_frame(self: 'SequenceOverlap', tracked_from_last_frame_mask: np.ndarray, num_new_keypoints: int):
        self.tracked_ids_per_frame.append(
            np.hstack((
                self.tracked_ids_per_frame[-1][tracked_from_last_frame_mask],
                np.array(range(num_new_keypoints)) + self._next_keypoint_id
            ))
        )
        self._next_keypoint_id += num_new_keypoints

    def find_frames_with_overlap(self: 'SequenceOverlap', frame_id: int, min_overlap: float) -> np.ndarray:
        num_keypoints_in_frame = self.tracked_ids_per_frame[frame_id].size
        if num_keypoints_in_frame == 0:
            return np.array([])

        frames = []
        for j in range(frame_id + 1, len(self.tracked_ids_per_frame)):
            overlap = np.intersect1d(self.tracked_ids_per_frame[frame_id], self.tracked_ids_per_frame[j], True).size
            overlap /= num_keypoints_in_frame
            if overlap < min_overlap:
                break
            frames.append(j)
        return np.array(frames)

    def plot_keypoint_ids(self: 'SequenceOverlap'):
        frame_ids = [i * np.ones(self.tracked_ids_per_frame[i].size)
                     for i in range(len(self.tracked_ids_per_frame))]
        frame_ids = np.hstack(frame_ids)
        keypoint_ids = np.hstack(self.tracked_ids_per_frame)

        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.plot(keypoint_ids, frame_ids, linestyle='', marker='.', markersize=1)

    def plot_frame_overlap(self: 'SequenceOverlap'):
        num_frames = len(self.tracked_ids_per_frame)
        frame_grid = np.zeros((num_frames, num_frames))
        for i in range(num_frames):
            frame_grid[i, i] = 1
            for j in range(i + 1, num_frames):
                overlap = np.intersect1d(self.tracked_ids_per_frame[i], self.tracked_ids_per_frame[j], True).size
                if overlap == 0:
                    break
                frame_grid[i, j] = overlap / self.tracked_ids_per_frame[i].size

        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.imshow(frame_grid)
        plt.colorbar()


class Tracker:
    TRACK_SUCCESS = 1
    TRACK_NEAR_BORDER = 2
    TRACK_POOR = 3

    def __init__(self: 'Tracker', border_margin: Optional[int] = 15, nms_radius: Optional[int] = 5):
        self._border_margin = border_margin
        self._nms_radius = nms_radius
        self.__detector = None

    @property
    def _FAST_detector(self) -> cv2.FastFeatureDetector:
        if self.__detector is None:
            self.__detector = cv2.FastFeatureDetector_create()
        return self.__detector

    @staticmethod
    def _pair_results_index(i: int, n: int) -> int:
        # Given a n*(n-1)/2 sized array where each item represents
        # the result of a function applied to a pair of elements from a set of n,
        # return the starting index of the results for the pairs of elements
        # i and j where i < j < n.
        #
        # This can be derived by simplifying the formula given by
        # scipy.spatial.distance.squareform  i.e.
        # (n choose 2) - ((n - i) choose 2) + (j - i - 1) with
        # j = i + 1.
        return i * n - (i * i + i) // 2

    def _find_new_FAST_keypoints(self: 'Tracker', img: np.ndarray,
                                 existing_keypoints_rc: Optional[np.ndarray] = None) -> np.ndarray:
        if existing_keypoints_rc is None:
            existing_keypoints_rc = np.array([[], []])

        # Find keypoints and store them as column vectors
        new_keypoints_cv = self._FAST_detector.detect(img)
        new_keypoints_rc = np.array([[i.pt[1], i.pt[0]] for i in new_keypoints_cv]).T
        new_keypoint_responses = np.array([i.response for i in new_keypoints_cv])

        # Reject any new keypoints that are close to existing keypoints
        if existing_keypoints_rc.shape[1] > 0:
            # sklearn requires the points to be in row vector format
            nearest_neighbor_searcher = sklearn.neighbors.NearestNeighbors(
                n_neighbors=1, metric='chebyshev'
            ).fit(existing_keypoints_rc.T)
            distances, _ = nearest_neighbor_searcher.kneighbors(new_keypoints_rc.T)
            nms_filter = (distances > self._nms_radius).ravel()
            new_keypoints_rc = new_keypoints_rc[:, nms_filter]
            new_keypoint_responses = new_keypoint_responses[nms_filter]

        n = new_keypoints_rc.shape[1]
        if n == 0:
            return new_keypoints_rc

        # Reject any new keypoints that are close to other new keypoints.
        # We sort the keypoints by their response from low to high here.
        # When we reject points via non maximum suppression, we want to
        # reject the points with the lowest responses first.
        new_keypoints_rc = new_keypoints_rc[:, np.argsort(new_keypoint_responses)]
        distances = scipy.spatial.distance.pdist(new_keypoints_rc.T, metric='chebyshev')
        # This is faster than applying squaredist and deriving the filter from there.
        # Checks if each keypoint is close to any of the others.
        # On each iteration of i, the number of distances considered shrinks.
        # This is because we only consider a pair of keypoints once.
        idx_fun = Tracker._pair_results_index
        nms_filter = np.array([
            not np.any(
                distances[idx_fun(i, n): idx_fun(i + 1, n)] < self._nms_radius
            ) for i in range(n)
        ])
        new_keypoints_rc = new_keypoints_rc[:, nms_filter]

        # re-sort the keypoints from best to worst
        new_keypoints_rc = np.fliplr(new_keypoints_rc)
        return new_keypoints_rc

    def _find_points_near_border(self: 'Tracker', points_xy: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:

        near_lower_border = np.array([[self._border_margin, self._border_margin]]) > points_xy
        near_lower_border = np.array([x[0][0] or x[0][1] for x in near_lower_border])
        near_upper_border = np.array([[img_shape[1] - self._border_margin,
                                       img_shape[0] - self._border_margin]]) <= points_xy
        near_upper_border = np.array([x[0][0] or x[0][1] for x in near_upper_border])
        near_border = np.logical_or(near_lower_border, near_upper_border)
        return near_border

    def track(self: 'Tracker', img_1: np.ndarray, img_2: np.ndarray,
              img_1_points_rc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if img_1_points_rc.size == 0:
            return np.array([], dtype=int), np.zeros_like(img_1_points_rc)

        img_1_points_xy = np.reshape(np.fliplr(img_1_points_rc.T), (-1, 1, 2)).astype(np.float32)
        img_2_points_xy, _, _ = cv2.calcOpticalFlowPyrLK(img_1, img_2, img_1_points_xy, None)
        img_1_points_xy_reverse_flow, _, _ = cv2.calcOpticalFlowPyrLK(img_2, img_1, img_2_points_xy, None)
        symmetry_err = np.linalg.norm(img_1_points_xy_reverse_flow - img_1_points_xy, axis=2)
        poor_tracks = (symmetry_err >= 1).ravel()

        near_border = self._find_points_near_border(img_2_points_xy, img_2.shape)

        track_status = np.ones(img_1_points_rc.shape[1], dtype=int) * Tracker.TRACK_SUCCESS
        track_status[poor_tracks] = Tracker.TRACK_POOR
        track_status[near_border] = Tracker.TRACK_NEAR_BORDER

        track_filter = ~(poor_tracks | near_border)
        img_2_points_rc = np.fliplr(np.reshape(
            img_2_points_xy[track_filter, :, :],
            (-1, 2)
        )).T

        return track_status, img_2_points_rc

    def find_sequence_overlap(self: 'Tracker', image_sequence: Sequence[np.ndarray],
                              max_num_points: int, show_keypoints_tracking: Optional[bool] = False) -> SequenceOverlap:
        prev_img = image_sequence[0]
        keypoints = self._find_new_FAST_keypoints(prev_img)
        keypoints = keypoints[:, :max_num_points] if keypoints.shape[1] > max_num_points else keypoints
        overlap_tracker = SequenceOverlap(keypoints.shape[1])

        if show_keypoints_tracking:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            plt.figure("Keypoint Tracking For Sequence Overlap")
            plt.title("Frame %d / %d" % (0, len(image_sequence)))
            plt.imshow(prev_img, cmap='gray')
            plt.plot(keypoints[1, :], keypoints[0, :], ls='', marker='x', ms=5, c='r')
            plt.show()
            plt.pause(0.001)
            plt.clf()

        for i in range(1, len(image_sequence)):
            curr_img = image_sequence[i]

            tracked_status, keypoints = self.track(prev_img, curr_img, keypoints)

            tracking_succeeded_mask = tracked_status == Tracker.TRACK_SUCCESS

            new_keypoints = self._find_new_FAST_keypoints(curr_img, keypoints)
            num_needed_keypoints = max_num_points - keypoints.shape[1]
            new_keypoints = new_keypoints[:, :num_needed_keypoints] \
                if new_keypoints.shape[1] > num_needed_keypoints else new_keypoints

            overlap_tracker.add_frame(tracking_succeeded_mask, new_keypoints.shape[1])

            keypoints = np.hstack((keypoints, new_keypoints))

            if show_keypoints_tracking:
                plt.figure("Keypoint Tracking For Sequence Overlap")
                plt.title("Frame %d / %d" % (i, len(image_sequence)))
                plt.imshow(curr_img, cmap='gray')
                plt.plot(keypoints[1, :], keypoints[0, :], ls='', marker='x', ms=5, c='r')
                plt.draw()
                plt.pause(0.001)
                plt.clf()

            prev_img = curr_img

        return overlap_tracker


class KLTPair(CorrespondencePair):

    def __init__(self, images: Sequence[np.ndarray], tracker: Tracker, name: str):
        self._sequence = list(images)  # Ensure the kitti_sequence is loaded into memory
        self._name = name
        self._tracker = tracker

    def correspondences(self, pixels_xy: np.ndarray, inverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        seq_length = len(self._sequence)
        iter_order = range(0, seq_length)
        if inverse:
            iter_order = iter_order[::-1]

        prev_img = self._sequence[iter_order[0]]

        keypoints_rc = np.flipud(pixels_xy)
        keypoint_indices = np.arange(keypoints_rc.shape[1], dtype=int)
        for i_iter in range(1, seq_length):
            #     import matplotlib
            #     matplotlib.use('TkAgg')
            #     import matplotlib.pyplot as plt
            #     plt.figure("Keypoint Tracking")
            #     plt.title("Frame %d / %d" % (0, seq_length))
            #     plt.imshow(prev_img, cmap='gray')
            #     plt.plot(keypoints_rc[1, :], keypoints_rc[0, :], ls='', marker='x', ms=5, c='r')
            #     plt.draw()
            #     plt.pause(0.001)
            #     plt.clf()
            i = iter_order[i_iter]
            curr_img = self._sequence[i]
            tracked_status, keypoints_rc = self._tracker.track(
                prev_img, curr_img, keypoints_rc
            )
            keypoint_indices = keypoint_indices[tracked_status == Tracker.TRACK_SUCCESS]
            prev_img = curr_img

        keypoints_xy = np.flipud(keypoints_rc)

        if keypoints_xy.shape[1] == 0:  # clean up empty results for torch to get rid of negative strides
            keypoints_xy = np.zeros(keypoints_xy.shape, dtype=keypoints_xy.dtype)
            keypoint_indices = np.zeros(0, dtype=int)

        return keypoints_xy, keypoint_indices

    @property
    def image_1(self) -> np.ndarray:
        return self._sequence[0]

    @property
    def image_2(self) -> np.ndarray:
        return self._sequence[-1]

    @property
    def name(self) -> str:
        return self._name


class KLTPairGenerator:

    def __init__(self: 'KLTPairGenerator', name: str, images: Sequence[np.ndarray],
                 tracker: Tracker, sequence_overlap: SequenceOverlap, minimum_overlap: float):
        self.name = name
        self.img_sequence = images
        self._tracker = tracker
        self._min_overlap = minimum_overlap
        self._overlapped_frames_cum_sum = []

        overlapped_frames = sequence_overlap.find_frames_with_overlap(0, self._min_overlap).size
        self._overlapped_frames_cum_sum.append(overlapped_frames)
        for i in range(1, len(self.img_sequence) - 1):
            overlapped_frames = sequence_overlap.find_frames_with_overlap(i, self._min_overlap).size
            self._overlapped_frames_cum_sum.append(overlapped_frames + self._overlapped_frames_cum_sum[-1])

    def __len__(self):
        return self._overlapped_frames_cum_sum[-1]

    def __getitem__(self, index: int) -> CorrespondencePair:
        if index > len(self):
            raise IndexError()

        img_1_index = bisect.bisect_right(self._overlapped_frames_cum_sum, index)

        if img_1_index > 0:
            img_2_index = (img_1_index + 1) + (index - self._overlapped_frames_cum_sum[img_1_index - 1])
        else:
            img_2_index = 1 + index

        img_sequence = self.img_sequence[img_1_index:img_2_index + 1]  # Add 1 since the end of a slice is exclusive

        pair_name = "{0}: {1} {2}".format(self.name, img_1_index, img_2_index)

        return KLTPair(img_sequence, self._tracker, pair_name)
