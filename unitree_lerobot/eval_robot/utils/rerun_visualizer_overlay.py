# Local addition (not upstream): per-joint overlay visualization for offline
# dataset evaluation. Each joint gets one panel showing the recorded state
# ("actual", gray-blue) and the policy's output ("predicted", orange) overlaid,
# with G1 arm joint names instead of indices. Used only by eval_g1_dataset.py;
# eval_g1.py and replay_robot.py keep the original RerunLogger.
import torch
from typing import Any

import rerun as rr
import rerun.blueprint as rrb

from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger

G1_ARM_JOINT_NAMES = [
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",
    "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw",
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
    "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw",
]

ACTUAL_COLOR = (110, 130, 170)
PREDICTED_COLOR = (255, 140, 0)


class OverlayRerunLogger(RerunLogger):
    """RerunLogger variant: one panel per joint, actual vs predicted overlaid."""

    def _joint_names(self) -> list[str]:
        if self._n_joints == len(G1_ARM_JOINT_NAMES):
            return G1_ARM_JOINT_NAMES
        return [f"joint_{i}" for i in range(self._n_joints)]

    def _initialize_from_data(self, step_data: dict[str, Any]):
        state = step_data.get("observation.state")
        self._n_joints = len(state) if state is not None else 0
        super()._initialize_from_data(step_data)

    def setup_blueprint(self):
        names = self._joint_names()

        def joint_view(name: str) -> rrb.TimeSeriesView:
            return rrb.TimeSeriesView(
                origin=f"{self.prefix}joints/{name}",
                name=name,
                time_ranges=[
                    rrb.VisibleTimeRange(
                        "frame",
                        start=rrb.TimeRangeBoundary.cursor_relative(seq=-self.idxrangeboundary),
                        end=rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
                plot_legend=rrb.PlotLegend(visible=True),
            )

        image_views = [
            rrb.Spatial2DView(
                origin=f"{self.prefix}images/{key.replace('observation.images.', '')}",
                name=key.replace("observation.images.", ""),
            )
            for key in self._image_keys
        ]

        # left-arm column and right-arm column, shoulder->wrist top to bottom
        half = len(names) // 2
        left_col = rrb.Vertical(contents=[joint_view(n) for n in names[:half]])
        right_col = rrb.Vertical(contents=[joint_view(n) for n in names[half:]])

        if image_views:
            camera = image_views[0] if len(image_views) == 1 else rrb.Vertical(contents=image_views)
            layout = rrb.Horizontal(contents=[camera, left_col, right_col], column_shares=[2, 1.2, 1.2])
        else:
            layout = rrb.Horizontal(contents=[left_col, right_col])

        rr.send_blueprint(layout)

        # fixed colors + legend labels, set once
        for name in names:
            rr.log(f"{self.prefix}joints/{name}/actual",
                   rr.SeriesLines(colors=ACTUAL_COLOR, names="actual"), static=True)
            rr.log(f"{self.prefix}joints/{name}/predicted",
                   rr.SeriesLines(colors=PREDICTED_COLOR, names="predicted"), static=True)

        self.blueprint_sent = True

    def log_step(self, step_data: dict[str, Any]):
        if not self.blueprint_sent:
            self._initialize_from_data(step_data)

        if self._index_key in step_data:
            rr.set_time_sequence("frame", step_data[self._index_key].item())

        episode_idx = step_data.get(self._episode_index_key, torch.tensor(-1)).item()
        if episode_idx != self.current_episode:
            self.current_episode = episode_idx
            task_name = step_data.get(self._task_key, "Unknown Task")
            rr.log(f"{self.prefix}info/task",
                   rr.TextLog(f"Starting Episode {self.current_episode}: {task_name}",
                              level=rr.TextLogLevel.INFO))

        for key in self._image_keys:
            if key in step_data:
                image_tensor = step_data[key]
                if image_tensor.ndim > 2:
                    if image_tensor.shape[0] in [1, 3, 4]:
                        image_tensor = image_tensor.permute(1, 2, 0)
                    rr.log(f"{self.prefix}images/{key.replace('observation.images.', '')}",
                           rr.Image(image_tensor))

        names = self._joint_names()
        if self._state_key in step_data:
            for name, val in zip(names, step_data[self._state_key]):
                rr.log(f"{self.prefix}joints/{name}/actual", rr.Scalars(val.item()))
        if self._action_key in step_data:
            for name, val in zip(names, step_data[self._action_key]):
                rr.log(f"{self.prefix}joints/{name}/predicted", rr.Scalars(val.item()))


def visualization_data(idx, observation, state, action, online_logger):
    """Same signature as rerun_visualizer.visualization_data."""
    item_data: dict[str, Any] = {
        "index": torch.tensor(idx),
        "observation.state": state,
        "action": action,
    }
    for k, v in observation.items():
        if k not in ("index", "observation.state", "action"):
            item_data[k] = v
    online_logger.log_step(item_data)
