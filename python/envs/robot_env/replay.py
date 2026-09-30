
import math
from collections import deque

import numpy as np
import pyqtgraph as pg

from interface.replay_core import BaseRenderer

GOAL_C = (78, 220, 150)
OBST_C = (232, 84, 92)
ROBOT_C = (120, 190, 255)
THRUST_C = (255, 196, 90)
TRAIL_C = (70, 110, 170)
DIM = (120, 128, 145)


def _parse(raw):
    robot, goal, obst = raw
    robot = np.asarray(robot, dtype=float).reshape(-1)
    goal = np.asarray(goal, dtype=float).reshape(-1)
    obst = np.asarray(obst, dtype=float).reshape(-1, 2)
    return robot[:2], robot[2:4], goal[:2], obst


class RobotRenderer(BaseRenderer):
    def setup(self, plot, cfg):
        self.cfg = cfg
        r_o, r_g = cfg["obstacle_r"], cfg["goal_r"]
        lo, hi, m = cfg["world_min"], cfg["world_max"], cfg["wall_margin"]

        self.world_range = (lo - 0.6, hi + 0.6, lo - 0.6, hi + 0.6)

        self.plot = plot
        plot.showGrid(x=True, y=True, alpha=0.15)

        plot.addItem(pg.PlotCurveItem(x=[lo, hi, hi, lo, lo], y=[lo, lo, hi, hi, lo], pen=pg.mkPen((200, 60, 66), width=2)))
        plot.addItem(pg.PlotCurveItem(x=[lo + m, hi - m, hi - m, lo + m, lo + m], y=[lo + m, lo + m, hi - m, hi - m, lo + m], pen=pg.mkPen((90, 45, 50), width=1, style=pg.QtCore.Qt.DashLine)))

        self.i_obst = pg.ScatterPlotItem(pxMode=False, size=2 * r_o, brush=pg.mkBrush(*OBST_C, 200), pen=pg.mkPen((255, 180, 185)))
        self.i_goal = pg.ScatterPlotItem(pxMode=False, size=2 * r_g, brush=pg.mkBrush(*GOAL_C, 90), pen=pg.mkPen((200, 255, 225), width=2))
        self.i_trail = pg.PlotCurveItem(pen=pg.mkPen(TRAIL_C, width=2))
        self.i_dots = pg.ScatterPlotItem(size=7, brush=pg.mkBrush(255, 255, 255), pen=pg.mkPen(ROBOT_C))
        self.i_thrust = pg.PlotCurveItem(pen=pg.mkPen(THRUST_C, width=3))
        self.i_robot = pg.ArrowItem(headLen=18, tailLen=6, tailWidth=4, brush=pg.mkBrush(*ROBOT_C), pen=pg.mkPen("w"))

        for it in (self.i_obst, self.i_goal, self.i_trail, self.i_dots, self.i_thrust, self.i_robot):
            plot.addItem(it)

        self.banner = pg.TextItem("", anchor=(0.5, 0.5))
        self.banner.setPos((lo + hi) / 2, (lo + hi) / 2)
        plot.addItem(self.banner)

    def on_reset(self, run):
        pos, _, goal, obst = _parse(run.raw)
        self.trail = deque(maxlen=1000)
        self.trail.append(tuple(pos))
        self.dots = []
        self.i_goal.setData(pos=[goal])
        self.i_obst.setData(pos=obst if len(obst) else np.empty((0, 2)))
        self.banner.setText("")

    def on_decision(self, run):
        pos_prev, _, _, _ = _parse(run.raw_prev)
        pos, _, _, _ = _parse(run.raw)
        self.dots.append(tuple(pos_prev))
        self.trail.append(tuple(pos))

    def draw(self, run, frac):
        pos_prev, _, _, _ = _parse(run.raw_prev)
        pos, _, goal, _ = _parse(run.raw)

        if run.done:
            frac = 1.0

        rx = pos_prev[0] + (pos[0] - pos_prev[0]) * frac
        ry = pos_prev[1] + (pos[1] - pos_prev[1]) * frac

        pts = list(self.trail)[:-1] + [(rx, ry)]
        arr = np.asarray(pts, dtype=float)
        self.i_trail.setData(arr[:, 0], arr[:, 1])

        if self.dots:
            self.i_dots.setData(pos=np.asarray(self.dots, dtype=float))

        dx, dy = pos[0] - pos_prev[0], pos[1] - pos_prev[1]
        ang = math.degrees(math.atan2(dy, dx)) if (dx * dx + dy * dy) > 1e-9 else 0.0
        self.i_robot.setStyle(angle=180.0 - ang)
        self.i_robot.setPos(rx, ry)

        tx = float(np.clip(run.action[0], -1, 1))
        ty = float(np.clip(run.action[1], -1, 1))
        self.i_thrust.setData([rx, rx + 0.5 * tx], [ry, ry + 0.5 * ty])

        if run.done:
            txt, col = {"GOAL": ("GOAL +20", GOAL_C), "CRASH": ("CRASH -10", OBST_C), "TIMEOUT": ("TIMEOUT", DIM)}.get(run.status, (run.status, DIM))
            self.banner.setText(txt, color=col)

    def hud(self, run) -> str:
        pos, vel, goal, obst = _parse(run.raw)
        dg = math.hypot(pos[0] - goal[0], pos[1] - goal[1])
        commit = run.action[2] if len(run.action) > 2 else float("nan")
        return (f"t={run.t:5.2f}s   reward={run.reward:8.3f}   tau={run.tau:4.2f}s   "
                f"d_goal={dg:4.2f}   |v|={np.linalg.norm(vel):4.2f}   "
                f"thrust=({run.action[0]:+.2f},{run.action[1]:+.2f})   "
                f"commit={commit:+.2f}   decisions={run.n_dec}   [{run.status}]")