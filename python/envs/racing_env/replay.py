
import numpy as np
import pyqtgraph as pg

from interface.replay_core import BaseRenderer

TRACK_C = (50, 50, 58)
CENTER_C = (80, 80, 90)
FINISH_C = (230, 230, 120)
ROCK_C = (120, 120, 120)
DEAD_C = (90, 90, 90)
CAR_COLS = [(220, 70, 70), (70, 130, 225), (90, 200, 120), (210, 180, 70)]


def _parse(raw):
    cars, rocks = raw[0], raw[1]
    cars = np.asarray(cars, dtype=float).reshape(-1, 4)
    rocks = np.asarray(rocks, dtype=float).reshape(-1, 2)
    return cars, rocks


class CarRenderer(BaseRenderer):
    lock_aspect = False

    def __init__(self, n_agents=2):
        self.n = n_agents

    def setup(self, plot, cfg):
        self.cfg = cfg
        L, W, self.n = cfg["track_len"], cfg["track_width"], cfg["n_agents"]

        half = W / 2
        self.world_range = (-2.0, L + 2.0, -half, half)

        self.plot = plot
        plot.showGrid(x=True, y=False, alpha=0.15)

        top = pg.PlotCurveItem([0, L], [half, half], pen=pg.mkPen(CENTER_C, width=2))
        bot = pg.PlotCurveItem([0, L], [-half, -half], pen=pg.mkPen(CENTER_C, width=2))
        plot.addItem(pg.FillBetweenItem(top, bot, brush=pg.mkBrush(*TRACK_C)))
        plot.addItem(top)
        plot.addItem(bot)
        plot.addItem(pg.PlotCurveItem([0, L], [0, 0], pen=pg.mkPen(CENTER_C, width=1, style=pg.QtCore.Qt.DashLine)))
        plot.addItem(pg.PlotCurveItem([L, L], [-half, half], pen=pg.mkPen(FINISH_C, width=3)))

        self.i_rocks = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(*ROCK_C), pen=None)
        plot.addItem(self.i_rocks)

        self.i_trail, self.i_dots, self.i_car, self.i_vel = [], [], [], []
        for a in range(self.n):
            col = CAR_COLS[a % len(CAR_COLS)]
            tr = pg.PlotCurveItem(pen=pg.mkPen(col, width=2))
            dt = pg.ScatterPlotItem(size=7, brush=pg.mkBrush(255, 255, 255), pen=pg.mkPen(col))
            car = pg.ScatterPlotItem(size=20, brush=pg.mkBrush(*col), pen=pg.mkPen("w"))
            vel = pg.PlotCurveItem(pen=pg.mkPen(col, width=2))
            for it in (tr, dt, vel, car):
                plot.addItem(it)

            self.i_trail.append(tr); self.i_dots.append(dt)
            self.i_car.append(car); self.i_vel.append(vel)

        self.banner = pg.TextItem("", anchor=(0.5, 0.5))
        self.banner.setPos(L / 2, 0)
        plot.addItem(self.banner)

    def on_reset(self, run):
        cars, rocks = _parse(run.raw)
        self.i_rocks.setData(pos=rocks if len(rocks) else np.empty((0, 2)))
        self.trails = [[tuple(c[:2])] for c in cars]
        self.dots = [[] for _ in range(len(cars))]
        self.banner.setText("")

    def on_decision(self, run):
        cars_prev, _ = _parse(run.raw_prev)
        cars, _ = _parse(run.raw)
        for i, c in enumerate(cars):
            if c[3] > 0.5:
                self.trails[i].append((c[0], c[1]))
        if run.aid < len(self.dots):
            self.dots[run.aid].append(tuple(cars_prev[run.aid][:2]))

    def draw(self, run, frac):
        cars_prev, _ = _parse(run.raw_prev)
        cars, _ = _parse(run.raw)
        if run.done:
            frac = 1.0

        for i in range(len(cars)):
            s = cars_prev[i, 0] + (cars[i, 0] - cars_prev[i, 0]) * frac
            d = cars_prev[i, 1] + (cars[i, 1] - cars_prev[i, 1]) * frac
            alive = cars[i, 3] > 0.5
            col = CAR_COLS[i % len(CAR_COLS)] if alive else DEAD_C

            arr = np.asarray(self.trails[i][:-1] + [(s, d)], dtype=float)
            if len(arr) >= 2:
                self.i_trail[i].setData(arr[:, 0], arr[:, 1])

            if self.dots[i]:
                self.i_dots[i].setData(pos=np.asarray(self.dots[i], dtype=float))

            self.i_car[i].setData(pos=[(s, d)], brush=pg.mkBrush(*col))
            self.i_vel[i].setData([s, s + cars[i, 2] * 0.15], [d, d])

        if run.done:
            self.banner.setText(run.status, color=FINISH_C)

    def hud(self, run) -> str:
        cars, _ = _parse(run.raw)
        alive = int((cars[:, 3] > 0.5).sum())
        lead = float(cars[:, 0].max()) if len(cars) else 0.0
        per = " ".join(f"a{i}:{n}" for i, n in enumerate(run.per_agent_dec))
        return (f"turn=agent {run.aid}   t={run.t:5.2f}s   reward={run.reward:8.3f}   "
                f"tau={run.tau:4.2f}s   lead_s={lead:5.1f}   alive={alive}/{len(cars)}   "
                f"decisions[{per}]   [{run.status}]")