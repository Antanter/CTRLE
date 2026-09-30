
import re

import pyqtgraph as pg
import torch
from pyqtgraph.Qt import QtCore, QtWidgets

from interface.registry import REGISTRY
from interface.replay_core import EpisodeRunner

FPS = 60
DT = 1.0 / FPS
HOLD_AFTER_DONE = 1.5


def _sibling_ckpts(path: str, n: int) -> list[str]:
    m = re.search(r"(\d+)(?=\.[^.]+$)", path)
    if not m:
        return [path]
    out = []
    for a in range(n):
        cand = path[:m.start(1)] + str(a) + path[m.end(1):]
        out.append(cand)
    return out


class ReplayTab(QtWidgets.QWidget):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.run = None
        self.renderer = None
        self.spec = None
        self.phase = 0.0
        self.hold = 0.0
        self.paused = False
        self.ep = 0

        top = QtWidgets.QHBoxLayout()
        self.cb_env = QtWidgets.QComboBox()
        for code, spec in REGISTRY.items():
            self.cb_env.addItem(spec.name, userData=code)

        self.btn_load = QtWidgets.QPushButton("Load checkpoint(s)…")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_next = QtWidgets.QPushButton("Next episode")
        self.btn_pause.setEnabled(False)
        self.btn_next.setEnabled(False)

        self.sl_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_speed.setRange(1, 200)
        self.sl_speed.setValue(20)
        self.sl_speed.setFixedWidth(160)
        self.lb_speed = QtWidgets.QLabel("x2.0")

        self.btn_load.clicked.connect(self.load_ckpt)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_next.clicked.connect(self.next_episode)
        self.sl_speed.valueChanged.connect(
            lambda v: self.lb_speed.setText(f"x{v / 10:.1f}"))

        for w in (QtWidgets.QLabel("env"), self.cb_env, self.btn_load,
                  self.btn_pause, self.btn_next,
                  QtWidgets.QLabel("speed"), self.sl_speed, self.lb_speed):
            top.addWidget(w)
        top.addStretch(1)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.hud = QtWidgets.QLabel("load a checkpoint to start")
        self.hud.setStyleSheet("font-family: monospace")

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.plot, 1)
        lay.addWidget(self.hud)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000 * DT))

    def load_ckpt(self):
        spec = REGISTRY[self.cb_env.currentData()]
        paths = []
        if spec.n_agents > 1:
            for a in range(spec.n_agents):
                p, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, f"Checkpoint for agent {a} ({spec.name})",
                    spec.default_ckpt, "PyTorch (*.pt *.pth)")
                if not p:
                    return
                paths.append(p)
        else:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, f"Checkpoint for {spec.name}", spec.default_ckpt, "PyTorch (*.pt *.pth)")
            if not p:
                return
            paths = [p]

        env = spec.make_single(0)
        agents = []
        try:
            for a, p in enumerate(paths):
                ag = spec.make_agent(env, a)
                ag.load_state_dict(torch.load(p, map_location=self.device))
                agents.append(ag.to(self.device).eval())
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
            return

        self.plot.clear()
        self.spec = spec
        self.renderer = spec.make_renderer()
        self.renderer.setup(self.plot, env.config())
        x0, x1, y0, y1 = self.renderer.world_range
        self.plot.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
        self.plot.setAspectLocked(self.renderer.lock_aspect)

        self.run = EpisodeRunner(spec, agents, self.device)
        self.renderer.on_reset(self.run)
        self.phase, self.hold, self.ep = 0.0, 0.0, 1
        self.btn_pause.setEnabled(True)
        self.btn_next.setEnabled(True)

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.setText("Resume" if self.paused else "Pause")

    def next_episode(self):
        if self.run is None:
            return
        self.run.reset()
        self.renderer.on_reset(self.run)
        self.phase, self.hold = 0.0, 0.0
        self.ep += 1

    def tick(self):
        if self.run is None:
            return

        speed = self.sl_speed.value() / 10.0

        if self.run.done:
            self.hold -= DT
            if self.hold <= 0:
                self.next_episode()
        elif not self.paused:
            self.phase += DT * speed
            while self.phase >= self.run.tau and not self.run.done:
                self.phase -= self.run.tau
                alive = self.run.step()
                self.renderer.on_decision(self.run)
                if not alive:
                    self.hold = HOLD_AFTER_DONE
                    break

        frac = min(self.phase / self.run.tau, 1.0) if self.run.tau > 0 else 0.0
        self.renderer.draw(self.run, frac)
        self.hud.setText(f"ep {self.ep}   " + self.renderer.hud(self.run))