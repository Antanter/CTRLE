
import re
import sys

import torch
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent


import agents.ppo as ppo

from interface.registry import discover, REGISTRY
discover(ROOT)

from interface.replay_tab import ReplayTab

torch.distributions.Distribution.set_default_validate_args(False)

MODELS = {
    "PPO": lambda env : ppo.Agent(env.act_dim(), env.obs_dim(), env),
}

OBST_R, GOAL_R = 0.3, 0.5

class TrainTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.proc = None
        self.x, self.y_ret, self.y_ent = [], [], []

        form = QtWidgets.QFormLayout()
        self.cb_model = QtWidgets.QComboBox(); self.cb_model.addItems(MODELS)
        self.cb_env = QtWidgets.QComboBox()
        for code, spec in REGISTRY.items():
            self.cb_env.addItem(spec.name, userData=code)

        self.sp_envs = QtWidgets.QSpinBox(); self.sp_envs.setRange(1, 4096); self.sp_envs.setValue(256)
        self.sp_eps = QtWidgets.QSpinBox(); self.sp_eps.setRange(1, 100000); self.sp_eps.setValue(200)
        self.sp_steps = QtWidgets.QSpinBox(); self.sp_steps.setRange(1, 4096); self.sp_steps.setValue(128)
        self.sp_seed = QtWidgets.QSpinBox(); self.sp_seed.setRange(0, 10**9); self.sp_seed.setValue(42)
        self.ed_lr = QtWidgets.QLineEdit("3e-4")
        self.ed_entr = QtWidgets.QLineEdit("0.02")
        for label, w in [("model", self.cb_model), ("env", self.cb_env), ("envs (N)", self.sp_envs), ("updates", self.sp_eps),
                         ("steps", self.sp_steps), ("seed", self.sp_seed), ("lr", self.ed_lr), ("entropy coef", self.ed_entr)]:
            form.addRow(label, w)

        self.btn_start = QtWidgets.QPushButton("Start training")
        self.btn_stop = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        form.addRow(self.btn_start); form.addRow(self.btn_stop)

        left = QtWidgets.QWidget(); left.setLayout(form); left.setMaximumWidth(260)

        pg.setConfigOptions(antialias=True)
        self.plot_ret = pg.PlotWidget(title="mean_return")
        self.plot_ent = pg.PlotWidget(title="entropy")
        for p in (self.plot_ret, self.plot_ent):
            p.showGrid(x=True, y=True, alpha=0.2)
        self.curve_ret = self.plot_ret.plot(pen=pg.mkPen((120, 190, 255), width=2))
        self.curve_ent = self.plot_ent.plot(pen=pg.mkPen((255, 196, 90), width=2))

        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.plot_ret, 3)
        right.addWidget(self.plot_ent, 2)
        right.addWidget(self.log, 2)

        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(left)
        lay.addLayout(right, 1)

        self.re_line = re.compile(r"update\s+(\d+).*?mean_return=\s*([-+\d.eE]+).*?entropy=\s*([-+\d.eE]+)")

    def start(self):
        self.x, self.y_ret, self.y_ent = [], [], []
        self.curve_ret.setData([], []); self.curve_ent.setData([], [])
        self.log.clear()

        args = ["-m", f"envs.{self.cb_env.currentData()}.main",
                "--envs", str(self.sp_envs.value()),
                "--episodes", str(self.sp_eps.value()),
                "--steps", str(self.sp_steps.value()),
                "--seed", str(self.sp_seed.value()),
                "--lr", self.ed_lr.text(),
                "--entr", self.ed_entr.text()]

        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_output)
        self.proc.finished.connect(self.on_finished)
        self.proc.start(sys.executable, ["-u"] + args)

        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.log.appendPlainText(f"$ {sys.executable} {' '.join(args)}")

    def stop(self):
        if self.proc:
            self.proc.kill()

    def on_output(self):
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        for line in text.splitlines():
            self.log.appendPlainText(line)
            m = self.re_line.search(line)
            if m:
                self.x.append(int(m.group(1)))
                self.y_ret.append(float(m.group(2)))
                self.y_ent.append(float(m.group(3)))
                self.curve_ret.setData(self.x, self.y_ret)
                self.curve_ent.setData(self.x, self.y_ent)

    def on_finished(self):
        self.log.appendPlainText("--- process finished ---")
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("RL lab — train & replay")
    tabs = QtWidgets.QTabWidget()
    tabs.addTab(TrainTab(), "Train")
    tabs.addTab(ReplayTab(), "Replay")
    win.setCentralWidget(tabs)
    win.resize(1100, 800)
    win.show()
    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())


if __name__ == "__main__":
    main()