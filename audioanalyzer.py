import sys
import numpy as np
import soundfile as sf
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from scipy.fft import rfft, rfftfreq

class AudioAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Analyzer")
        self.resize(1400, 900)

        # State
        self.audio_data = None
        self.sample_rate = None
        self.duration = None
        self.last_selection = None
        self.fft_size = 512
        self.freq_scale = "Linear"
        self.octave_smoothing_fraction = 3

        # Central widget & layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Figure
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.ax_wave = self.fig.add_subplot(2, 1, 1)
        self.ax_spec = self.fig.add_subplot(2, 1, 2)

        # Adjust spacing dynamically based on window size
        self.update_subplot_spacing()
        self.canvas.mpl_connect('resize_event', lambda event: self.update_subplot_spacing())

        self.span = None

        # Menu
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        open_action = QtWidgets.QAction("Open File", self)
        open_action.triggered.connect(self.load_wav)
        file_menu.addAction(open_action)

        # Control menu
        control_menu = menu.addMenu("Controls")

        # FFT Size
        self.fft_combo_menu = QtWidgets.QMenu("FFT Size", self)
        fft_group = QtWidgets.QActionGroup(self)  # Exclusive group
        fft_group.setExclusive(True)
        for val in [256, 512, 1024, 2048, 4096, 8192, 16384]:
            act = QtWidgets.QAction(str(val), self, checkable=True)
            act.triggered.connect(lambda checked, v=val: self.set_fft_size(v))
            self.fft_combo_menu.addAction(act)
            fft_group.addAction(act)
            # Default FFT Size
            if val == 512:
                act.setChecked(True)
        control_menu.addMenu(self.fft_combo_menu)
        
        # Octave Smoothing
        self.smooth_menu = QtWidgets.QMenu("Octave Smoothing", self)
        smooth_group = QtWidgets.QActionGroup(self)
        smooth_group.setExclusive(True)
        for val in ["Off", "1/2", "1/3", "1/4", "1/6"]:
            act = QtWidgets.QAction(val, self, checkable=True)
            act.triggered.connect(lambda checked, v=val: self.set_smooth_fraction(v))
            self.smooth_menu.addAction(act)
            smooth_group.addAction(act)
            # Default smoothing
            if val == "Off":
                act.setChecked(True)
        control_menu.addMenu(self.smooth_menu)
        
        # Frequency Scale
        self.scale_menu = QtWidgets.QMenu("Frequency Scale", self)
        scale_group = QtWidgets.QActionGroup(self)
        scale_group.setExclusive(True)
        for val in ["Linear", "Logarithmic"]:
            act = QtWidgets.QAction(val, self, checkable=True)
            act.triggered.connect(lambda checked, v=val: self.set_freq_scale(v))
            self.scale_menu.addAction(act)
            scale_group.addAction(act)
            # Default scale
            if val == "Linear":
                act.setChecked(True)
        control_menu.addMenu(self.scale_menu)

        # Zoom submenu
        zoom_menu = QtWidgets.QMenu("Zoom", self)
        
        # Zoom In (+)
        zoom_in_btn = QtWidgets.QPushButton("+")
        zoom_in_btn.setFixedWidth(30)  # make it small
        zoom_in_btn.clicked.connect(lambda: self.zoom_time_axis(0.8))
        zoom_in_action = QtWidgets.QWidgetAction(self)
        zoom_in_action.setDefaultWidget(zoom_in_btn)
        zoom_menu.addAction(zoom_in_action)
        
        # Zoom Out (-)
        zoom_out_btn = QtWidgets.QPushButton("-")
        zoom_out_btn.setFixedWidth(30)  # make it small
        zoom_out_btn.clicked.connect(lambda: self.zoom_time_axis(1.25))
        zoom_out_action = QtWidgets.QWidgetAction(self)
        zoom_out_action.setDefaultWidget(zoom_out_btn)
        zoom_menu.addAction(zoom_out_action)
        
        control_menu.addMenu(zoom_menu)
        
        # Reset View
        reset_action = QtWidgets.QAction("Reset View", self)
        reset_action.triggered.connect(self.reset_time_view)
        control_menu.addAction(reset_action)


    def update_subplot_spacing(self):
        width, height = self.canvas.get_width_height()
        top = 0.93
        bottom = 0.08
        left = 0.08
        right = 0.98
        hspace = max(0.25, 0.35 * (900/height))  # Scale spacing with window height
        self.fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace)
        self.canvas.draw_idle()
        
    def set_fft_size(self, val):
        self.fft_size = val
        self.update_spectrum_for_last_selection()
        
    def set_smooth_fraction(self, val):
        self.octave_smoothing_fraction = None if val == "Off" else int(val.split('/')[1])
        self.update_spectrum_for_last_selection()
        
    def set_freq_scale(self, val):
        self.freq_scale = val
        self.update_spectrum_for_last_selection()

    def zoom_time_axis(self, factor):
        if self.audio_data is None or len(self.audio_data) == 0:
            return
        x0, x1 = self.ax_wave.get_xlim()
        center = 0.5 * (x0 + x1)
        half_width = 0.5 * (x1 - x0) * factor
        new_left = max(0.0, center - half_width)
        new_right = min(self.duration, center + half_width)
        if new_right - new_left < 1.0 / max(self.sample_rate, 1):
            return
        self.ax_wave.set_xlim(new_left, new_right)
        self.canvas.draw_idle()

    def reset_time_view(self):
        if self.audio_data is None or len(self.audio_data) == 0:
            return
        self.ax_wave.set_xlim(0, self.duration)
        self.canvas.draw_idle()

    def load_wav(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open WAV File", "", "WAV Files (*.wav)")
        if not file_path:
            return
        data, sr = sf.read(file_path, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        self.audio_data = data
        self.sample_rate = int(sr)
        self.duration = len(data) / float(sr)
        self.last_selection = None
        self.plot_waveform_initial()
        self.clear_spectrum()

    def plot_waveform_initial(self):
        self.ax_wave.clear()
        t = np.arange(len(self.audio_data)) / self.sample_rate
        self.ax_wave.plot(t, self.audio_data, lw=0.8)
        self.ax_wave.set_title("Waveform")
        self.ax_wave.set_xlabel("Time [s]")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.set_xlim(0, self.duration)
        if self.span:
            try:
                self.span.disconnect_events()
            except Exception:
                pass
        self.span = SpanSelector(self.ax_wave, onselect=self.on_select, direction='horizontal', useblit=True, props=dict(alpha=0.3, facecolor='orange'), interactive=True)
        self.canvas.draw_idle()

    def on_select(self, xmin, xmax):
        if xmin == xmax or self.audio_data is None or len(self.audio_data) == 0:
            return
        xmin, xmax = sorted([max(0, xmin), min(self.duration, xmax)])
        self.last_selection = (xmin, xmax)
        self.update_spectrum_for_last_selection()

    def update_spectrum_for_last_selection(self):
        if self.audio_data is None or len(self.audio_data) == 0 or not self.last_selection:
            return
        xmin, xmax = self.last_selection
        segment = self.audio_data[int(xmin*self.sample_rate):int(xmax*self.sample_rate)]
        self.plot_magnitude_spectrum(segment)

    def clear_spectrum(self):
        self.ax_spec.clear()
        self.ax_spec.set_title("Magnitude Spectrum")
        self.ax_spec.set_xlabel("Frequency [Hz]")
        self.ax_spec.set_ylabel("Magnitude [dB]")
        self.ax_spec.grid(True, which='both')
        self.canvas.draw_idle()

    def plot_magnitude_spectrum(self, segment):
        N = min(len(segment), self.fft_size)
        x = np.pad(segment, (0, max(0, N - len(segment))))[:N] * np.hanning(N)
        spectrum = rfft(x, n=N)
        freqs = rfftfreq(N, 1/self.sample_rate)
        mag_db = 20*np.log10(np.maximum(np.abs(spectrum), 1e-12))
        if self.octave_smoothing_fraction:
            mag_db = self.fractional_octave_smooth_db(freqs, mag_db, fraction=self.octave_smoothing_fraction)
        self.ax_spec.clear()
        self.ax_spec.plot(freqs, mag_db)
        self.ax_spec.set_title("Magnitude Spectrum")
        self.ax_spec.set_xlabel("Frequency [Hz]")
        self.ax_spec.set_ylabel("Magnitude [dB]")
        self.ax_spec.grid(True, which='both', ls=':')
        self.ax_spec.set_xscale('log' if self.freq_scale=='Logarithmic' else 'linear')
        self.ax_spec.set_xlim(1 if self.freq_scale=='Logarithmic' else 0, self.sample_rate/2)
        self.canvas.draw_idle()

    @staticmethod
    def fractional_octave_smooth_db(freqs, db_vals, fraction=3):
        freqs, db_vals = np.asarray(freqs), np.asarray(db_vals)
        out = np.copy(db_vals)
        n = len(freqs)
        for i in range(n):
            f = freqs[i]
            if f <= 0: continue
            band = 2 ** (1 / (2*fraction))
            lo, hi = np.searchsorted(freqs, f/band), np.searchsorted(freqs, f*band)
            if hi > lo:
                out[i] = np.mean(db_vals[lo:hi])
        return out

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = AudioAnalyzer()
    win.show()
    sys.exit(app.exec_())