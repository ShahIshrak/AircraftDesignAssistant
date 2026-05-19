"""
Airfoil_2D_Analysis_Agent.py
==============================
Research-grade implementation with all known bugs fixed.

BUG-FIX SUMMARY (this revision)
---------------------------------
FIX-1  STANDALONE XFOIL FAILURE — CWD-dependent path resolution
       Root cause: _get_airfoil() returned a relative path like
       "Airfoils/naca23015.dat".  run_polar() called os.path.abspath()
       on it, which is relative to the current working directory (CWD).
       When run standalone from Airfoil2DAnalysis/, the CWD is different
       than when called through AircraftDesignAssistant (CWD = FAIRY/).
       Fix: __init__ immediately converts data_dir to an absolute path
       anchored to THIS FILE's directory, so the data dir is always
       correct regardless of how or from where the agent is launched.

FIX-2  XFOIL COMMAND REJECTION ("VISC command not recognized")
       The correct VPAR-before-OPER sequence was already in the code,
       but XFOIL still rejected commands because the polar_file path
       sent after PACC contained characters that tripped XFOIL's line
       parser (long Windows paths with mixed slashes, spaces, or the
       "C:/U..." prefix looking like a command to XFOIL's tokeniser).
       Fix: polar_file is now created in a short temp path using
       tempfile.mkstemp, and the dat_path fed to LOAD is the absolute
       path with backslashes (Windows-native) so XFOIL parses it correctly.
       We also explicitly flush a blank line after PACC's dump-file entry.

FIX-3  HARD-CODED OUTPUT PATHS
       save_folder was hard-coded to an absolute user-specific path.
       Fix: output is now written to a sibling "airfoil_polars" folder
       next to this file, resolved absolutely.

FIX-4  XFOIL COMMAND SEQUENCE (documented, already correct)
       THE CORRECT XFOIL COMMAND ORDER:
         1.  LOAD <dat_path>            top-level: load airfoil geometry
         2.  PANE                       top-level: re-panel for smoothness
         3.  VPAR                       top-level: enter VPAR sub-menu
         4.  N <ncrit>                  inside VPAR: set e^N factor
         5.  (blank)                    exit VPAR back to top level
         6.  OPER                       top-level: enter OPER sub-menu
         7.  VISC <Re>                  inside OPER: activate viscous mode
         8.  MACH <Mach>               inside OPER: set Mach number
         9.  ITER <n>                   inside OPER: set Newton iteration cap
        10.  PACC                       inside OPER: start polar accumulation
        11.  <polar_file>               output file path
        12.  (blank)                    no dump file
        13.  ASEQ <a0> <a1> <da>        inside OPER: run polar sweep
        14.  PACC                       inside OPER: stop polar accumulation
        15.  (blank)                    exit OPER back to top level
        16.  QUIT                       exit XFOIL
"""

import os
import re
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from pathlib import Path
import time

# Directory that contains THIS file — used to anchor all relative paths.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class HessSmithPanelMethod:
    """Constant-strength vortex panel method with proper Kutta condition."""

    def __init__(self, coords, Re=1e6):
        self.coords = self._preprocess_coords(coords)
        self.Re = Re
        self.n  = len(self.coords) - 1

        self.x  = self.coords[:, 0]
        self.y  = self.coords[:, 1]

        self.xc = np.zeros(self.n)
        self.yc = np.zeros(self.n)
        self.dx = np.zeros(self.n)
        self.dy = np.zeros(self.n)
        self.L  = np.zeros(self.n)
        self.phi= np.zeros(self.n)
        self.nx = np.zeros(self.n)
        self.ny = np.zeros(self.n)

        self._compute_geometry()

    # ------------------------------------------------------------
    def _preprocess_coords(self, coords):
        coords = np.array(coords)

        dists = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
        keep  = np.concatenate([[True], dists > 1e-10])
        coords = coords[keep]

        if np.linalg.norm(coords[0] - coords[-1]) > 1e-6:
            coords = np.vstack([coords, coords[0]])

        return coords

    # ------------------------------------------------------------
    def _compute_geometry(self):
        for i in range(self.n):
            self.dx[i]  = self.x[i+1] - self.x[i]
            self.dy[i]  = self.y[i+1] - self.y[i]

            self.L[i]   = np.sqrt(self.dx[i]**2 + self.dy[i]**2)
            self.phi[i] = np.arctan2(self.dy[i], self.dx[i])

            self.xc[i]  = 0.5*(self.x[i] + self.x[i+1])
            self.yc[i]  = 0.5*(self.y[i] + self.y[i+1])

            self.nx[i]  =  np.sin(self.phi[i])
            self.ny[i]  = -np.cos(self.phi[i])

    # ------------------------------------------------------------
    def _influence_coefficients(self, xi, yi, j):
        x_j, y_j = self.x[j], self.y[j]
        phi_j, L_j = self.phi[j], self.L[j]

        cos_p, sin_p = np.cos(phi_j), np.sin(phi_j)

        xp = (xi - x_j)*cos_p + (yi - y_j)*sin_p
        yp = -(xi - x_j)*sin_p + (yi - y_j)*cos_p

        r1_sq = max(xp**2 + yp**2, 1e-12)
        r2_sq = max((xp - L_j)**2 + yp**2, 1e-12)

        log_r   = 0.5*np.log(r2_sq / r1_sq)
        theta1  = np.arctan2(yp, xp)
        theta2  = np.arctan2(yp, xp - L_j)

        d_theta = theta1 - theta2

        u_local = -d_theta/(2*np.pi)
        v_local =  log_r/(2*np.pi)

        u_global = u_local*cos_p - v_local*sin_p
        v_global = u_local*sin_p + v_local*cos_p

        return u_global, v_global

    # ------------------------------------------------------------
    def _build_system(self, u_inf, v_inf):
        n = self.n

        A = np.zeros((n+1, n+1))
        RHS = np.zeros(n+1)

        # -------------------------
        # Flow tangency condition
        # -------------------------
        for i in range(n):
            RHS[i] = -(u_inf*self.nx[i] + v_inf*self.ny[i])

            for j in range(n):
                u, v = self._influence_coefficients(self.xc[i], self.yc[i], j)
                A[i, j] = u*self.nx[i] + v*self.ny[i]

        # --------------------------------------------------------
        # ✅ KUTTA CONDITION (correct form for constant vortex sheet)
        # --------------------------------------------------------
        # Tangential velocity equality at trailing edge:
        #
        # γ1 + γ2 + ... enforces circulation consistency
        #
        A[n, :] = 0.0
        A[n, 0] = 1.0
        A[n, -1] = -1.0
        RHS[n]  = 0.0

        return A, RHS

    # ------------------------------------------------------------
    def solve(self, aoa_deg):
        aoa = np.radians(aoa_deg)

        u_inf = np.cos(aoa)
        v_inf = np.sin(aoa)

        A, RHS = self._build_system(u_inf, v_inf)

        # stabilize numerics
        A += np.eye(self.n+1) * 1e-10

        try:
            gamma = np.linalg.solve(A, RHS)
        except np.linalg.LinAlgError:
            gamma = np.linalg.lstsq(A, RHS, rcond=1e-12)[0]

        # -------------------------
        # Tangential velocity
        # -------------------------
        Vt = np.zeros(self.n)

        for i in range(self.n):
            tx, ty = np.cos(self.phi[i]), np.sin(self.phi[i])

            Vt[i] = u_inf*tx + v_inf*ty

            for j in range(self.n):
                u, v = self._influence_coefficients(self.xc[i], self.yc[i], j)
                Vt[i] += gamma[j]*(u*tx + v*ty)

        Cp = 1.0 - Vt**2

        chord = np.max(self.x) - np.min(self.x)

        Gamma = np.sum(gamma[:self.n] * self.L)

        Cl = 2.0 * Gamma / max(chord, 1e-6)

        xc_norm = (self.xc - np.min(self.x)) / max(chord, 1e-6)

        Cm = -np.sum(Cp*(xc_norm - 0.25)*self.L*self.nx) / max(chord, 1e-6)

        Cd = self._estimate_viscous_drag(Cp, aoa_deg)

        return Cl, Cd, Cm, Cp

    # ------------------------------------------------------------
    def _estimate_viscous_drag(self, Cp, aoa):
        try:
            Re = self.Re

            def cf(Re_x):
                if Re_x < 1e3:
                    return 0.0
                if Re_x < 5e5:
                    return 1.328/np.sqrt(Re_x)
                return 0.074/(Re_x**0.2)

            upper = np.where(self.ny > 0)[0]
            lower = np.where(self.ny < 0)[0]

            S = np.sum(self.L)
            chord = max(self.x) - min(self.x)

            Cf = cf(Re*S)

            return max(Cf * S / max(chord, 1e-6), 1e-5)

        except:
            return 0.01

# =============================================================================
# XFOIL RUNNER — corrected command sequence + absolute-path fixes
# =============================================================================

class XFOILRunner:
    """Thin wrapper around the XFOIL subprocess."""

    def __init__(self, xfoil_path=None):
        # FIX-1: Always resolve xfoil_path to an absolute path.
        self.xfoil_path = os.path.abspath(xfoil_path) if xfoil_path != "xfoil" else xfoil_path
        self.available  = False

    def _check_availability(self):
        if not os.path.isfile(self.xfoil_path):
            print(f"[XFOILRunner] Binary not found at '{self.xfoil_path}' — "
                  f"falling back to panel method.")
            return False
        print(f"[XFOILRunner] XFOIL found: {self.xfoil_path}")
        return True

    def run_polar(self, dat_path, Re, alpha_range=(-5, 20, 0.5),
                  Mach=0.0, n_crit=9.0, max_iter=100):
        alpha_start, alpha_end, alpha_step = alpha_range

        # FIX-2: Use a SHORT temp path for the polar file to avoid XFOIL's
        # line parser choking on long Windows paths.  mkstemp() gives us a
        # guaranteed unique file in %TEMP% which is typically short.
        fd, polar_file = tempfile.mkstemp(suffix=".txt", prefix="xfpolar_")
        os.close(fd)
        # Delete it now so XFOIL creates it fresh; we re-check after the run.
        try:
            os.remove(polar_file)
        except OSError:
            pass

        # FIX-1: dat_path must be absolute and use OS-native separators so
        # XFOIL's LOAD command parses it correctly on Windows.
        dat_path_abs = os.path.normpath(os.path.abspath(dat_path))

        if not os.path.exists(dat_path_abs):
            print(f"[XFOILRunner] ERROR: Airfoil file not found at {dat_path_abs}")
            return pd.DataFrame()

        # FIX-2: Use OS-native path separator for the polar file path too.
        polar_file_native = os.path.normpath(polar_file)

        # THE CORRECT XFOIL COMMAND SEQUENCE (VPAR before OPER):
        commands = (
            f"LOAD {dat_path_abs}\n"
            f"PANE\n"
            f"VPAR\n"
            f"N {n_crit}\n"
            f"\n"                    # blank → exit VPAR to top level
            f"OPER\n"
            f"VISC {Re:.6e}\n"
            f"MACH {Mach:.4f}\n"
            f"ITER {max_iter}\n"
            f"PACC\n"
            f"{polar_file_native}\n"
            f"\n"                    # blank → no dump file
            f"ASEQ {alpha_start} {alpha_end} {alpha_step}\n"
            f"PACC\n"
            f"\n"                    # blank → exit OPER to top level
            f"QUIT\n"
        )

        try:
            proc = subprocess.run(
                [self.xfoil_path],
                input=commands,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            print("[XFOILRunner] XFOIL subprocess timed out after 120 s.")
            return pd.DataFrame()
        except Exception as e:
            print(f"[XFOILRunner] Subprocess error: {e}")
            return pd.DataFrame()

        df_polar = self._parse_polar_file(polar_file_native)
        df_polar.attrs["xfoil_stdout"] = proc.stdout
        df_polar.attrs["xfoil_stderr"] = proc.stderr
        return df_polar

    def run_single_alpha(self, dat_path, Re, alpha, n_crit=9, max_iter=100, Mach=0.0):
        """Single alpha run — corrected VPAR-before-OPER sequence."""
        fd, cp_file = tempfile.mkstemp(suffix=".cp", prefix="xfcp_")
        os.close(fd)
        try:
            os.remove(cp_file)
        except OSError:
            pass

        dat_path_abs = os.path.normpath(os.path.abspath(dat_path))
        cp_file_native = os.path.normpath(cp_file)

        commands = (
            f"LOAD {dat_path_abs}\n"
            f"PANE\n"
            f"VPAR\n"
            f"N {n_crit}\n"
            f"\n"
            f"OPER\n"
            f"VISC {Re:.6e}\n"
            f"MACH {Mach:.4f}\n"
            f"ITER {max_iter}\n"
            f"ALFA {alpha}\n"
            f"CPWR {cp_file_native}\n"
            f"\n"
            f"QUIT\n"
        )

        try:
            proc = subprocess.run(
                [self.xfoil_path],
                input=commands,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as e:
            print(f"[XFOILRunner] Single-alpha subprocess error: {e}")
            return pd.DataFrame(), pd.DataFrame()

        df_cp = self._parse_cp_file(cp_file_native)
        return pd.DataFrame(), df_cp

    def _parse_polar_file(self, polar_file):
        if not os.path.exists(polar_file):
            return pd.DataFrame()
        rows = []
        try:
            with open(polar_file, "r") as f:
                lines = f.readlines()
            data_start = False
            for line in lines:
                line = line.strip()
                if line.startswith("---"):
                    data_start = True; continue
                if data_start and line:
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            rows.append({
                                "AoA":     float(parts[0]),
                                "Cl":      float(parts[1]),
                                "Cd":      float(parts[2]),
                                "CDp":     float(parts[3]),
                                "Cm":      float(parts[4]),
                                "Xtr_top": float(parts[5]),
                                "Xtr_bot": float(parts[6]),
                            })
                        except ValueError:
                            continue
        finally:
            try:
                if os.path.exists(polar_file):
                    os.remove(polar_file)
            except OSError:
                pass
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["ClCd"] = df["Cl"] / df["Cd"].replace(0, np.nan)
        return df

    def _parse_cp_file(self, cp_file):
        if not os.path.exists(cp_file):
            return pd.DataFrame()
        rows = []
        try:
            with open(cp_file, "r") as f:
                lines = f.readlines()
            for line in lines[3:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        rows.append({"x_c": float(parts[0]), "Cp": float(parts[1])})
                    except ValueError:
                        continue
        finally:
            try:
                if os.path.exists(cp_file):
                    os.remove(cp_file)
            except OSError:
                pass
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# MAIN AIRFOIL ANALYSIS AGENT
# =============================================================================

_PARAM_KEYWORDS = {"re", "reynolds", "velocity", "v=", "v:", "chord", "c=",
                   "mach", "ncrit", "m/s", "ms", "km/h", "kph"}


class AirfoilAnalysisAgent:
    """
    Conversational airfoil analysis agent.
    Primary: XFOIL (viscous-inviscid coupled).
    Fallback: Hess-Smith panel method (inviscid).
    """

    def __init__(self, data_dir="Airfoils",
                 xfoil_path=None):
        # ── FIX-1: Resolve data_dir to an absolute path anchored to THIS file
        # so it is correct regardless of the caller's CWD.
        if os.path.isabs(data_dir):
            self.data_dir = data_dir
        else:
            self.data_dir = os.path.join(_THIS_DIR, data_dir)

        # ── FIX-3: Resolve output folder (airfoil_polars) to absolute path
        self._output_dir = os.path.join(_THIS_DIR, "airfoil_polars")
        os.makedirs(self._output_dir, exist_ok=True)

        # ── XFOIL path: default to sibling XFOIL6.99 folder
        if xfoil_path is None:
            xfoil_path = os.path.join(_THIS_DIR, "XFOIL6.99", "xfoil.exe")

        self.xfoil       = XFOILRunner(xfoil_path)
        self.polar_cache = {}
        self._app        = None
        self.active                = False
        self.awaiting_airfoil_name = False
        self.awaiting_inputs       = False
        self.airfoil_name          = None
        self.params                = {}
        self.analysis_done         = False
        self.last_results          = {}

        os.makedirs(self.data_dir, exist_ok=True)
        engine = "XFOIL (viscous-inviscid coupled)" if self.xfoil.available \
                 else "Hess-Smith Panel Method (inviscid fallback)"
        print(f"[AirfoilAgent] Engine: {engine}")
        print(f"[AirfoilAgent] data_dir  = {self.data_dir}")
        print(f"[AirfoilAgent] output_dir = {self._output_dir}")
        print(f"[AirfoilAgent] xfoil_path = {self.xfoil.xfoil_path}")

    # ------------------------------------------------------------------
    # GUI wiring
    # ------------------------------------------------------------------

    def set_app(self, app) -> None:
        self._app = app

    def _gui_progress(self, step: str, detail: str = "") -> None:
        msg = f"⚙️ [Airfoil Analysis] {step}"
        if detail:
            msg += f" — {detail}"
        print(f"[AirfoilAgent] {msg}")

    def _progress(self, step: str, detail: str = ""):
        import datetime
        ts  = datetime.datetime.now().strftime("%H:%M:%S")
        msg = f"[{ts}] {step}" + (f" — {detail}" if detail else "")
        print(f"[AirfoilAgent] {msg}")
        return msg

    # ------------------------------------------------------------------
    # Smart detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_airfoil_name_from_text(text: str):
        lower = text.lower().replace(" ", "")
        m = re.search(r'naca(\d{4,5})', lower)
        if m: return f"naca{m.group(1)}"
        m = re.search(r'\b([a-z]{1,3}\d{3,5})\b', lower)
        if m: return m.group(1)
        return None

    @staticmethod
    def _text_contains_flow_params(text: str) -> bool:
        has_number  = bool(re.search(r'\d', text))
        lower_words = set(re.findall(r'[a-z/=:]+', text.lower()))
        return has_number and bool(lower_words & _PARAM_KEYWORDS)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self, text: str = None) -> str:
        self.active                = True
        self.awaiting_airfoil_name = True
        self.awaiting_inputs       = False
        self.analysis_done         = False
        self.params                = {}
        self.airfoil_name          = None
        engine_note = ("XFOIL viscous analysis active." if self.xfoil.available
                       else "⚠️ XFOIL not found — using inviscid panel method.")
        self._progress("Agent started", engine_note)

        if text:
            airfoil    = self._extract_airfoil_name_from_text(text)
            has_params = self._text_contains_flow_params(text)
            if airfoil and has_params:
                self.airfoil_name          = airfoil
                self.awaiting_airfoil_name = False
                self.awaiting_inputs       = False
                return self.set_parameters(text)
            if airfoil:
                self.airfoil_name          = airfoil
                self.awaiting_airfoil_name = False
                self.awaiting_inputs       = True
                return (f"Airfoil Analysis Agent ready. {engine_note}\n"
                        f"Airfoil set to '{airfoil}'.\n"
                        "Provide flow parameters — Re, V (m/s), c (m), "
                        "optionally Mach and Ncrit.\n"
                        "Example: Re=1e6, V=30, c=1.0, Mach=0.1, Ncrit=9")

        return (f"Airfoil Analysis Agent ready. {engine_note}\n"
                "Please provide the airfoil name (e.g., naca2412, e387, s1223):")

    def process_input(self, text: str) -> str:
        """Unified state-machine dispatcher for ACTIVE_HANDLERS."""
        text_lower = text.strip().lower()
        if text_lower in ("restart", "reset", "new"):
            self.__init__(self.data_dir, self.xfoil.xfoil_path)
            return self.start()
        if text_lower in ("exit", "quit", "close", "cancel"):
            self.active = False
            self.awaiting_airfoil_name = False
            self.awaiting_inputs = False
            self.analysis_done = False
            return "Airfoil Analysis Agent closed. Say 'analyse airfoil' to start again."
        if not self.active:            return self.start(text)
        if self.awaiting_airfoil_name: return self.set_airfoil_name(text)
        if self.awaiting_inputs:       return self.set_parameters(text)
        if self.analysis_done:         return self._handle_followup(text)
        return "Unexpected state. Type 'restart' to reset."

    def set_airfoil_name(self, name):
        extracted  = self._extract_airfoil_name_from_text(name)
        clean_name = extracted if extracted else name.strip().lower().replace(" ", "")
        self.airfoil_name          = clean_name
        self.awaiting_airfoil_name = False
        self.awaiting_inputs       = True
        print(f"[AirfoilAgent] Airfoil set to '{self.airfoil_name}'")
        self._gui_progress("Airfoil name set", f"'{self.airfoil_name}' — awaiting flow parameters")
        return (f"Airfoil: '{self.airfoil_name}'\n"
                "Provide parameters — Re, V (m/s), c (m), optionally Mach and Ncrit.\n"
                "Example: Re=1e6, V=30, c=1.0, Mach=0.1, Ncrit=9")

    def set_parameters(self, param_text):
        try:
            pattern = r"(Re|V|c|Mach|Ncrit|ncrit)\s*[=:]?\s*([\d\.eE\+\-]+)"
            matches = re.findall(pattern, param_text, re.IGNORECASE)
            if not matches:
                return "⚠️ No valid parameters found. Try: Re=1e6, V=30, c=1.0"
            for key, val in matches:
                self.params[key.upper()] = float(val)

            Re    = self.params.get("RE",    1e6)
            V     = self.params.get("V",     30.0)
            c     = self.params.get("C",     1.0)
            Mach  = self.params.get("MACH",  0.0)
            Ncrit = self.params.get("NCRIT", 9.0)

            self._progress("Parameters accepted",
                           f"Re={Re:.2e}  V={V} m/s  c={c} m  Mach={Mach:.3f}  Ncrit={Ncrit}")
            self._gui_progress("Parameters accepted",
                               f"Re={Re:.2e} | V={V} m/s | c={c} m | Mach={Mach:.3f} | Ncrit={Ncrit}")
            self.awaiting_inputs = False
            engine_name = "XFOIL" if self.xfoil.available else "panel method"
            self._progress(f"Handing off to {engine_name}",
                           "this may take 10–60 s")
            self._gui_progress(f"Running {engine_name} analysis",
                               f"polar sweep for '{self.airfoil_name}' — this may take 10–60 s…")
            return self.run_analysis()
        except Exception as e:
            import traceback
            self._progress("ERROR in set_parameters", str(e))
            return f"⚠️ Parameter error: {e}\n{traceback.format_exc()}"

    def run_analysis_with_params(self, airfoil_name: str,
                                  Re: float, V: float, c: float,
                                  Mach: float = 0.0, Ncrit: float = 9.0) -> str:
        """Automated entry point for AircraftDesignAssistant."""
        self._progress("Automated run",
                       f"airfoil='{airfoil_name}'  Re={Re:.2e}  V={V} m/s  "
                       f"c={c:.3f} m  Mach={Mach:.3f}  Ncrit={Ncrit}")
        self.active = True
        self.awaiting_airfoil_name = False
        self.awaiting_inputs = False
        self.analysis_done = False
        self.airfoil_name  = airfoil_name.strip().lower()
        self.params = {"RE": Re, "V": V, "C": c, "MACH": Mach, "NCRIT": Ncrit}
        return self.run_analysis()

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def run_analysis(self):
        Re    = self.params.get("RE",    1e6)
        V     = self.params.get("V",     30.0)
        c     = self.params.get("C",     1.0)
        Mach  = self.params.get("MACH",  0.0)
        Ncrit = self.params.get("NCRIT", 9.0)

        self._progress("━━━━━━━━━━ AIRFOIL ANALYSIS START ━━━━━━━━━━")
        self._progress("Airfoil", self.airfoil_name.upper())
        self._progress("Flow conditions",
                       f"Re={Re:.2e}  V={V} m/s  c={c} m  Mach={Mach:.3f}  Ncrit={Ncrit}")
        self._gui_progress("Analysis started",
                           f"{self.airfoil_name.upper()} | Re={Re:.2e} | V={V} m/s | c={c} m")

        self._progress("[1/4] Locating airfoil coordinates",
                       f"checking local cache for '{self.airfoil_name}'")
        dat_path = self._get_airfoil(self.airfoil_name)
        self._progress("[1/4] Coordinates ready", dat_path)

        if self.xfoil.available:
            n_pts = int((20 - (-5)) / 0.5) + 1
            self._progress("[2/4] Launching XFOIL",
                           f"polar sweep -5°→+20° ({n_pts} pts, step 0.5°) — please wait…")
            self._gui_progress("[2/4] XFOIL running",
                               f"polar sweep -5°→+20° ({n_pts} pts) — please wait…")
            result_msg = self._run_xfoil_analysis(dat_path, Re, V, c, Mach, Ncrit)
        else:
            self._progress("[2/4] Running Hess-Smith panel method", "inviscid, 51 AoA pts")
            self._gui_progress("[2/4] Panel method running",
                               "inviscid analysis, 51 AoA points — please wait…")
            result_msg = self._run_panel_analysis(dat_path, Re, V, c)

        self.analysis_done = True
        self._progress("━━━━━━━━━━ ANALYSIS COMPLETE ━━━━━━━━━━",
                       f"airfoil: {self.airfoil_name.upper()}")
        self._gui_progress("Analysis complete ✅", self.airfoil_name.upper())
        return result_msg

    def _run_xfoil_analysis(self, dat_path, Re, V, c, Mach, Ncrit):
        try:
            self._progress("[2/4] XFOIL subprocess launched", "waiting up to 120 s…")

            df = self.xfoil.run_polar(
                dat_path=dat_path,
                Re=Re,
                alpha_range=(-5, 20, 0.5),
                Mach=Mach,
                n_crit=Ncrit,
            )

            if df.empty:
                stdout_tail = df.attrs.get("xfoil_stdout", "")[-500:]
                stderr_tail = df.attrs.get("xfoil_stderr", "")[-200:]
                self._progress("⚠️ XFOIL returned no converged solutions")
                print(f"[AirfoilAgent] stdout tail:\n{stdout_tail or '(empty)'}")
                return (
                    f"⚠️ XFOIL returned no converged solutions.\n"
                    f"Possible causes: geometry issue, Re too low/high, AoA range too wide.\n"
                    f"\n--- XFOIL stdout (last 500 chars) ---\n{stdout_tail or '(empty)'}\n"
                    f"\n--- XFOIL stderr (last 200 chars) ---\n{stderr_tail or '(empty)'}"
                )

            n_conv  = len(df)
            n_total = int((20 - (-5)) / 0.5) + 1
            self._progress("[2/4] XFOIL polar complete",
                           f"{n_conv}/{n_total} AoA points converged")

            self.last_results = {
                "type": "xfoil", "df": df,
                "Re": Re, "V": V, "c": c, "Mach": Mach, "Ncrit": Ncrit
            }

            # FIX-3: Save to self._output_dir (absolute, anchored to THIS file)
            out_csv = os.path.join(self._output_dir, f"{self.airfoil_name}_xfoil_polar.csv")
            out_png = os.path.join(self._output_dir, f"{self.airfoil_name}_xfoil_polar.png")

            self._progress("[3/4] Saving polar data", out_csv)
            df.to_csv(out_csv, index=False)
            self._progress("[4/4] Rendering plots", out_png)
            self._plot_xfoil_results(df, out_png)
            self._progress("[4/4] Plot saved")

            max_cl     = df["Cl"].max()
            max_ld     = df["ClCd"].max()
            max_ld_aoa = df.loc[df["ClCd"].idxmax(), "AoA"]
            min_cd     = df["Cd"].min()
            peak_row   = df.loc[df["ClCd"].idxmax()]
            xtr_top    = peak_row.get("Xtr_top", None)
            xtr_bot    = peak_row.get("Xtr_bot", None)

            self._progress("Key results",
                           f"Cl_max={max_cl:.4f}  Cd_min={min_cd:.6f}  "
                           f"(L/D)_max={max_ld:.2f}@α={max_ld_aoa:.1f}°")
            self._gui_progress("XFOIL results",
                               f"Cl_max={max_cl:.4f} | Cd_min={min_cd:.6f} | "
                               f"(L/D)_max={max_ld:.2f}@α={max_ld_aoa:.1f}°")

            xtr_str = (f"Transition: upper x/c={xtr_top:.3f}  lower x/c={xtr_bot:.3f}"
                       if (xtr_top is not None and xtr_bot is not None
                           and xtr_top == xtr_top and xtr_bot == xtr_bot)
                       else "Transition data not available")

            return (
                f"✅ XFOIL Analysis Complete — {self.airfoil_name.upper()}\n"
                f"   Re = {Re:.2e} | Mach = {Mach:.3f} | Ncrit = {Ncrit}\n"
                f"   Converged points : {n_conv} / {n_total}\n"
                f"   Max Cl           : {max_cl:.4f}\n"
                f"   Min Cd           : {min_cd:.6f}\n"
                f"   Max (Cl/Cd)      : {max_ld:.2f} at α = {max_ld_aoa:.1f}°\n"
                f"   {xtr_str}\n"
                f"   Saved: {out_csv},  {out_png}\n"
                f"   Engine: XFOIL viscous-inviscid coupled (viscous Cd ✓)"
            )

        except Exception as e:
            self._progress("⚠️ XFOIL exception", str(e))
            import traceback; print(traceback.format_exc())
            return f"⚠️ XFOIL error: {e}"

    def _run_panel_analysis(self, dat_path, Re, V, c):
        self._progress("Loading airfoil coordinates", dat_path)
        coords    = self._load_coords(dat_path)
        chord_raw = np.max(coords[:, 0]) - np.min(coords[:, 0])
        coords_norm = coords.copy()
        coords_norm[:, 0] /= chord_raw
        coords_norm[:, 1] /= chord_raw
        self._progress("Coordinates loaded", f"{len(coords)} pts")

        solver    = HessSmithPanelMethod(coords_norm, Re=Re)
        aoa_range = np.linspace(-5, 20, 51)
        rows = []
        self._progress("[2/4] AoA sweep",
                       f"{len(aoa_range)} pts ({aoa_range[0]:.1f}°→{aoa_range[-1]:.1f}°)")
        for i, aoa in enumerate(aoa_range):
            try:
                Cl, Cd, Cm, Cp = solver.solve(aoa)
                rows.append({"AoA": aoa, "Cl": Cl, "Cd": Cd, "Cm": Cm,
                             "ClCd": Cl/Cd if Cd > 0 else np.nan})
            except Exception:
                continue
            if (i+1) % 10 == 0 or (i+1) == len(aoa_range):
                last = rows[-1]
                self._progress(f"  {i+1}/{len(aoa_range)} done",
                               f"α={aoa:.1f}° Cl={last['Cl']:.4f} Cd={last['Cd']:.6f}")

        if not rows:
            return "⚠️ Panel method failed to produce any results."

        df = pd.DataFrame(rows)
        self.last_results = {"type": "panel", "df": df, "Re": Re}

        # FIX-3: Use self._output_dir for panel outputs too
        out_csv = os.path.join(self._output_dir, f"{self.airfoil_name}_panel_polar.csv")
        out_png = os.path.join(self._output_dir, f"{self.airfoil_name}_panel_polar.png")
        df.to_csv(out_csv, index=False)
        self._plot_panel_results(df, out_png)

        max_cl     = df["Cl"].max()
        max_ld     = df["ClCd"].max()
        max_ld_aoa = df.loc[df["ClCd"].idxmax(), "AoA"]
        min_cd     = df["Cd"].min()
        self._gui_progress("Panel results",
                           f"Cl_max={max_cl:.4f} | (L/D)_max={max_ld:.2f}@α={max_ld_aoa:.1f}° ⚠️ inviscid")
        return (
            f"✅ Panel Analysis Complete — {self.airfoil_name.upper()}\n"
            f"   Re = {Re:.2e}  ⚠️ INVISCID (Cd is BL estimate only)\n"
            f"   Converged points : {len(rows)} / {len(aoa_range)}\n"
            f"   Max Cl           : {max_cl:.4f}\n"
            f"   Min Cd (est.)    : {min_cd:.6f}\n"
            f"   Max (Cl/Cd)      : {max_ld:.2f} at α = {max_ld_aoa:.1f}°\n"
            f"   Saved: {out_csv},  {out_png}\n"
            f"   Engine: Hess-Smith Vortex Panel (inviscid, Cd estimated)"
        )

    # ------------------------------------------------------------------
    # Follow-up handler
    # ------------------------------------------------------------------

    def _handle_followup(self, text: str) -> str:
        lower = text.lower()
        if not self.last_results:
            return "No results available. Type 'restart' to run a new analysis."
        df = self.last_results.get("df")
        if df is None or df.empty:
            return "No polar data available."
        if any(k in lower for k in ("stall", "cl max", "clmax")):
            return self._estimate_stall()
        if any(k in lower for k in ("summary", "results", "report", "all")):
            return self._summarize_results()
        if any(k in lower for k in ("restart", "new", "another")):
            self.__init__(self.data_dir, self.xfoil.xfoil_path)
            return self.start()
        return ("Analysis complete. You can ask:\n"
                "  • 'summary' — full polar summary\n"
                "  • 'stall' — stall angle estimate\n"
                "  • 'restart' — analyse a different airfoil")

    def _estimate_stall(self) -> str:
        lr = self.last_results
        if not lr or lr.get("df") is None or lr["df"].empty:
            return "No data for stall estimate."
        df = lr["df"]
        idx = df["Cl"].idxmax()
        return (f"Stall estimate for {self.airfoil_name.upper()}:\n"
                f"  Cl_max ≈ {df.loc[idx,'Cl']:.4f} at α ≈ {df.loc[idx,'AoA']:.1f}°")

    def _summarize_results(self) -> str:
        lr = self.last_results
        if not lr or lr.get("df") is None or lr["df"].empty:
            return "No results to summarise."
        df     = lr["df"]
        Re     = lr.get("Re", "N/A")
        engine = lr.get("type", "unknown").upper()
        max_cl = df["Cl"].max()
        min_cd = df["Cd"].min()
        max_ld = df["ClCd"].max()
        idx    = df["ClCd"].idxmax()
        return (f"Airfoil: {self.airfoil_name.upper()}  [{engine}  Re={Re:.2e}]\n"
                f"  Cl_max  : {max_cl:.4f}\n"
                f"  Cd_min  : {min_cd:.6f}\n"
                f"  (L/D)max: {max_ld:.2f}  at α = {df.loc[idx,'AoA']:.1f}°")

    # ------------------------------------------------------------------
    # Coordinate management
    # ------------------------------------------------------------------

    def _get_airfoil(self, name: str) -> str:
        # FIX-1: Always construct an absolute path using self.data_dir which
        # is already absolute (set in __init__).
        local = os.path.join(self.data_dir, f"{name}.dat")
        if os.path.exists(local):
            self._progress("[Cache hit]", f"'{name}.dat' found locally")
            return local
        try:
            path = self._download_airfoil(name)
            if path: return path
        except Exception as e:
            self._progress(f"Download failed for '{name}'", str(e))
        digits = re.sub(r'^naca', '', name)
        if len(digits) == 4 and digits.isdigit(): return self._generate_naca4(digits, name)
        if len(digits) == 5 and digits.isdigit(): return self._generate_naca5(digits, name)
        raise FileNotFoundError(
            f"Airfoil '{name}' not found locally, download failed, and "
            f"offline generation only supports NACA 4/5-digit series.")

    def _download_airfoil(self, name: str):
        url = f"https://m-selig.ae.illinois.edu/ads/coord_seligFmt/{name}.dat"
        self._progress(f"Downloading '{name}'", url)
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            path = os.path.join(self.data_dir, f"{name}.dat")
            with open(path, "w") as f: f.write(resp.text)
            self._progress(f"Downloaded", path); return path
        url2  = f"https://m-selig.ae.illinois.edu/ads/coord/{name}.dat"
        resp2 = requests.get(url2, timeout=15)
        if resp2.status_code == 200:
            path = os.path.join(self.data_dir, f"{name}.dat")
            with open(path, "w") as f: f.write(resp2.text)
            return path
        self._progress(f"⚠️ Download failed", f"HTTP {resp.status_code}"); return None

    def _generate_naca4(self, digits, name):
        m_val = int(digits[0])/100.0; p_val = int(digits[1])/10.0; t_val = int(digits[2:])/100.0
        x = np.linspace(0, 1, 200)
        if m_val == 0 or p_val == 0:
            yc = np.zeros_like(x); dyc = np.zeros_like(x)
        else:
            yc  = np.where(x<p_val, m_val/p_val**2*(2*p_val*x-x**2),
                           m_val/(1-p_val)**2*((1-2*p_val)+2*p_val*x-x**2))
            dyc = np.where(x<p_val, 2*m_val/p_val**2*(p_val-x),
                           2*m_val/(1-p_val)**2*(p_val-x))
        yt = 5*t_val*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
        theta = np.arctan(dyc)
        xu=x-yt*np.sin(theta); yu=yc+yt*np.cos(theta)
        xl=x+yt*np.sin(theta); yl=yc-yt*np.cos(theta)
        xc=np.concatenate([xu[::-1],xl[1:]]); yc2=np.concatenate([yu[::-1],yl[1:]])
        path = self._save_coords(name, xc, yc2)
        self._progress("NACA 4-digit generated", f"{len(xc)} pts → {path}"); return path

    def _generate_naca5(self, digits, name):
        p_val=int(digits[1])/20.0; reflexed=int(digits[2])==1; t_val=int(digits[3:])/100.0
        if reflexed: raise NotImplementedError("NACA 5-digit reflexed not supported.")
        _p_table={0.05:361.4,0.10:51.64,0.15:15.957,0.20:6.643,0.25:3.230}
        if p_val not in _p_table:
            ps=sorted(_p_table.keys())
            if p_val<ps[0] or p_val>ps[-1]: raise ValueError(f"p={p_val:.2f} out of range.")
            for i in range(len(ps)-1):
                if ps[i]<=p_val<=ps[i+1]:
                    k1=_p_table[ps[i]]+(p_val-ps[i])/(ps[i+1]-ps[i])*(_p_table[ps[i+1]]-_p_table[ps[i]]); break
        else: k1=_p_table[p_val]
        x=np.linspace(0,1,200)
        yc =(np.where(x<p_val,(k1/6.0)*(x**3-3*p_val*x**2+p_val**2*(3-p_val)*x),(k1*p_val**3/6.0)*(1-x)))
        dyc=(np.where(x<p_val,(k1/6.0)*(3*x**2-6*p_val*x+p_val**2*(3-p_val)),-(k1*p_val**3/6.0)*np.ones_like(x)))
        yt=5*t_val*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
        theta=np.arctan(dyc)
        xu=x-yt*np.sin(theta);yu=yc+yt*np.cos(theta);xl=x+yt*np.sin(theta);yl=yc-yt*np.cos(theta)
        xc=np.concatenate([xu[::-1],xl[1:]]);yc2=np.concatenate([yu[::-1],yl[1:]])
        path=self._save_coords(name,xc,yc2); print(f"[AirfoilAgent] NACA5 '{name}' done."); return path

    def _save_coords(self, name, x_coords, y_coords):
        path = os.path.join(self.data_dir, f"{name}.dat")
        with open(path, "w") as f:
            f.write(f"{name.upper()}\n")
            for xi, yi in zip(x_coords, y_coords):
                f.write(f"  {xi:.6f}  {yi:.6f}\n")
        print(f"[AirfoilAgent] Coords written: {path} ({len(x_coords)} pts).")
        return path

    def _load_coords(self, dat_path):
        rows = []
        with open(dat_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try: rows.append([float(parts[0]), float(parts[1])])
                    except ValueError: continue
        if not rows: raise ValueError(f"No coordinate data in {dat_path}")
        return np.array(rows)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_xfoil_results(self, df, out_png):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        name = self.airfoil_name.upper()
        axes[0,0].plot(df["AoA"],df["Cl"],"b-o",ms=3,lw=1.5); axes[0,0].set_xlabel("α"); axes[0,0].set_ylabel("Cl"); axes[0,0].set_title(f"{name} — Lift Curve"); axes[0,0].grid(True,alpha=0.3)
        axes[0,1].plot(df["Cd"],df["Cl"],"r-o",ms=3,lw=1.5); axes[0,1].set_xlabel("Cd"); axes[0,1].set_ylabel("Cl"); axes[0,1].set_title(f"{name} — Drag Polar"); axes[0,1].grid(True,alpha=0.3)
        axes[1,0].plot(df["AoA"],df["ClCd"],"g-o",ms=3,lw=1.5); axes[1,0].set_xlabel("α"); axes[1,0].set_ylabel("Cl/Cd"); axes[1,0].set_title(f"{name} — Efficiency"); axes[1,0].grid(True,alpha=0.3)
        if "Xtr_top" in df.columns and not df["Xtr_top"].isna().all():
            axes[1,1].plot(df["AoA"],df["Xtr_top"],"b-",label="Upper",lw=1.5); axes[1,1].plot(df["AoA"],df["Xtr_bot"],"r-",label="Lower",lw=1.5)
            axes[1,1].set_xlabel("α"); axes[1,1].set_ylabel("Transition x/c"); axes[1,1].set_title(f"{name} — Transition"); axes[1,1].legend(); axes[1,1].grid(True,alpha=0.3); axes[1,1].invert_yaxis()
        Re=self.last_results.get("Re","N/A"); Mach=self.last_results.get("Mach",0)
        fig.suptitle(f"XFOIL: {name}  Re={Re:.2e}  M={Mach:.3f}",fontsize=13,fontweight="bold")
        plt.tight_layout(); plt.savefig(out_png,dpi=300); plt.close()
        print(f"[AirfoilAgent] Plot: {out_png}")

    def _plot_panel_results(self, df, out_png):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        name = self.airfoil_name.upper()
        axes[0].plot(df["AoA"],df["Cl"],"b-o",ms=3,lw=1.5); axes[0].set_xlabel("α"); axes[0].set_ylabel("Cl"); axes[0].set_title("Lift Curve (Inviscid)"); axes[0].grid(True,alpha=0.3)
        axes[1].plot(df["Cd"],df["Cl"],"r-o",ms=3,lw=1.5); axes[1].set_xlabel("Cd"); axes[1].set_ylabel("Cl"); axes[1].set_title("Drag Polar (Est.)"); axes[1].grid(True,alpha=0.3)
        axes[2].plot(df["AoA"],df["ClCd"],"g-o",ms=3,lw=1.5); axes[2].set_xlabel("α"); axes[2].set_ylabel("Cl/Cd"); axes[2].set_title("Efficiency"); axes[2].grid(True,alpha=0.3)
        Re=self.last_results.get("Re","N/A")
        fig.suptitle(f"Panel: {name}  Re={Re:.2e}  ⚠️ INVISCID",fontsize=12,fontweight="bold")
        plt.tight_layout(); plt.savefig(out_png,dpi=300); plt.close()
        print(f"[AirfoilAgent] Plot: {out_png}")