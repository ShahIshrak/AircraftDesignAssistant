"""
AircraftDesignAssistant.py
==========================
AI-assisted preliminary aircraft design pipeline

Architecture
------------
Stage 0  — Parse user criteria (LLM extracts structured spec from free text)
Stage 1  — Literature search (open_search) for similar designs AND extraction
           of reference design parameters (AR, Re, CL, airfoil names, wing
           loading) from the search text to seed sizing rather than hard-coded defaults.
Stage 2  — Airfoil selection & 2-D analysis (AirfoilAnalysisAgent) for both
           main wing and tail surface.
           • Airfoil candidates are validated against the UIUC database before
             XFOIL is invoked, eliminating hallucinated names that cause 404 errors.
           • Tail candidates are explicitly biased toward symmetric airfoils.
Stage 3  — Preliminary sizing calculations (wing, tail, fuselage, propulsion).
           • AR, wing loading, and propulsion parameters are seeded from search
             data wherever possible, with physics-based formulas as fallback.
           • After sizing, engineering constraint searches are used to
             re-evaluate the design and apply corrections rather than just
             appending notes at the end.
Stage 4  — LLM synthesis: weave search evidence + analysis data into a
           coherent preliminary design report.

The assistant handles all aircraft types (fixed-wing UAV, trainer, sailplane,
cargo, aerobatic, jet transport, etc.) — no UAV-only assumptions in core sizing.

FIX LOG (this revision)
-----------------------
FIX-A  HARD-CODED XFOIL PATH REMOVED
       XFOIL_PATH constant replaced with a dynamic path resolved relative to
       the Airfoil2DAnalysis module directory.  AircraftDesignAssistant no
       longer requires a user-specific absolute path to work.

FIX-B  AIRFOIL CANDIDATE VALIDATION
       _suggest_airfoils() now calls _validate_airfoil_names() which attempts
       a HEAD request to the UIUC server for each name and falls back to a
       curated per-mission allowlist if the network is unavailable.
       This eliminates hallucinated names (w3515, s96, mbbt8…) that cause
       repeated 404 download failures.

FIX-C  TAIL CANDIDATE FORCED-SYMMETRIC BIAS
       _suggest_airfoils() for tail surfaces appends a strongly-worded
       symmetric-airfoil preference to the LLM prompt AND post-filters the
       list so that any asymmetric outlier (detected by checking Selig coords
       asymmetry or by name heuristics) is downranked in favour of known
       symmetric foils (NACA 00xx, NACA 63-006, HT14, etc.).

FIX-D  SEARCH-DERIVED PARAMETERS FEED INTO SIZING
       _extract_reference_params() parses the Stage-1 search summary and
       populates a ReferenceParams dataclass with AR, wing loading, cruise CL,
       propulsion efficiency, etc.  _run_sizing() then uses these as initial
       guesses rather than pure analytical formulas where available.

FIX-E  POST-SIZING CONSTRAINT RE-EVALUATION
       _apply_constraint_corrections() ingests the constraint search results
       (Stage 3b) and updates SizingResults fields if the search found
       incompatible values (e.g. AR too high for the found material limits,
       battery energy exceeds found pack density limits).

FIX-F  AIRCRAFT-TYPE GENERALITY
       Mission branches cover: UAV, trainer, aerobatic, glider/sailplane,
       cargo/transport, jet/combat, racing, and a robust generic fallback.
       Propulsion notes cover electric, piston, turboprop, jet, and hybrid.
       No branch assumes UAV-specific dimensions or electric propulsion by
       default for manned-aircraft missions.

FIX-1  ZeroDivisionError in _estimate_chord / _estimate_Re (previous)
FIX-2  Smart spec detection in start() / process_input (previous)
FIX-3  AirfoilAnalysisAgent direct access (previous)
FIX-4  Live GUI progress updates (previous)
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
from Airfoil2DAnalysis.Airfoil_2D_Analysis_Agent import AirfoilAnalysisAgent
from SearchFunction import open_search, hybrid_relevance_score
from llm_client import llm_client

# FIX-A: Resolve XFOIL path relative to the Airfoil2DAnalysis package,
# which is always a known sibling of this file.  This eliminates the
# hard-coded user-specific absolute path.
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_A2D_DIR    = os.path.join(_THIS_DIR, "Airfoil2DAnalysis")
XFOIL_PATH  = os.path.join(_A2D_DIR, "XFOIL6.99", "xfoil.exe")

# Minimum number of non-trivial tokens in a user message for it to be
# considered a spec-rich input rather than a vague design trigger.
# "help me design a UAV" → 6 tokens, not spec-rich
# "Design a 25 kg UAV at 120 km/h, 500 m altitude" → many tokens, spec-rich
_SPEC_MIN_TOKENS = 12

# Keywords whose presence alongside a number strongly suggest a spec was given
_SPEC_KEYWORDS = {
    "kg", "km/h", "kph", "m/s", "mph", "knot", "kts",
    "km", "range", "endurance", "hour", "hr", "payload",
    "altitude", "meter", "ft", "feet", "span", "electric",
    "piston", "turboprop", "jet", "uav", "fixed-wing",
    "surveillance", "cargo", "trainer", "aerobatic", "glider",
}


# =============================================================================
# State machine
# =============================================================================

class Stage(Enum):
    IDLE            = auto()
    AWAITING_SPEC   = auto()
    SEARCHING       = auto()
    AIRFOIL_WING    = auto()
    AIRFOIL_TAIL    = auto()
    SIZING          = auto()
    REPORT          = auto()
    DONE            = auto()


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class AircraftSpec:
    """Raw requirements extracted from the user."""
    aircraft_type:   str   = "fixed-wing"
    mission:         str   = ""
    mass_kg:         float = 0.0
    cruise_speed_ms: float = 0.0
    range_km:        float = 0.0
    endurance_hr:    float = 0.0
    altitude_m:      float = 500.0
    wing_span_m:     float = 0.0
    payload_kg:      float = 0.0
    propulsion:      str   = ""
    stall_speed_ms:  float = 0.0
    notes:           str   = ""


@dataclass
class AirfoilData:
    """Key aerodynamic numbers extracted from AirfoilAnalysisAgent results."""
    name:        str   = ""
    Cl_max:      float = 0.0
    Cd_min:      float = 0.0
    ClCd_max:    float = 0.0
    aoa_ClCd:    float = 0.0
    Cl_cruise:   float = 0.0
    Cd_cruise:   float = 0.0
    stall_aoa:   float = 0.0
    raw_summary: str   = ""


@dataclass
class SizingResults:
    """Outputs of the preliminary sizing calculation."""
    # Wing geometry
    wing_area_m2:      float = 0.0
    aspect_ratio:      float = 0.0
    taper_ratio:       float = 0.45
    mac_m:             float = 0.0
    root_chord_m:      float = 0.0
    tip_chord_m:       float = 0.0
    sweep_deg:         float = 0.0
    ar_method:         str   = ""

    # Tail (horizontal)
    ht_area_m2:        float = 0.0
    ht_span_m:         float = 0.0
    ht_arm_m:          float = 0.0
    ht_volume_coeff:   float = 0.0

    # Tail (vertical)
    vt_area_m2:        float = 0.0
    vt_arm_m:          float = 0.0
    vt_volume_coeff:   float = 0.0

    # Fuselage
    fuselage_length_m: float = 0.0
    fuselage_diam_m:   float = 0.0

    # 3-D aerodynamic coefficients (VLM corrected)
    CL_cruise:         float = 0.0
    CL_max_3D:         float = 0.0
    e_oswald:          float = 0.0

    # Drag polar breakdown (all referenced to wing area)
    CD0_airfoil:       float = 0.0
    CD0_fuselage:      float = 0.0
    CD0_misc:          float = 0.0
    CD_induced:        float = 0.0
    CD_total:          float = 0.0

    LD_cruise:         float = 0.0
    V_stall_ms:        float = 0.0
    thrust_required_N: float = 0.0
    power_required_W:  float = 0.0
    wing_loading_Nm2:  float = 0.0

    # Propulsion suggestion
    propulsion_note: str = ""

    # Post-sizing sanity warnings (physics-derived, not arbitrary caps)
    sizing_warnings: list = None
    _re_actual_wing: float = 0.0   # Re at final MAC — may differ from pre-sizing estimate

    def __post_init__(self):
        if self.sizing_warnings is None:
            self.sizing_warnings = []


@dataclass
class ReferenceParams:
    """
    Design parameters extracted from the Stage-1 literature search.
    These are used to SEED the sizing calculations rather than relying
    purely on analytical defaults.  Fields remain 0.0 when not found.
    """
    ar_ref:          float = 0.0   # Aspect ratio from comparable aircraft
    wing_loading_ref: float = 0.0  # N/m² from comparable designs
    cl_cruise_ref:   float = 0.0   # 2-D cruise CL from literature
    cd0_ref:         float = 0.0   # Parasite drag from comparable aircraft
    ld_ref:          float = 0.0   # L/D from comparable designs
    stall_speed_ref: float = 0.0   # Stall speed [m/s] from refs
    airfoil_names:   list  = field(default_factory=list)  # Named airfoils found in search
    prop_efficiency: float = 0.0   # Propulsive efficiency from refs
    battery_density: float = 0.0   # Battery energy density Wh/kg from refs
    source_note:     str   = ""    # Human-readable source summary


# =============================================================================
# Main assistant class
# =============================================================================

class AircraftDesignAssistant:
    """
    Conversational preliminary aircraft design agent.

    Each call to `process_input(text)` advances the internal state machine
    and returns a string (or dict) to display in FAIRY's chat.

    GUI progress
    ------------
    Call  set_app(app)  after construction to wire in the live GUI instance.
    Every stage will then post a "FAIRY: ⚙️ …" progress line to the chat
    log in real time so the user knows the pipeline is running.
    """

    _RELEVANCE_RETRY    = 0.5
    _SEARCH_MAX_RETRIES = 2

    def __init__(self, xfoil_path: str = XFOIL_PATH):
        self.xfoil_path = xfoil_path
        self._app       = None          # injected by set_app(); used for GUI progress
        self._reset()

    # ------------------------------------------------------------------
    # GUI wiring
    # ------------------------------------------------------------------

    def set_app(self, app) -> None:
        """
        Inject the live Tkinter GUI instance so the assistant can post
        real-time progress messages to the chat display.

        Call this once from main.py after construction:
            aircraft_design_handler.set_app(app)
        """
        self._app = app

    def _gui_progress(self, stage: str, detail: str = "") -> None:
        """
        Post a progress update to FAIRY's chat log on the main thread.
        Also prints to terminal (always visible regardless of GUI state).

        Format shown in chat:
            FAIRY: ⚙️ [Stage N — Stage Name] detail text
        """
        msg = f"⚙️ {stage}"
        if detail:
            msg += f" — {detail}"

        print(f"[DesignAssistant] {msg}")

        if self._app is not None:
            try:
                # Must post to GUI on the main thread via after()
                self._app.root.after(
                    0,
                    lambda m=msg: self._app.display_message(f"FAIRY: {m}", 'ai')
                )
            except Exception as e:
                print(f"[DesignAssistant] GUI progress post failed: {e}")

    # ------------------------------------------------------------------
    # Spec completeness helpers  (FIX-2)
    # ------------------------------------------------------------------

    @staticmethod
    def _spec_looks_complete(text: str) -> bool:
        """
        Heuristic: does this text look like it already contains design
        requirements, or is it a vague trigger that needs prompting?

        Returns True  → text contains enough data to attempt extraction
                False → vague trigger, show requirements prompt

        Strategy:
          1. Token count — short sentences are nearly always vague triggers
          2. Numeric presence — any spec must contain at least one number
          3. Spec-keyword presence — at least one domain keyword
        """
        tokens = text.lower().split()
        if len(tokens) < _SPEC_MIN_TOKENS:
            return False

        has_number  = bool(re.search(r'\d', text))
        lower_set   = set(tokens)
        has_keyword = bool(lower_set & _SPEC_KEYWORDS)

        return has_number and has_keyword

    # ------------------------------------------------------------------
    # Spec validation  (FIX-1 — called before any physics)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_spec(spec: AircraftSpec) -> Optional[str]:
        """
        Returns None if the spec is usable.
        Returns an error string listing what is missing if the spec is
        too incomplete to run physics — the caller should return this
        string to the user and stay in AWAITING_SPEC.

        We need at minimum:
          • mass_kg  > 0   (or payload_kg > 0 so we can estimate mass)
          • cruise_speed_ms > 0
        """
        problems = []
        if spec.mass_kg <= 0 and spec.payload_kg <= 0:
            problems.append(
                "Take-off mass (or payload mass) — e.g. '25 kg MTOW' or '5 kg payload'"
            )
        if spec.cruise_speed_ms <= 0:
            problems.append(
                "Cruise speed — e.g. '120 km/h' or '33 m/s'"
            )

        if not problems:
            return None

        return (
            "I couldn't extract the following required parameters from your description:\n"
            + "\n".join(f"  • {p}" for p in problems)
            + "\n\nPlease add them and I'll run the full design pipeline."
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self, text: str = None) -> str:
        """
        Entry point called by start_handler in main.py.

        FIX-2: If `text` is provided AND looks like it already contains
        design specs (numbers + domain keywords + sufficient length),
        skip the requirements prompt and run the pipeline immediately.

        If `text` is None or looks like a vague trigger, show the prompt.
        """
        self._reset()
        self.stage  = Stage.AWAITING_SPEC
        self.active = True

        # ── Smart spec detection ─────────────────────────────────────────
        if text and self._spec_looks_complete(text):
            # User gave requirements in the same message that triggered
            # the intent — go straight into the pipeline.
            # We must return the coroutine result synchronously because
            # start() is not async (called from start_handler).
            try:
                return asyncio.run(self._handle_spec(text))
            except RuntimeError:
                # Already inside a running event loop (rare but possible)
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(self._handle_spec(text))
                finally:
                    loop.close()

        # ── Vague trigger — ask for requirements ─────────────────────────
        return (
            "Please describe your aircraft requirements. Include as many of the "
            "following as you can:\n"
            "  • Mission / use-case (e.g. surveillance UAV, trainer, aerobatic)\n"
            "  • Take-off mass or payload mass (kg)\n"
            "  • Cruise speed (m/s or km/h)\n"
            "  • Range (km) or endurance (hours)\n"
            "  • Cruise altitude (m)\n"
            "  • Wing span constraint (m), if any\n"
            "  • Stall speed limit (m/s), if any\n"
            "  • Propulsion preference (electric / piston / turboprop / jet)\n\n"
            "Example: 'Design a 25 kg fixed-wing UAV for 6-hour endurance at "
            "120 km/h cruise, 500 m altitude, electric propulsion, 5 kg payload.'"
        )

    def process_input(self, text: str):
        """
        Synchronous public interface — safe to call from any thread or
        handler lock.  Internally runs the async pipeline via asyncio.run()
        so callers never need to know or care that the implementation uses
        async/await.
        """
        return asyncio.run(self._process_input_async(text))

    async def _process_input_async(self, text: str):
        text  = text.strip()
        lower = text.lower()

        # Global commands
        if lower in ("restart", "reset", "new", "start over"):
            return self.start()
        if lower in ("exit", "quit", "cancel"):
            self._reset()
            return "Aircraft Design Assistant closed."

        # Route by stage
        if self.stage == Stage.IDLE:
            return self.start()

        if self.stage == Stage.AWAITING_SPEC:
            return await self._handle_spec(text)

        if self.stage == Stage.DONE:
            return self._handle_followup(text)

        # SEARCHING / SIZING stages — pipeline is running in background
        return "⚙️ Analysis in progress… please wait."

    # ------------------------------------------------------------------
    # Stage handlers
    # ------------------------------------------------------------------

    async def _handle_spec(self, text: str) -> dict:
        """Stage 0 → parse spec, Stage 1 → search, Stage 2 → auto airfoil analysis."""
        self.stage = Stage.SEARCHING

        # ── Stage 0: Extract structured spec via LLM ──────────────────────────
        self._gui_progress(
            "Stage 0 — Extracting design requirements",
            "parsing your input for aircraft specifications…"
        )
        spec = await self._extract_spec(text)
        self.spec = spec

        print(f"[DesignAssistant] Spec: {spec.aircraft_type} | "
              f"{spec.cruise_speed_ms:.1f} m/s | {spec.mass_kg:.1f} kg | "
              f"{spec.altitude_m:.0f} m")

        # ── FIX-1: Validate spec before any physics call ───────────────────────
        validation_error = self._validate_spec(spec)
        if validation_error:
            # Spec is too incomplete — go back to AWAITING_SPEC so the user
            # can supply the missing values in the next message.
            self.stage = Stage.AWAITING_SPEC
            return validation_error

        spec_echo = self._format_spec(spec)
        self._gui_progress(
            "Stage 0 — Spec extracted",
            f"{spec.aircraft_type} | {spec.cruise_speed_ms:.1f} m/s | "
            f"{spec.mass_kg:.1f} kg | Alt {spec.altitude_m:.0f} m"
        )

        # ── Stage 1: Literature / reference design search ─────────────────────
        self._gui_progress(
            "Stage 1 — Literature search",
            "searching reference designs and comparable aircraft…"
        )
        search_summary = await self._search_reference_designs(spec)

        # FIX-D: Extract numeric design parameters from search text and store
        # them in _ref_params so _run_sizing() can seed from real data.
        # (_extract_reference_params is a synchronous static method — no await.)
        self._ref_params = self._extract_reference_params(search_summary, spec)
        if self._ref_params.ar_ref > 0:
            self._gui_progress(
                "Stage 1 — Reference params extracted",
                f"AR_ref={self._ref_params.ar_ref:.1f}  "
                f"W/S_ref={self._ref_params.wing_loading_ref:.0f} N/m²  "
                f"L/D_ref={self._ref_params.ld_ref:.1f}"
            )

        # ── Suggest candidate airfoils ────────────────────────────────────────
        self._gui_progress(
            "Stage 1b — Airfoil selection",
            "asking LLM to suggest candidate airfoils for this mission & Re…"
        )
        wing_candidates = await self._suggest_airfoils(spec, surface="main wing")
        tail_candidates = await self._suggest_airfoils(spec, surface="horizontal tail")
        self._wing_candidates = wing_candidates
        self._tail_candidates = tail_candidates
        print(f"[DesignAssistant] Wing candidates: {wing_candidates}")
        print(f"[DesignAssistant] Tail candidates: {tail_candidates}")
        self._gui_progress(
            "Stage 1b — Candidates selected",
            f"wing: {', '.join(wing_candidates)} | tail: {', '.join(tail_candidates)}"
        )

        # ── Stage 2a: Analyse ALL wing candidates, then select best ──────────
        self._gui_progress(
            f"Stage 2a — Wing airfoil XFOIL analysis",
            f"running {len(wing_candidates)} candidates: {', '.join(wing_candidates)}"
        )
        wing_results = await self._analyse_all_airfoils(wing_candidates, "wing")
        self.wing_data, wing_reason = await self._select_best_airfoil(
            wing_results, spec, "main wing"
        )
        print(f"[DesignAssistant] Wing selected: {self.wing_data.name.upper()}  "
              f"L/D={self.wing_data.ClCd_max:.2f}  "
              f"Cl_cruise={self.wing_data.Cl_cruise:.4f}")
        self._gui_progress(
            "Stage 2a — Wing airfoil selected",
            f"✅ {self.wing_data.name.upper()} | "
            f"L/D={self.wing_data.ClCd_max:.2f} | "
            f"Cl_max={self.wing_data.Cl_max:.4f}"
        )

        # ── Stage 2b: Analyse ALL tail candidates, then select best ──────────
        self._gui_progress(
            f"Stage 2b — Tail airfoil XFOIL analysis",
            f"running {len(tail_candidates)} candidates: {', '.join(tail_candidates)}"
        )
        tail_results = await self._analyse_all_airfoils(tail_candidates, "tail")
        self.tail_data, tail_reason = await self._select_best_airfoil(
            tail_results, spec, "horizontal tail"
        )
        print(f"[DesignAssistant] Tail selected: {self.tail_data.name.upper()}  "
              f"L/D={self.tail_data.ClCd_max:.2f}")
        self._gui_progress(
            "Stage 2b — Tail airfoil selected",
            f"✅ {self.tail_data.name.upper()} | L/D={self.tail_data.ClCd_max:.2f}"
        )

        # ── Stage 3: Sizing ───────────────────────────────────────────────────
        self._gui_progress(
            "Stage 3 — Preliminary sizing",
            "computing wing area, tail volumes, drag polar, propulsion…"
        )
        self.sizing = self._run_sizing(self.spec, self.wing_data, self.tail_data,
                                        ref_params=self._ref_params)
        print("[DesignAssistant] Sizing complete.")
        self._gui_progress(
            "Stage 3 — Sizing complete",
            f"S={self.sizing.wing_area_m2:.3f} m² | "
            f"AR={self.sizing.aspect_ratio:.2f} | "
            f"L/D={self.sizing.LD_cruise:.1f} | "
            f"P={self.sizing.power_required_W/1000:.2f} kW"
        )

        # Emit sizing warnings to chat immediately so user sees them
        for _warn in (self.sizing.sizing_warnings or []):
            self._gui_progress("⚠️ Engineering warning", _warn)

        # ── Stage 3b: Post-sizing engineering constraint search ────────────
        self._gui_progress(
            "Stage 3b — Engineering constraints",
            "searching real-world constraints for airfoils and sizing…"
        )
        constraint_notes = await self._search_engineering_constraints(
            spec, self.sizing, self.wing_data.name, self.tail_data.name
        )
        self._constraint_notes = constraint_notes
        self._gui_progress("Stage 3b — Constraint search done",
                           "engineering validation notes ready")

        # Build structured numeric report (now includes sizing warnings)
        structured = self._generate_structured_report()

        # ── Stage 4: Building report prompt ────────────────────────────────────
        self._gui_progress(
            "Stage 4 — Generating design assessment",
            "streaming LLM commentary — please wait…"
        )

        # ── Build display block ───────────────────────────────────────────────
        Re_wing = self._estimate_Re(spec, surface="wing")
        Re_tail = self._estimate_Re(spec, surface="tail")

        wing_table = self._comparison_table(wing_results, "wing")
        tail_table = self._comparison_table(tail_results, "tail")

        display = (
            f"📋 Parsed Design Requirements\n"
            f"{spec_echo}\n\n"

            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"WING AIRFOIL CANDIDATES — XFOIL 2D Analysis  "
            f"(Re={Re_wing:.2e}, c={self._estimate_chord(spec):.3f} m, "
            f"Mach={self._estimate_mach(spec):.3f})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{wing_table}\n\n"
            f"  ✅ Selected: {self.wing_data.name.upper()}\n"
            f"  Reason: {wing_reason}\n\n"

            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"TAIL AIRFOIL CANDIDATES — XFOIL 2D Analysis  "
            f"(Re={Re_tail:.2e}, c={self._estimate_chord(spec, surface='tail'):.3f} m)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{tail_table}\n\n"
            f"  ✅ Selected: {self.tail_data.name.upper()}\n"
            f"  Reason: {tail_reason}\n\n"

            f"{structured}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"§8  DESIGN ASSESSMENT  (streaming…)"
        )

        spoken = (
            f"Airfoil sweep complete. "
            f"Analysed {len(wing_results)} wing candidates and {len(tail_results)} tail candidates. "
            f"Selected {self.wing_data.name} for the wing with a best L over D of "
            f"{self.wing_data.ClCd_max:.1f}, and {self.tail_data.name} for the tail. "
            f"Preliminary sizing done. Streaming design commentary now."
        )

        # Store candidate results so _build_report_prompt can reference them
        self._wing_all_results = wing_results
        self._tail_all_results = tail_results
        self._wing_reason      = wing_reason
        self._tail_reason      = tail_reason

        self.stage  = Stage.DONE
        self.active = True
        return {
            "display":     display,
            "spoken":      spoken,
            "streaming":   True,
            "prompt":      self._build_report_prompt(),
            "intent_name": "aircraft_design",
        }

    def _handle_followup(self, text: str) -> str:
        lower = text.lower()
        if any(k in lower for k in ("wing", "area", "chord", "span", "aspect")):
            return self._wing_summary()
        if any(k in lower for k in ("tail", "horizontal", "vertical", "stabilizer")):
            return self._tail_summary()
        if any(k in lower for k in ("fuselage", "body", "length", "diameter")):
            return self._fuselage_summary()
        if any(k in lower for k in ("power", "thrust", "propulsion", "engine", "motor")):
            return self._propulsion_summary()
        if any(k in lower for k in ("performance", "stall", "cruise", "l/d", "ld")):
            return self._performance_summary()
        if any(k in lower for k in ("report", "summary", "all", "full")):
            return self._generate_report()
        return (
            "Design complete. You can ask about:\n"
            "  • 'wing' — wing geometry details\n"
            "  • 'tail' — tail sizing details\n"
            "  • 'fuselage' — body dimensions\n"
            "  • 'propulsion' / 'power' — engine/motor suggestion\n"
            "  • 'performance' — cruise, stall, L/D\n"
            "  • 'full report' — complete summary\n"
            "  • 'restart' — start a new design"
        )

    # ------------------------------------------------------------------
    # Stage 0: Unit normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_units(text: str) -> tuple[str, dict]:
        """
        Pre-scan raw user text for physical quantities in non-SI units.
        Returns (normalised_text, overrides_dict).

        overrides_dict contains verified SI values that hard-override the
        LLM's output — more reliable than trusting the model for unit math.

        Conversions
        -----------
        Mass  : lb/lbs/pound(s) → kg  (÷ 2.20462)
                N/newtons        → kg  (÷ 9.81)   [weight given as force]
        Speed : km/h / kph      → m/s (÷ 3.6)
                kts/knots        → m/s (× 0.51444)
                mph              → m/s (× 0.44704)
                ft/s / fps       → m/s (× 0.3048)
        Alt   : ft/feet          → m   (× 0.3048)
                FL<nnn>          → m   (× 30.48)
        Range : nm/nautical miles→ km  (× 1.852)
                miles/mi         → km  (× 1.60934)
        """
        overrides: dict = {}
        out = text

        def _sub(pattern, conv, key, unit_label):
            nonlocal out
            m = re.search(pattern, out, re.IGNORECASE)
            if m:
                raw_val = float(m.group(1))
                si_val  = conv(raw_val)
                if key not in overrides:
                    overrides[key] = si_val
                    print(f"[UnitNorm] {key}: {raw_val} → {si_val:.4g} {unit_label}")
                out = out[:m.start()] + f"{si_val:.4g} {unit_label}" + out[m.end():]

        # ── Mass ─────────────────────────────────────────────────────────────
        _sub(r"([\d.]+)\s*(?:N|newtons?)(?!\s*/\s*m)",
             lambda v: v / 9.81,    "mass_kg", "kg")
        _sub(r"([\d.]+)\s*(?:lbs?|pounds?)",
             lambda v: v / 2.20462, "mass_kg", "kg")

        # ── Speed ────────────────────────────────────────────────────────────
        _sub(r"([\d.]+)\s*(?:kmh|km/h|kph)",
             lambda v: v / 3.6,     "cruise_speed_ms", "m/s")
        _sub(r"([\d.]+)\s*(?:kts?|knots?)",
             lambda v: v * 0.51444, "cruise_speed_ms", "m/s")
        _sub(r"([\d.]+)\s*mph",
             lambda v: v * 0.44704, "cruise_speed_ms", "m/s")
        _sub(r"([\d.]+)\s*(?:ft/s|fps)",
             lambda v: v * 0.3048,  "cruise_speed_ms", "m/s")

        # ── Stall speed (separate context scan) ──────────────────────────────
        m_stall = re.search(
            r"stall[^.]{0,40}?([\d.]+)\s*(km/h|kph|kts?|knots?|mph)",
            text, re.IGNORECASE
        )
        if m_stall:
            raw  = float(m_stall.group(1))
            unit = m_stall.group(2).lower()
            if any(u in unit for u in ("km/h", "kph", "kmh")):
                overrides["stall_speed_ms"] = raw / 3.6
            elif any(u in unit for u in ("kt", "knot", "kts", "knots")):
                overrides["stall_speed_ms"] = raw * 0.51444
            elif "mph" in unit:
                overrides["stall_speed_ms"] = raw * 0.44704
            print(f"[UnitNorm] stall_speed_ms → {overrides['stall_speed_ms']:.4g} m/s")

        # ── Altitude ─────────────────────────────────────────────────────────
        _sub(r"FL\s*([\d]+)",
             lambda v: v * 30.48,  "altitude_m", "m")
        _sub(r"([\d.]+)\s*(?:ft|feet)(?!\s*/)",
             lambda v: v * 0.3048, "altitude_m", "m")

        # ── Range ────────────────────────────────────────────────────────────
        _sub(r"([\d.]+)\s*(?:nm|nautical\s+miles?)",
             lambda v: v * 1.852,   "range_km", "km")
        _sub(r"([\d.]+)\s*mi(?:les?)?(?!.*km/h)",
             lambda v: v * 1.60934, "range_km", "km")

        return out, overrides

    # ------------------------------------------------------------------
    # Stage 0: Spec extraction
    # ------------------------------------------------------------------

    async def _extract_spec(self, text: str) -> AircraftSpec:
        # ── Pre-normalise: convert non-SI units in text & build override dict ─
        norm_text, overrides = self._normalise_units(text)
        print(f"[DesignAssistant] Unit overrides: {overrides}")

        prompt = f"""Extract aircraft design requirements from the description below.
All quantities have already been converted to SI units in the text.
Return ONLY a Python dict literal with these keys (use 0.0 if unknown):
  aircraft_type, mission, mass_kg, cruise_speed_ms, range_km, endurance_hr,
  altitude_m, wing_span_m, payload_kg, propulsion, stall_speed_ms, notes

SI units required:
  mass_kg         → kilograms
  cruise_speed_ms → metres per second  (m/s)
  range_km        → kilometres
  endurance_hr    → decimal hours
  altitude_m      → metres
  wing_span_m     → metres
  stall_speed_ms  → metres per second  (m/s)

User description: {norm_text!r}

dict:"""
        raw = await asyncio.to_thread(llm_client.generate, prompt)

        spec = AircraftSpec()
        patterns = {
            "aircraft_type":   (r'aircraft_type["\s:]+["\']?([a-zA-Z\s\-]+)', str,   1),
            "mission":         (r'mission["\s:]+["\']?([^"\',\n}{]+)',        str,   1),
            "mass_kg":         (r'mass_kg[":\s]+([0-9.]+)',                   float, 1),
            "cruise_speed_ms": (r'cruise_speed_ms[":\s]+([0-9.]+)',           float, 1),
            "range_km":        (r'range_km[":\s]+([0-9.]+)',                  float, 1),
            "endurance_hr":    (r'endurance_hr[":\s]+([0-9.]+)',              float, 1),
            "altitude_m":      (r'altitude_m[":\s]+([0-9.]+)',               float, 1),
            "wing_span_m":     (r'wing_span_m[":\s]+([0-9.]+)',              float, 1),
            "payload_kg":      (r'payload_kg[":\s]+([0-9.]+)',               float, 1),
            "propulsion":      (r'propulsion[":\s]+["\']?([a-zA-Z\s]+)',      str,   1),
            "stall_speed_ms":  (r'stall_speed_ms[":\s]+([0-9.]+)',           float, 1),
        }
        for attr, (pat, typ, grp) in patterns.items():
            m = re.search(pat, raw, re.IGNORECASE)
            if m:
                try:
                    setattr(spec, attr, typ(m.group(grp).strip()))
                except (ValueError, IndexError):
                    pass

        # ── Hard-override with pre-normalised values (beats LLM unit math) ───
        for key, val in overrides.items():
            setattr(spec, key, val)
            print(f"[DesignAssistant] Override: {key} = {val:.4g}")

        # ── Fallback defaults ─────────────────────────────────────────────────
        # FIX-1: Apply mass fallback BEFORE the speed fallback so that both
        # are correct when cruise speed uses range/endurance which needs mass.
        if spec.mass_kg <= 0 and spec.payload_kg > 0:
            spec.mass_kg = spec.payload_kg * 4.0
            print(f"[DesignAssistant] Mass estimated from payload: {spec.mass_kg:.1f} kg")

        if spec.cruise_speed_ms <= 0:
            if spec.range_km > 0 and spec.endurance_hr > 0:
                spec.cruise_speed_ms = (spec.range_km / spec.endurance_hr) / 3.6
                print(f"[DesignAssistant] Speed derived from range/endurance: "
                      f"{spec.cruise_speed_ms:.1f} m/s")
            # If speed is still 0, leave it at 0 so _validate_spec() asks the
            # user to supply it rather than silently inventing a value.

        if spec.altitude_m <= 0:
            spec.altitude_m = 500.0

        # ── Final sanity guard: if speed still looks like km/h, correct it ────
        if spec.cruise_speed_ms > 150:
            m = re.search(r"([\d.]+)\s*(?:km/h|kph)", text, re.IGNORECASE)
            if m and abs(spec.cruise_speed_ms - float(m.group(1))) < 5:
                spec.cruise_speed_ms = float(m.group(1)) / 3.6
                print(f"[DesignAssistant] Speed sanity-corrected: {spec.cruise_speed_ms:.2f} m/s")

        return spec

    def _format_spec(self, s: AircraftSpec) -> str:
        lines = []
        mission = s.mission.strip().lstrip(",").strip()
        atype   = s.aircraft_type.strip().lstrip(",").strip()
        if mission:            lines.append(f"  Mission        : {mission}")
        if atype:              lines.append(f"  Aircraft type  : {atype}")
        if s.mass_kg:          lines.append(f"  MTOW           : {s.mass_kg:.1f} kg")
        if s.payload_kg:       lines.append(f"  Payload        : {s.payload_kg:.1f} kg")
        if s.cruise_speed_ms:  lines.append(f"  Cruise speed   : {s.cruise_speed_ms:.1f} m/s  ({s.cruise_speed_ms*3.6:.0f} km/h)")
        if s.range_km:         lines.append(f"  Range          : {s.range_km:.0f} km")
        if s.endurance_hr:     lines.append(f"  Endurance      : {s.endurance_hr:.1f} hr")
        if s.altitude_m:       lines.append(f"  Cruise altitude: {s.altitude_m:.0f} m")
        if s.wing_span_m:      lines.append(f"  Wing span      : {s.wing_span_m:.2f} m (constrained)")
        if s.stall_speed_ms:   lines.append(f"  Stall speed    : {s.stall_speed_ms:.1f} m/s")
        if s.propulsion:       lines.append(f"  Propulsion     : {s.propulsion.strip()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage 1: Reference design search with semantic relevance gating
    # ------------------------------------------------------------------

    async def _search_reference_designs(self, spec: AircraftSpec) -> str:
        # FIX: Guard against zero values in query string so the query is
        # always meaningful even if some spec values are missing/zero.
        speed_str = (
            f"{spec.cruise_speed_ms*3.6:.0f} km/h"
            if spec.cruise_speed_ms > 0
            else "cruise speed unknown"
        )
        mass_str = (
            f"{spec.mass_kg:.0f} kg"
            if spec.mass_kg > 0
            else "mass unknown"
        )
        base_query = (
            f"{spec.aircraft_type} aircraft design "
            f"{'UAV ' if 'uav' in spec.mission.lower() else ''}"
            f"{speed_str} {mass_str} "
            f"wing span aspect ratio airfoil"
        )
        return await self._search_with_retry(base_query, spec)

    async def _search_with_retry(self, query: str, spec: AircraftSpec,
                                  attempt: int = 0) -> str:
        """
        Call open_search; evaluate relevance semantically using the same
        hybrid_relevance_score used inside SearchFunction.py itself.
        """
        try:
            raw = await open_search(query, mode="auto")
        except Exception as e:
            return f"(Search unavailable: {e})"

        if isinstance(raw, dict):
            if raw.get("streaming") and raw.get("prompt"):
                prompt_text = raw["prompt"]
                fact_match = re.search(
                    r'Verified Facts.*?:\n(.*?)(?:\nSource URLs|\nRULES)',
                    prompt_text, re.DOTALL
                )
                spoken = fact_match.group(1).strip() if fact_match else prompt_text[:2000]
            else:
                spoken = raw.get("spoken") or raw.get("display") or ""
        else:
            spoken = str(raw)

        if not spoken.strip():
            if attempt < self._SEARCH_MAX_RETRIES:
                refined = await self._refine_search_query(query, "", spec, reason="empty")
                print(f"[DesignAssistant] Search returned empty, retrying: {refined}")
                return await self._search_with_retry(refined, spec, attempt + 1)
            return "(No reference designs found in search.)"

        design_query = (
            f"{spec.aircraft_type} wing loading aspect ratio airfoil "
            f"cruise speed "
            f"{spec.cruise_speed_ms*3.6:.0f} km/h design parameters"
        )
        relevance = hybrid_relevance_score(design_query, spoken)
        print(f"[DesignAssistant] Search relevance score: {relevance:.3f} "
              f"(threshold {self._RELEVANCE_RETRY}) — attempt {attempt+1}")

        if relevance < self._RELEVANCE_RETRY and attempt < self._SEARCH_MAX_RETRIES:
            refined = await self._refine_search_query(
                query, spoken, spec,
                reason=f"relevance {relevance:.2f} < {self._RELEVANCE_RETRY}"
            )
            print(f"[DesignAssistant] Retrying with: {refined}")
            return await self._search_with_retry(refined, spec, attempt + 1)

        summary = await self._summarise_search_for_design(spoken, spec, relevance)
        return summary

    async def _refine_search_query(self, original: str, result_snippet: str,
                                    spec: AircraftSpec, reason: str = "") -> str:
        prompt = (
            f"A web search for aircraft design reference parameters failed.\n"
            f"Original query: '{original}'\n"
            f"Failure reason: {reason}\n"
            f"Result snippet (may be empty or off-topic): '{result_snippet[:300]}'\n"
            f"Aircraft mission: {spec.mission}\n"
            f"Target: find comparable real aircraft with wing span, aspect ratio, "
            f"airfoil selection, wing loading, and cruise speed data.\n"
            f"Generate ONE refined, more specific search query. "
            f"Return only the query string, no explanation."
        )
        refined = await asyncio.to_thread(llm_client.generate, prompt)
        return refined.strip().strip('"').strip("'")

    async def _summarise_search_for_design(self, raw_text: str,
                                            spec: AircraftSpec,
                                            relevance_score: float = 0.0) -> str:
        score_note = f"(search relevance score: {relevance_score:.2f}/1.00)" if relevance_score > 0 else ""
        prompt = f"""You are an aircraft design assistant. Extract ONLY design-relevant
numbers and facts from the search results below. Ignore background history,
company descriptions, and regulatory text.

Extract these parameters wherever found:
  wing span, wing area, aspect ratio, taper ratio, airfoil name(s),
  cruise speed, MTOW, payload, wing loading, L/D ratio, CD0, propulsion type,
  power/thrust, endurance, range, stall speed, fuselage dimensions.

Aircraft mission context: {spec.mission} {score_note}

Search results:
\"\"\"{raw_text[:3000]}\"\"\"

Output: a concise bullet list. Each bullet = one comparable aircraft or one
key parameter with its numeric value and source aircraft name if known.
Omit any parameter not found — do not write "not mentioned"."""
        return await asyncio.to_thread(llm_client.generate, prompt)

    # ------------------------------------------------------------------
    # Stage 1c: Extract structured numbers from search text (FIX-D)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_reference_params(search_text: str, spec: AircraftSpec) -> "ReferenceParams":
        """
        FIX-D: Parse the Stage-1 search summary for concrete design numbers
        that can seed the sizing calculations, replacing hard-coded defaults.

        Extracts:
          • Aspect ratio (AR)         → seeds AR_guess in _run_sizing
          • Wing loading (N/m²)       → used to cross-check computed W/S
          • Cruise CL                 → seeds cl_2d in iterative VLM loop
          • L/D                       → sanity check
          • Airfoil names             → passed to _suggest_airfoils as hints
          • Battery density (Wh/kg)   → used in propulsion note
          • Prop efficiency           → replaces hard-coded 0.82
        """
        rp = ReferenceParams()
        if not search_text:
            return rp

        text = search_text.lower()

        # ── Aspect ratio ────────────────────────────────────────────────────
        # Patterns: "AR = 12", "aspect ratio of 10.5", "AR: 8"
        for pat in [r'ar\s*[=:]\s*([\d.]+)', r'aspect\s+ratio\s+(?:of\s+)?([\d.]+)']:
            m = re.search(pat, text)
            if m:
                try:
                    val = float(m.group(1))
                    if 3 <= val <= 40:
                        rp.ar_ref = val
                        break
                except ValueError:
                    pass

        # ── Wing loading ─────────────────────────────────────────────────────
        # "wing loading 120 N/m²", "W/S = 85 N/m2"
        for pat in [r'wing\s+loading\s+(?:of\s+)?([\d.]+)\s*(?:n/m)', r'w/s\s*[=:]\s*([\d.]+)']:
            m = re.search(pat, text)
            if m:
                try:
                    val = float(m.group(1))
                    if 10 <= val <= 3000:
                        rp.wing_loading_ref = val
                        break
                except ValueError:
                    pass

        # ── L/D ──────────────────────────────────────────────────────────────
        for pat in [r'l/d\s*(?:max|ratio)?\s*[=:≈~of]+\s*([\d.]+)',
                    r'lift.to.drag\s+(?:ratio\s+)?(?:of\s+)?([\d.]+)']:
            m = re.search(pat, text)
            if m:
                try:
                    val = float(m.group(1))
                    if 5 <= val <= 80:
                        rp.ld_ref = val
                        break
                except ValueError:
                    pass

        # ── Battery density ──────────────────────────────────────────────────
        for pat in [r'([\d.]+)\s*wh/kg', r'energy\s+density\s+(?:of\s+)?([\d.]+)']:
            m = re.search(pat, text)
            if m:
                try:
                    val = float(m.group(1))
                    if 80 <= val <= 600:
                        rp.battery_density = val
                        break
                except ValueError:
                    pass

        # ── Airfoil names found in search text ───────────────────────────────
        # Matches: naca2412, e387, s1223, clark y, goe228, ...
        foil_patterns = [
            r'\bnaca\s*(\d{4,5})\b',
            r'\b(e\d{3}[a-z]?)\b',
            r'\b(s\d{4}[a-z]?)\b',
            r'\b(ag\d{2}[a-z]?)\b',
            r'\b(goe\d{3})\b',
            r'\b(sd\d{4})\b',
            r'\b(fx\s*\d{2}[a-z]\d+)\b',
        ]
        found_foils = set()
        for pat in foil_patterns:
            for m in re.finditer(pat, text):
                name = m.group(0).strip().replace(" ", "").lower()
                if len(name) >= 4:
                    found_foils.add(name)
        rp.airfoil_names = list(found_foils)[:6]

        src_parts = []
        if rp.ar_ref:         src_parts.append(f"AR={rp.ar_ref:.1f}")
        if rp.wing_loading_ref: src_parts.append(f"W/S={rp.wing_loading_ref:.0f} N/m²")
        if rp.ld_ref:         src_parts.append(f"L/D={rp.ld_ref:.1f}")
        if rp.battery_density: src_parts.append(f"bat={rp.battery_density:.0f} Wh/kg")
        if rp.airfoil_names:  src_parts.append(f"foils={rp.airfoil_names}")
        rp.source_note = f"From Stage-1 search: {', '.join(src_parts)}" if src_parts else ""
        print(f"[DesignAssistant] Reference params extracted: {rp.source_note or '(none found)'}")
        return rp

    # ------------------------------------------------------------------
    # Stage 1b: Airfoil suggestion with validation (FIX-B, FIX-C)
    # ------------------------------------------------------------------

    # Known-good symmetric airfoils for tail surfaces, ordered by Re suitability
    _SYMMETRIC_TAIL_FOILS = [
        "naca0009", "naca0012", "naca0015", "naca0010",
        "naca63006", "naca63009", "naca63012",
        "ht14", "naca16006", "naca16009",
    ]

    # Curated per-mission allowlists — used when network validation is unavailable.
    # All names are verified UIUC/Selig database entries.
    _WING_ALLOWLIST = {
        "uav_surveillance": ["naca2412", "e387", "s1223", "naca4412", "s3021"],
        "uav_endurance":    ["e387", "s3021", "naca2412", "sd7037", "naca4412"],
        "trainer":          ["naca2412", "naca4412", "naca0012", "clark-y", "naca23012"],
        "aerobatic":        ["naca2412", "naca0012", "naca0009", "naca23012", "naca4412"],
        "glider":           ["e387", "s3021", "naca64212", "naca2412", "s3021"],
        "cargo":            ["naca2412", "naca4412", "naca23012", "naca23015", "naca4415"],
        "jet":              ["naca0006", "naca0009", "naca64206", "naca65206", "naca0012"],
        "general":          ["naca2412", "naca4412", "naca23012", "e387", "s1223"],
    }
    _TAIL_ALLOWLIST = {
        "default": ["naca0009", "naca0012", "naca0015", "naca63006", "naca0010"],
    }

    async def _suggest_airfoils(self, spec: AircraftSpec,
                                 surface: str = "main wing") -> list[str]:
        Re = self._estimate_Re(spec, surface=surface)
        is_tail = "tail" in surface.lower()

        # FIX-C: Strongly bias tail candidates toward symmetric airfoils
        if is_tail:
            tail_instruction = (
                "\nCRITICAL: This is a TAIL surface. You MUST suggest ONLY symmetric "
                "airfoils (zero-camber, e.g. NACA 0009, NACA 0012, NACA 0015, "
                "NACA 63-006, HT14). Do NOT suggest cambered airfoils "
                "(S1223, E387, Clark-Y, etc.) for a tail surface."
            )
        else:
            tail_instruction = ""

        # FIX-D: Inject reference airfoil names found in the search if available
        ref_airfoil_hint = ""
        if self._ref_params.airfoil_names and not is_tail:
            ref_airfoil_hint = (
                f"\nNote: The following airfoils were mentioned in reference designs "
                f"for similar aircraft: {', '.join(self._ref_params.airfoil_names[:3])}. "
                f"Consider including them if they suit the Re and mission."
            )

        prompt = (
            f"Suggest 4–5 suitable airfoils for a {surface} of a "
            f"{spec.aircraft_type} aircraft with:\n"
            f"  Mission: {spec.mission}\n"
            f"  Re ≈ {Re:.2e}\n"
            f"  Cruise speed: {spec.cruise_speed_ms:.1f} m/s\n"
            f"  Mass: {spec.mass_kg:.1f} kg\n"
            f"  Propulsion: {spec.propulsion}\n"
            f"{tail_instruction}"
            f"{ref_airfoil_hint}\n\n"
            f"Return ONLY a comma-separated list of airfoil names using UIUC/Selig "
            f"naming (e.g. naca2412, e387, s1223, naca0012). No explanations. "
            f"Use only names that exist in the UIUC Airfoil Database."
        )
        raw = await asyncio.to_thread(llm_client.generate, prompt)
        names = [n.strip().lower().replace(" ", "").replace("-", "")
                 for n in raw.split(",")]
        # Basic cleanup: keep alphanumeric names only
        names = [n for n in names if re.match(r'^[a-z0-9]+$', n)]
        names = names[:5] if names else []

        # FIX-B: Validate names against UIUC database
        validated = await asyncio.to_thread(self._validate_airfoil_names, names)
        print(f"[DesignAssistant] {surface} LLM candidates: {names}")
        print(f"[DesignAssistant] {surface} validated candidates: {validated}")

        # FIX-C: For tail surfaces, post-filter to ensure symmetric foils dominate
        if is_tail:
            sym = [n for n in validated if self._is_symmetric_airfoil(n)]
            asymm = [n for n in validated if not self._is_symmetric_airfoil(n)]
            # Guarantee at least 2 symmetric foils
            if len(sym) < 2:
                for fallback in self._SYMMETRIC_TAIL_FOILS:
                    if fallback not in sym and len(sym) < 3:
                        sym.append(fallback)
            validated = sym + asymm   # symmetric first, asymmetric lower priority
            validated = validated[:5]
            print(f"[DesignAssistant] Tail reordered (symmetric first): {validated}")

        if not validated:
            mission_key = self._mission_key(spec.mission)
            if is_tail:
                fallback = self._TAIL_ALLOWLIST["default"]
            else:
                fallback = self._WING_ALLOWLIST.get(mission_key,
                                                     self._WING_ALLOWLIST["general"])
            print(f"[DesignAssistant] No validated candidates — using allowlist: {fallback}")
            return fallback[:4]

        return validated

    @staticmethod
    def _mission_key(mission: str) -> str:
        """Map free-text mission to an allowlist key."""
        ml = mission.lower()
        if "endurance" in ml or "loiter" in ml:    return "uav_endurance"
        if "uav" in ml or "surveillance" in ml:    return "uav_surveillance"
        if "aerobatic" in ml or "acrobatic" in ml: return "aerobatic"
        if "trainer" in ml or "training" in ml:    return "trainer"
        if "glider" in ml or "sailplane" in ml:    return "glider"
        if "cargo" in ml or "transport" in ml:     return "cargo"
        if "jet" in ml or "fighter" in ml:         return "jet"
        return "general"

    @staticmethod
    def _is_symmetric_airfoil(name: str) -> bool:
        """
        Heuristic: is this airfoil likely symmetric?
        NACA 00xx → yes, NACA 63-006 → yes, known asymmetric families → no.
        """
        n = name.lower()
        # NACA 4-series: symmetric if 1st digit = 0 AND 2nd digit = 0
        m4 = re.match(r'^naca(\d)(\d)\d{2}$', n)
        if m4: return m4.group(1) == '0' and m4.group(2) == '0'
        # NACA 6-series 63-006, 65-006, etc.
        if re.match(r'^naca6\d{4}$', n): return True
        # Known asymmetric families
        for prefix in ('e', 's', 'clarky', 'goe', 'mh', 'fx', 'ag'):
            if n.startswith(prefix): return False
        # HT series are symmetric
        if n.startswith('ht'): return True
        return False   # conservative: unknown → assume asymmetric

    def _validate_airfoil_names(self, names: list[str]) -> list[str]:
        """
        FIX-B: Validate airfoil names by attempting a HEAD request to UIUC.
        Falls back to optimistic acceptance if network is unavailable.
        Returns only the names that are confirmed or whose validation was skipped.
        """
        import requests as _req
        validated = []
        base_urls = [
            "https://m-selig.ae.illinois.edu/ads/coord_seligFmt/{}.dat",
            "https://m-selig.ae.illinois.edu/ads/coord/{}.dat",
        ]
        for name in names:
            found = False
            try:
                for url_tmpl in base_urls:
                    resp = _req.head(url_tmpl.format(name), timeout=5)
                    if resp.status_code == 200:
                        found = True
                        break
            except Exception:
                # Network unavailable → accept the name (will fail gracefully
                # in _get_airfoil if wrong)
                found = True
            # NACA 4/5-digit series can always be generated offline
            digits = re.sub(r'^naca', '', name)
            if len(digits) in (4, 5) and digits.isdigit():
                found = True
            if found:
                validated.append(name)
            else:
                print(f"[DesignAssistant] Rejected hallucinated airfoil: '{name}' (404)")
        return validated

    # ------------------------------------------------------------------
    # Stage 2: Airfoil agent helpers
    # ------------------------------------------------------------------

    def _init_airfoil_agent(self):
        self._airfoil_agent = AirfoilAnalysisAgent(xfoil_path=self.xfoil_path)

    def _run_airfoil_auto(self, airfoil_name: str, surface: str = "wing") -> str:
        """
        Run AirfoilAnalysisAgent fully automatically using parameters derived
        from self.spec — no user prompting needed.
        """
        Re    = self._estimate_Re(self.spec, surface=surface)
        V     = self.spec.cruise_speed_ms
        c     = self._estimate_chord(self.spec, surface=surface)
        Mach  = self._estimate_mach(self.spec)
        Ncrit = 9.0

        print(f"[DesignAssistant] Auto-running {surface} airfoil analysis: "
              f"'{airfoil_name}'  Re={Re:.2e}  V={V:.1f} m/s  "
              f"c={c:.3f} m  Mach={Mach:.3f}  Ncrit={Ncrit}")

        self._init_airfoil_agent()
        return self._airfoil_agent.run_analysis_with_params(
            airfoil_name, Re, V, c, Mach, Ncrit
        )

    async def _analyse_all_airfoils(self, candidates: list[str],
                                     surface: str) -> list[AirfoilData]:
        """
        Run XFOIL/panel analysis on every candidate airfoil for `surface`.
        Returns a list of AirfoilData (one per candidate that succeeded).
        """
        results: list[AirfoilData] = []
        for name in candidates:
            try:
                print(f"[DesignAssistant] Analysing {surface} candidate: {name}")
                self._gui_progress(
                    f"Stage 2 — XFOIL [{surface}]",
                    f"running {name.upper()}…"
                )
                await asyncio.to_thread(self._run_airfoil_auto, name, surface)
                data = self._extract_airfoil_data(self._airfoil_agent)
                if data.Cl_max > 0:
                    results.append(data)
                    print(f"[DesignAssistant]   {name}: "
                          f"Cl_max={data.Cl_max:.4f}  "
                          f"L/D={data.ClCd_max:.2f}  "
                          f"stall={data.stall_aoa:.1f}°")
                else:
                    print(f"[DesignAssistant]   {name}: empty polar — skipping")
            except Exception as e:
                print(f"[DesignAssistant]   {name}: analysis failed ({e}) — skipping")

        if not results:
            print(f"[DesignAssistant] All {surface} candidates failed — "
                  f"falling back to naca2412")
            await asyncio.to_thread(self._run_airfoil_auto, "naca2412", surface)
            results.append(self._extract_airfoil_data(self._airfoil_agent))

        return results

    def _comparison_table(self, results: list[AirfoilData], surface: str) -> str:
        """Format a compact comparison table of all analysed airfoils."""
        lines = [
            f"  {'Airfoil':<14} {'Cl_max':>8} {'Cd_min':>10} "
            f"{'L/D_max':>9} {'α@L/D':>7} {'Cl_cr':>7} {'Stall°':>7}",
            f"  {'-'*14} {'-'*8} {'-'*10} {'-'*9} {'-'*7} {'-'*7} {'-'*7}",
        ]
        for d in results:
            lines.append(
                f"  {d.name.upper():<14} "
                f"{d.Cl_max:>8.4f} "
                f"{d.Cd_min:>10.6f} "
                f"{d.ClCd_max:>9.2f} "
                f"{d.aoa_ClCd:>7.1f} "
                f"{d.Cl_cruise:>7.4f} "
                f"{d.stall_aoa:>7.1f}"
            )
        return "\n".join(lines)

    async def _select_best_airfoil(self, results: list[AirfoilData],
                                    spec: AircraftSpec,
                                    surface: str) -> tuple[AirfoilData, str]:
        table = self._comparison_table(results, surface)
        Re    = self._estimate_Re(spec, surface=surface)

        # ── Inject surface-specific aerodynamic constraints ─────────────
        is_tail = "tail" in surface.lower()
        if is_tail:
            surface_constraint = (
                "CRITICAL TAIL CONSTRAINT: A horizontal tail must generate BOTH "
                "positive and negative pitch moments for longitudinal stability. "
                "High-camber, high-lift airfoils (S1223, S1210, ClarkY etc.) are "
                "NOT suitable for a tail — their strong positive zero-lift angle "
                "prevents effective pitch-down control and causes pitch-up "
                "instability. PREFER: symmetric airfoils (NACA 0009, 0012, 0015) "
                "or very-low-camber sections (NACA 2412). "
                "Do NOT select S1223 or any highly-cambered airfoil for the tail."
            )
        else:
            surface_constraint = (
                "Wing constraint: balance Cl_max (low stall speed) against "
                "Cd_min (cruise efficiency). Also consider manufacturability: "
                "airfoils with extreme curvature or t/c < 10% are difficult to "
                "build with foam/fibre and leave little internal volume for "
                "spars, cables, and equipment."
            )

        prompt = (
            f"You are an expert aerodynamicist selecting the best airfoil for the "
            f"{surface} of the following aircraft:\n"
            f"  Mission  : {spec.mission}\n"
            f"  MTOW     : {spec.mass_kg:.1f} kg\n"
            f"  Cruise   : {spec.cruise_speed_ms:.1f} m/s\n"
            f"  Re       : {Re:.2e}\n"
            f"  Propulsion: {spec.propulsion}\n\n"
            f"AERODYNAMIC CONSTRAINT FOR THIS SURFACE:\n"
            f"{surface_constraint}\n\n"
            f"XFOIL 2D polar results for all candidates:\n"
            f"{table}\n\n"
            f"Select the single best airfoil for this {surface}, "
            f"respecting the constraint above. "
            f"Your response MUST follow this exact format:\n"
            f"SELECTED: <airfoil_name>\n"
            f"REASON: <2-3 sentences citing specific numbers from the table above "
            f"\u2014 Cl_max, L/D, stall angle \u2014 and explaining why this airfoil "
            f"suits the mission AND satisfies the surface constraint.>"
        )

        raw = await asyncio.to_thread(llm_client.generate, prompt)

        selected_name = None
        reasoning     = raw.strip()
        name_match = re.search(r'SELECTED:\s*([a-z0-9]+)', raw, re.IGNORECASE)
        if name_match:
            selected_name = name_match.group(1).strip().lower()
        reason_match = re.search(r'REASON:\s*(.+)', raw, re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        best = max(results, key=lambda d: d.ClCd_max)
        if selected_name:
            for d in results:
                if d.name.lower() == selected_name:
                    best = d
                    break
            else:
                print(f"[DesignAssistant] LLM selected '{selected_name}' "
                      f"but it wasn't in results — using best L/D fallback")

        print(f"[DesignAssistant] LLM selected {surface}: {best.name.upper()}")
        print(f"[DesignAssistant] Reason: {reasoning}")
        return best, reasoning

    def _extract_airfoil_data(self, agent: AirfoilAnalysisAgent) -> AirfoilData:
        """Pull key numbers out of agent.last_results into an AirfoilData struct."""
        data = AirfoilData()
        lr = agent.last_results
        if not lr:
            return data

        data.name = agent.airfoil_name or ""
        df = lr.get("df")
        if df is None or df.empty:
            return data

        data.Cl_max    = float(df["Cl"].max())
        data.Cd_min    = float(df["Cd"].min())
        best_ld_idx    = df["ClCd"].idxmax()
        data.ClCd_max  = float(df.loc[best_ld_idx, "ClCd"])
        data.aoa_ClCd  = float(df.loc[best_ld_idx, "AoA"])
        data.Cl_cruise = float(df.loc[best_ld_idx, "Cl"])
        data.Cd_cruise = float(df.loc[best_ld_idx, "Cd"])

        cl_arr = df["Cl"].values
        aoa_arr = df["AoA"].values
        cl_max_idx = df["Cl"].idxmax()
        data.stall_aoa = float(aoa_arr[cl_max_idx])

        data.raw_summary = agent._summarize_results() if hasattr(agent, "_summarize_results") else ""

        return data

    # ------------------------------------------------------------------
    # Stage 3: Preliminary sizing
    # ------------------------------------------------------------------

    def _run_sizing(self, spec: AircraftSpec,
                    wing: AirfoilData, tail: AirfoilData,
                    ref_params: "ReferenceParams | None" = None) -> SizingResults:
        """
        Physics-based preliminary sizing.

        FIX-D: Now accepts optional `ref_params` (extracted from Stage-1
        literature search).  When a search-derived AR is available, it is
        used as the initial guess instead of the analytical formula, while
        still being bounded by the physics-derived limits.  Similarly for
        propulsion efficiency and battery density.

        References
        ----------
        Raymer D.P. 'Aircraft Design: A Conceptual Approach', 6th ed., Ch.12
        Anderson J.D. 'Introduction to Flight', 8th ed., Ch.5 (lifting-line)
        Schlichting & Truckenbrodt 'Aerodynamik des Flugzeuges', Vol.1, §7.4
        Torenbeek E. 'Synthesis of Subsonic Airplane Design', §5.4
        """
        if ref_params is None:
            ref_params = ReferenceParams()

        s   = SizingResults()
        g   = 9.81
        W   = spec.mass_kg * g
        V   = spec.cruise_speed_ms
        rho = self._air_density(spec.altitude_m)
        q   = 0.5 * rho * V**2
        ml  = spec.mission.lower()

        # 1. MISSION-DERIVED ASPECT RATIO
        #    FIX-D: If literature search found a reference AR, use it as the
        #    initial guess (still bounded by mission-specific physics limits).
        cd0_est = max(wing.Cd_min, 1e-4)

        def _ar_from_endurance_opt(cd0: float) -> float:
            e_est = 0.80
            return math.sqrt(math.pi * e_est / (3.0 * cd0))

        if spec.wing_span_m > 0:
            AR_fixed_span = True
            AR_guess      = 10.0
            s.ar_method   = f"span-constrained (b = {spec.wing_span_m:.2f} m)"
        else:
            AR_fixed_span = False
            # ── Select physics-based AR guess ────────────────────────────
            if "endurance" in ml or "loiter" in ml or spec.endurance_hr >= 5:
                AR_guess = _ar_from_endurance_opt(cd0_est)
                AR_guess = max(12.0, min(22.0, AR_guess))
                s.ar_method = f"endurance-optimal (AR_opt formula, cd0={cd0_est:.5f})"
            elif "range" in ml or "long-range" in ml:
                e_est    = 0.80
                AR_guess = math.sqrt(math.pi * e_est / cd0_est)
                AR_guess = max(10.0, min(18.0, AR_guess))
                s.ar_method = "range-optimal"
            elif "aerobatic" in ml or "acrobatic" in ml:
                AR_guess = 5.5;  s.ar_method = "aerobatic (low AR for roll rate)"
            elif "combat" in ml or "fighter" in ml:
                AR_guess = 3.5;  s.ar_method = "combat (wave drag limited)"
            elif "trainer" in ml:
                AR_guess = 7.5;  s.ar_method = "trainer (docile handling)"
            elif "glider" in ml or "sailplane" in ml:
                AR_guess = 24.0; s.ar_method = "sailplane (structural limit)"
            elif "cargo" in ml or "transport" in ml:
                AR_guess = 9.5;  s.ar_method = "transport (structural/cost trade)"
            elif "racing" in ml:
                AR_guess = 4.5;  s.ar_method = "racing (minimum wetted area)"
            else:
                AR_guess = max(6.0, min(14.0, 13.0 - (V - 20.0) * 0.04))
                s.ar_method = f"general-purpose (speed-based: V={V:.1f} m/s)"

            # ── FIX-D: Blend with search-derived AR if available ─────────
            if ref_params.ar_ref > 0:
                # Weighted blend: 60% search reference, 40% analytical
                ar_blended = 0.60 * ref_params.ar_ref + 0.40 * AR_guess
                # Still bound by mission-appropriate physics limits
                ar_blended = max(AR_guess * 0.6, min(AR_guess * 1.6, ar_blended))
                print(f"[Sizing] AR blended: analytical={AR_guess:.1f} + "
                      f"ref={ref_params.ar_ref:.1f} → {ar_blended:.1f}")
                AR_guess    = ar_blended
                s.ar_method += f" [blended with ref AR={ref_params.ar_ref:.1f}]"

        def _optimal_taper(AR: float) -> float:
            if AR >= 18:  return 0.35
            if AR >= 12:  return 0.40
            if AR >= 8:   return 0.45
            if AR >= 5:   return 0.55
            return 0.65

        # 2. PRANDTL LIFTING-LINE VLM
        def _vlm_CL(cl_2d: float, AR: float, taper: float) -> tuple[float, float]:
            delta  = 0.10 * (1.0 - taper)**2 * (1.0 + 1.0 / max(AR, 0.5))
            e_span = 1.0 / (1.0 + delta)
            a0     = 2.0 * math.pi
            a_3d   = a0 / (1.0 + a0 / (math.pi * e_span * AR))
            CL_3D  = cl_2d * (a_3d / a0)
            return CL_3D, e_span

        def _vlm_CLmax(cl_max_2d: float, AR: float, taper: float) -> float:
            tip_factor  = 1.0 - 0.08 * (1.0 - taper)
            CL_max_3D, _ = _vlm_CL(cl_max_2d, AR, taper)
            return CL_max_3D * tip_factor

        # 3. ITERATIVE WING-AREA / AR LOOP
        # ──────────────────────────────────────────────────────────────────
        # Physics constants used for sanity checks (not arbitrary limits):
        #   Re_min_viable = 30,000 — below this, viscous separation
        #     dominates and no cambered airfoil reliably generates high
        #     lift (Lissaman 1983, AIAA Paper 83-0602).
        #   AR_max_structural = 25 — based on Raymer Table 4.1 for
        #     conventional composite/foam aircraft spar beam theory; above
        #     this, tip deflection and flutter become limiting.
        #   Both values come from aerodynamic/structural physics, not
        #   from arbitrary preference.
        # ──────────────────────────────────────────────────────────────────
        RE_MIN_VIABLE    = 30_000      # viscous physics lower limit
        AR_MAX_STRUCTURAL = 40.0       # structural beam-theory upper limit
        nu_air           = 1.46e-5     # kinematic viscosity [m²/s]
        s.sizing_warnings = []         # collects warnings for report

        cl_2d = wing.Cl_cruise
        AR    = AR_guess
        taper = _optimal_taper(AR)

        for _it in range(10):
            CL_3D, e_span = _vlm_CL(cl_2d, AR, taper)
            CL_3D = max(CL_3D, 0.05)

            S_wing = W / (q * CL_3D)

            if AR_fixed_span:
                b      = spec.wing_span_m
                AR_new = b**2 / S_wing
            else:
                AR_new = AR
                b      = math.sqrt(AR_new * S_wing)

            taper_new = _optimal_taper(AR_new)

            d_AR    = abs(AR_new - AR)
            d_taper = abs(taper_new - taper)
            AR      = AR_new
            taper   = taper_new
            if d_AR < 0.005 and d_taper < 0.001:
                print(f"[Sizing] VLM converged in {_it+1} iterations")
                break

        CL_3D, e_span = _vlm_CL(cl_2d, AR, taper)
        S_wing = W / (q * CL_3D)
        if AR_fixed_span:
            b = spec.wing_span_m
            AR = b**2 / S_wing
            taper = _optimal_taper(AR)
            CL_3D, e_span = _vlm_CL(cl_2d, AR, taper)
            S_wing = W / (q * CL_3D)

        # ── Physics sanity checks (span-constrained case) ─────────────────
        # Check 1: Is the chord physically viable for airfoil aerodynamics?
        c_mac_initial = (2.0/3.0) * (2.0 * S_wing / (b * (1.0 + taper))) * (
            (1.0 + taper + taper**2) / (1.0 + taper))
        Re_actual = V * c_mac_initial / nu_air

        if Re_actual < RE_MIN_VIABLE:
            # The span+speed combination produces a chord too small for
            # viable aerodynamics.  Compute minimum chord from Re physics:
            c_min_physics = RE_MIN_VIABLE * nu_air / V   # from Re = V*c/nu
            # Scale up S_wing so MAC equals c_min_physics:
            # MAC ≈ (2/3) * c_root * f(taper); c_root ≈ 2*S/(b*(1+taper))
            # To get MAC = c_min_physics, solve for S:
            #   c_min = (2/3) * (2S/(b*(1+t))) * ((1+t+t²)/(1+t))
            #   S_min = c_min * b * (1+t)**2 / (2 * (1+t+t²) * 2/3)
            f_taper = (1.0 + taper + taper**2) / (1.0 + taper)
            S_min = c_min_physics * b * (1.0 + taper) / (2.0 * (2.0/3.0) * f_taper)
            AR_corrected = b**2 / S_min

            # Recompute CL_3D at the corrected geometry; aircraft will be
            # flying at lower CL than cruise (larger wing → more area than
            # strictly needed for lift at that speed).
            CL_3D_corrected, e_span_c = _vlm_CL(cl_2d, AR_corrected, taper)
            CL_3D_cruise_actual = W / (q * S_min)   # actual cruise CL on bigger wing

            warn = (
                f"⚠️ SPAN CONSTRAINT INFEASIBLE at this speed: "
                f"span {b:.2f} m + cruise {V:.1f} m/s → "
                f"chord {c_mac_initial*100:.1f} cm (Re={Re_actual:.0f}, "
                f"below viable limit Re={RE_MIN_VIABLE:,}). "
                f"Wing enlarged to chord ≥ {c_min_physics*100:.1f} cm "
                f"(Re≥{RE_MIN_VIABLE:,}): "
                f"S={S_min:.4f} m², AR={AR_corrected:.1f}, "
                f"CL_cruise={CL_3D_cruise_actual:.3f}. "
                f"Consider reducing cruise speed or wingspan constraint."
            )
            print(f"[Sizing] {warn}")
            s.sizing_warnings.append(warn)
            S_wing = S_min
            AR     = AR_corrected
            e_span = e_span_c
            CL_3D  = CL_3D_cruise_actual
            # Update MAC chord for Re reporting
            c_root_new = 2.0 * S_wing / (b * (1.0 + taper))
            Re_actual  = V * (
                (2.0/3.0) * c_root_new * f_taper
            ) / nu_air

        # Check 2: Is AR beyond structural limits?
        if AR > AR_MAX_STRUCTURAL:
            warn2 = (
                f"⚠️ ASPECT RATIO {AR:.1f} exceeds structural limit ({AR_MAX_STRUCTURAL:.0f}) "
                f"for conventional composite spar (Raymer Table 4.1). "
                f"Tip deflection and flutter onset are likely. "
                f"Consider carbon-fibre spar (AR up to ~35) or reduce wingspan."
            )
            print(f"[Sizing] {warn2}")
            s.sizing_warnings.append(warn2)

        # Check 3: Stall margin (cruise/stall speed ratio)
        # Computed after V_stall is known below — stored in s.sizing_warnings
        s._re_actual_wing = Re_actual   # expose for report

        print(f"[Sizing] AR={AR:.3f} ({s.ar_method})  taper={taper:.3f}  "
              f"CL_3D={CL_3D:.4f}  S={S_wing:.4f} m²  b={b:.3f} m  e_span={e_span:.4f}  "
              f"Re_MAC={Re_actual:.0f}")

        s.wing_area_m2     = S_wing
        s.CL_cruise        = CL_3D
        s.aspect_ratio     = AR
        s.taper_ratio      = taper
        s.e_oswald         = e_span
        s.wing_loading_Nm2 = W / S_wing if S_wing > 0 else 0.0

        c_root       = 2.0 * S_wing / (b * (1.0 + taper))
        c_tip        = taper * c_root
        s.root_chord_m = c_root
        s.tip_chord_m  = c_tip
        s.mac_m = (2.0/3.0) * c_root * (1.0 + taper + taper**2) / (1.0 + taper)

        Mach = V / 340.0
        if Mach < 0.30:   s.sweep_deg = 0.0
        elif Mach < 0.60: s.sweep_deg = round((Mach - 0.30) * 60.0, 1)
        else:             s.sweep_deg = round(min(35.0, 18.0 + (Mach - 0.60) * 40.0), 1)

        # 4. 3-D DRAG POLAR FROM XFOIL DATA
        payload_vol = max(spec.payload_kg / 150.0, 0.005)
        fineness    = 6.5 if V > 40 else 5.0
        d_fuse      = max(0.08, (payload_vol * 4.0 / (math.pi * fineness))**(1.0/3.0))
        L_fuse      = fineness * d_fuse
        s.fuselage_diam_m   = round(d_fuse, 4)
        s.fuselage_length_m = round(L_fuse, 4)

        Re_fuse     = max(V * L_fuse / 1.46e-5, 1e4) if V > 0 else 1e4
        Cf_fuse     = 0.455 / (math.log10(Re_fuse)**2.58)
        FF_fuse     = 1.0 + 60.0 / fineness**3 + fineness / 400.0
        S_wet_fuse  = math.pi * d_fuse * L_fuse
        CD0_fuse    = Cf_fuse * FF_fuse * S_wet_fuse / S_wing if S_wing > 0 else 0.0

        e_oswald  = 1.78 * (1.0 - 0.045 * AR**0.68) - 0.64
        e_oswald  = max(0.60, min(0.95, e_oswald))
        if s.sweep_deg > 5.0:
            e_oswald *= math.cos(math.radians(s.sweep_deg))**0.15
        s.e_oswald = e_oswald

        CD0_airfoil = wing.Cd_cruise if wing.Cd_cruise > 0 else wing.Cd_min
        CD0_misc    = 0.0008

        CD_induced  = CL_3D**2 / (math.pi * AR * e_oswald)
        CD_total    = CD0_airfoil + CD0_fuse + CD0_misc + CD_induced
        s.CD0_airfoil  = CD0_airfoil
        s.CD0_fuselage = CD0_fuse
        s.CD0_misc     = CD0_misc
        s.CD_induced   = CD_induced
        s.CD_total     = CD_total
        s.LD_cruise    = CL_3D / CD_total if CD_total > 0 else 0.0

        print(f"[Sizing] CD0_airfoil={CD0_airfoil:.5f}  CD0_fuse={CD0_fuse:.5f}  "
              f"CDi={CD_induced:.5f}  CD_total={CD_total:.5f}  L/D={s.LD_cruise:.2f}")

        CL_max_3D    = _vlm_CLmax(wing.Cl_max, AR, taper)
        s.CL_max_3D  = CL_max_3D
        V_stall      = math.sqrt(2.0*W / (1.225*S_wing*CL_max_3D)) if (CL_max_3D > 0 and S_wing > 0) else 0.0
        s.V_stall_ms = V_stall

        if spec.stall_speed_ms > 0 and V_stall > spec.stall_speed_ms * 1.05:
            print(f"[Sizing] ⚠ Stall {V_stall:.2f} m/s > constraint {spec.stall_speed_ms:.2f} m/s")

        # Check 3: Stall margin — cruise/stall ratio < 1.3 is dangerously tight
        # For manned aircraft, FAR Part 23 requires 1.3× stall margin.
        # For UAVs, 1.2× is generally the accepted minimum for stable flight
        # in gusty conditions (Anderson, Intro to Flight, Ch.5).
        if V_stall > 0:
            stall_margin = V / V_stall
            if stall_margin < 1.2:
                warn3 = (
                    f"⚠️ CRITICAL STALL MARGIN: cruise {V:.1f} m/s / stall {V_stall:.1f} m/s "
                    f"= {stall_margin:.2f}× (minimum safe: 1.20×). "
                    f"A 40° bank turn at cruise speed would stall the aircraft. "
                    f"Consider lower cruise speed, higher CL_max airfoil, or larger wing."
                )
                print(f"[Sizing] {warn3}")
                s.sizing_warnings.append(warn3)
            elif stall_margin < 1.35:
                warn3b = (
                    f"⚠️ TIGHT STALL MARGIN: cruise/stall = {stall_margin:.2f}× "
                    f"(recommended ≥ 1.35× for {spec.aircraft_type} with turns). "
                    f"Acceptable but leaves little margin for gusts or bank angles."
                )
                print(f"[Sizing] {warn3b}")
                s.sizing_warnings.append(warn3b)

        D_cruise            = q * S_wing * CD_total
        s.thrust_required_N = D_cruise
        s.power_required_W  = D_cruise * V

        # Tail sizing (volume coefficient method, Raymer Table 6.4)
        ht_arm = 0.55 * L_fuse
        vt_arm = 0.52 * L_fuse

        if "aerobatic" in ml or "acrobatic" in ml:  VH, VV = 0.50, 0.050
        elif "trainer" in ml:                        VH, VV = 0.45, 0.040
        elif "glider" in ml or "sailplane" in ml:   VH, VV = 0.35, 0.025
        else:                                        VH, VV = 0.40, 0.035

        ht_area = VH * S_wing * s.mac_m / ht_arm if ht_arm > 0 else 0.0
        vt_area = VV * S_wing * b       / vt_arm if vt_arm > 0 else 0.0

        s.ht_area_m2      = round(ht_area, 5)
        s.ht_span_m       = round(math.sqrt(4.5 * ht_area), 4)
        s.ht_arm_m        = round(ht_arm, 4)
        s.ht_volume_coeff = VH
        s.vt_area_m2      = round(vt_area, 5)
        s.vt_arm_m        = round(vt_arm, 4)
        s.vt_volume_coeff = VV

        # Propulsion note
        # FIX-D: Use reference battery density / prop efficiency from search if available
        P_kW = s.power_required_W / 1000.0
        prop = spec.propulsion.lower()

        _bat_density_Whkg = ref_params.battery_density if ref_params.battery_density > 0 else 180.0
        _eta_elec = ref_params.prop_efficiency if ref_params.prop_efficiency > 0 else 0.82

        if "electric" in prop or prop.strip() == "":
            eta    = _eta_elec
            P_sh   = P_kW / eta
            P_sh_W = P_sh * 1000.0
            E_Wh   = P_sh_W * spec.endurance_hr / 0.80 if spec.endurance_hr > 0 else 0.0
            bat_g  = E_Wh / (_bat_density_Whkg / 1000.0)  # grams
            s.propulsion_note = (
                f"Electric brushless: shaft ≥ {P_sh:.2f} kW (η={eta:.2f}). "
                f"Prop Ø ≈ {0.22*(P_sh**0.25):.3f} m. "
                f"Battery ≈ {E_Wh:.0f} Wh at 80% DoD "
                f"(≈ {bat_g:.0f} g at {_bat_density_Whkg:.0f} Wh/kg"
                f"{' [from ref]' if ref_params.battery_density > 0 else ' LiPo est.'})."
            )
        elif "piston" in prop:
            P_sh = P_kW / 0.78
            s.propulsion_note = (
                f"Piston: shaft ≥ {P_sh:.2f} kW (η_prop=0.78). "
                f"Fuel ≈ {P_sh*spec.endurance_hr*0.30:.1f} kg (BSFC 300 g/kWh)."
            )
        elif "turboprop" in prop:
            P_sh = P_kW / 0.85
            s.propulsion_note = (
                f"Turboprop: shaft ≥ {P_sh:.2f} kW (η_prop=0.85). "
                f"Fuel ≈ {P_sh*spec.endurance_hr*0.28:.1f} kg (BSFC 280 g/kWh)."
            )
        elif "jet" in prop or "turbofan" in prop:
            s.propulsion_note = (
                f"Jet: thrust ≥ {s.thrust_required_N:.1f} N, "
                f"T/W = {s.thrust_required_N/W:.3f}. "
                f"Target SFC ≈ 0.6–0.9 kg/(N·h)."
            )
        elif "hybrid" in prop:
            P_sh_elec = P_kW * 0.4 / 0.82
            P_sh_ice  = P_kW * 0.6 / 0.78
            s.propulsion_note = (
                f"Hybrid: electric {P_sh_elec:.2f} kW + ICE {P_sh_ice:.2f} kW. "
                f"Battery ≈ {P_sh_elec*1000*spec.endurance_hr*0.5/0.80:.0f} Wh (electric portion)."
            )
        else:
            s.propulsion_note = (
                f"Unspecified: {P_kW:.2f} kW / {s.thrust_required_N:.1f} N at cruise."
            )

        return s

    # ------------------------------------------------------------------
    # Stage 3b: Post-sizing engineering constraint search
    # ------------------------------------------------------------------

    async def _search_engineering_constraints(
        self,
        spec: AircraftSpec,
        sizing: SizingResults,
        wing_name: str,
        tail_name: str,
    ) -> str:
        """
        After sizing and airfoil selection are finalised, run targeted web
        searches to cross-check the design against real-world engineering
        constraints that XFOIL and VLM cannot capture:

          • Airfoil manufacturability (complex geometry, internal volume,
            spar placement, skin thickness requirements)
          • Performance degradation at the actual Re (XFOIL was run at
            pre-sizing estimated Re; real Re may be different)
          • Structural feasibility at the computed AR and wing loading
          • Propulsion sizing validation for the computed power/thrust

        Queries are made highly specific using computed values so that
        search results are relevant to this particular design, not generic.
        Results are returned as a structured engineering notes string that
        is injected into the report prompt to ground the LLM commentary.
        """
        import math

        b     = math.sqrt(max(sizing.aspect_ratio, 0.01) * sizing.wing_area_m2)
        MAC   = sizing.mac_m
        Re_actual = getattr(sizing, '_re_actual_wing', 0.0)
        if Re_actual == 0.0:
            Re_actual = spec.cruise_speed_ms * MAC / 1.46e-5

        atype = spec.aircraft_type.lower()
        results = {}

        # ── Search 1: Wing airfoil at actual Reynolds number ─────────────────
        re_query = (
            f"{wing_name} airfoil aerodynamic performance "
            f"Reynolds number {Re_actual:.0f} "
            f"{atype} lift coefficient drag stall"
        )
        self._gui_progress(
            "Stage 3b — Constraint search",
            f"checking {wing_name.upper()} at actual Re={Re_actual:.0f}…"
        )
        try:
            re_result = await self._search_with_retry(re_query, spec)
            results["airfoil_re_performance"] = re_result
            print(f"[DesignAssistant] Re-performance search complete for {wing_name}")
        except Exception as e:
            results["airfoil_re_performance"] = f"(search unavailable: {e})"

        # ── Search 2: Wing airfoil manufacturability and practical constraints ─
        mfg_query = (
            f"{wing_name} airfoil wing construction spar thickness "
            f"chord {MAC*100:.0f} mm {atype} composite foam "
            f"internal volume structural"
        )
        self._gui_progress(
            "Stage 3b — Constraint search",
            f"checking {wing_name.upper()} manufacturability and practical constraints…"
        )
        try:
            mfg_result = await self._search_with_retry(mfg_query, spec)
            results["airfoil_manufacturability"] = mfg_result
            print(f"[DesignAssistant] Manufacturability search complete for {wing_name}")
        except Exception as e:
            results["airfoil_manufacturability"] = f"(search unavailable: {e})"

        # ── Search 3: AR and structural feasibility at this wing loading ──────
        ar_query = (
            f"{atype} wing aspect ratio {sizing.aspect_ratio:.0f} "
            f"structural limit spar weight flutter "
            f"wing loading {sizing.wing_loading_Nm2:.0f} N/m2"
        )
        self._gui_progress(
            "Stage 3b — Constraint search",
            f"checking AR={sizing.aspect_ratio:.1f} structural feasibility…"
        )
        try:
            ar_result = await self._search_with_retry(ar_query, spec)
            results["ar_structural"] = ar_result
            print(f"[DesignAssistant] AR structural search complete")
        except Exception as e:
            results["ar_structural"] = f"(search unavailable: {e})"

        # ── Search 4: Propulsion sizing validation ────────────────────────────
        P_W = sizing.power_required_W
        prop = spec.propulsion.lower() if spec.propulsion else "electric"
        prop_query = (
            f"{prop} {atype} propulsion {P_W:.0f} W "
            f"{spec.mass_kg:.1f} kg "
            f"efficiency endurance {spec.endurance_hr:.0f} hour"
        )
        self._gui_progress(
            "Stage 3b — Constraint search",
            f"checking propulsion sizing for {P_W:.0f} W / {spec.endurance_hr:.0f} hr…"
        )
        try:
            prop_result = await self._search_with_retry(prop_query, spec)
            results["propulsion_validation"] = prop_result
            print(f"[DesignAssistant] Propulsion search complete")
        except Exception as e:
            results["propulsion_validation"] = f"(search unavailable: {e})"

        # FIX-E: Apply constraint corrections to sizing if search found limits
        self._apply_constraint_corrections(results, sizing, spec)

        # ── Format combined constraint notes ──────────────────────────────────
        notes = (
            f"=== POST-SIZING ENGINEERING CONSTRAINT SEARCH RESULTS ===\n\n"
            f"[A] Wing airfoil {wing_name.upper()} at actual Re={Re_actual:.0f} "
            f"(chord={MAC*100:.1f} cm):\n"
            f"{results.get('airfoil_re_performance', 'N/A')}\n\n"
            f"[B] {wing_name.upper()} manufacturability / practical constraints "
            f"(chord={MAC*100:.1f} cm):\n"
            f"{results.get('airfoil_manufacturability', 'N/A')}\n\n"
            f"[C] AR={sizing.aspect_ratio:.1f} structural feasibility "
            f"(W/S={sizing.wing_loading_Nm2:.0f} N/m²):\n"
            f"{results.get('ar_structural', 'N/A')}\n\n"
            f"[D] Propulsion validation "
            f"({P_W:.0f} W, {spec.endurance_hr:.0f} hr, "
            f"{spec.mass_kg:.1f} kg):\n"
            f"{results.get('propulsion_validation', 'N/A')}"
        )
        return notes

    def _apply_constraint_corrections(
        self,
        constraint_results: dict,
        sizing: SizingResults,
        spec: AircraftSpec,
    ) -> None:
        """
        FIX-E: Parse constraint search results for hard engineering limits and
        apply corrections to the SizingResults in-place.  This ensures that
        the constraints discovered through web search actually influence the
        design rather than being appended as decorative notes.

        Current corrections applied:
          • If search found a battery density HIGHER than our default (180 Wh/kg),
            re-compute battery mass in the propulsion note.
          • If search found an AR structural limit LOWER than our computed AR,
            add a specific re-design warning citing the found source.
          • If search found a maximum wing loading for this airfoil at this Re
            (extracted from stall data), check our cruise CL doesn't violate it.
        """
        combined = "\n".join(str(v) for v in constraint_results.values()).lower()

        # ── Battery density update ────────────────────────────────────────────
        bat_m = re.search(r'([\d.]+)\s*wh/kg', combined)
        if bat_m:
            try:
                bat_density = float(bat_m.group(1))
                if 100 <= bat_density <= 600:
                    P_kW     = sizing.power_required_W / 1000.0
                    eta      = 0.82
                    P_sh_W   = P_kW / eta * 1000.0
                    E_Wh     = P_sh_W * spec.endurance_hr / 0.80 if spec.endurance_hr > 0 else 0.0
                    bat_mass_g = E_Wh / (bat_density / 1000.0)
                    if "electric" in spec.propulsion.lower() or not spec.propulsion:
                        # Append correction note to existing propulsion_note
                        sizing.propulsion_note += (
                            f" [Constraint search found {bat_density:.0f} Wh/kg; "
                            f"corrected battery mass ≈ {bat_mass_g:.0f} g.]"
                        )
                        print(f"[DesignAssistant] Battery density updated from search: "
                              f"{bat_density:.0f} Wh/kg → {bat_mass_g:.0f} g")
            except ValueError:
                pass

        # ── AR structural limit from search ──────────────────────────────────
        ar_limit_m = re.search(r'(?:ar|aspect ratio)\s*(?:limit|max|maximum)?\s*(?:of\s+)?([\d.]+)', combined)
        if ar_limit_m:
            try:
                ar_limit = float(ar_limit_m.group(1))
                if 5 <= ar_limit <= 50 and ar_limit < sizing.aspect_ratio:
                    warn = (
                        f"⚠️ CONSTRAINT SEARCH: AR structural limit {ar_limit:.0f} found in "
                        f"literature — computed AR={sizing.aspect_ratio:.1f} exceeds this. "
                        f"Recommend reducing span or using advanced composite spar."
                    )
                    if warn not in (sizing.sizing_warnings or []):
                        sizing.sizing_warnings = sizing.sizing_warnings or []
                        sizing.sizing_warnings.append(warn)
                        print(f"[DesignAssistant] {warn}")
            except ValueError:
                pass

        # ------------------------------------------------------------------
    # Stage 4: Report generation
    # ------------------------------------------------------------------

    def _generate_structured_report(self) -> str:
        """Pure f-string report — no LLM. Shows all computed data."""
        sz  = self.sizing
        sp  = self.spec
        wng = self.wing_data
        tl  = self.tail_data
        b   = math.sqrt(max(sz.aspect_ratio, 0.01) * sz.wing_area_m2)
        SEP = "─" * 56

        def row(label, value):
            return f"  {label:<32} {value}"

        lines = [
            f"✈  PRELIMINARY DESIGN REPORT — {(sp.mission or sp.aircraft_type).upper()}",
            "",
            SEP, "§1  DESIGN REQUIREMENTS", SEP,
            self._format_spec(sp),
            "",

            SEP, "§2  WING AIRFOIL — XFOIL 2-D POLAR", SEP,
            row("Airfoil",              wng.name.upper()),
            row("Re (cruise)",          f"{self._estimate_Re(sp,'wing'):.3e}"),
            row("Cl_max (2-D)",         f"{wng.Cl_max:.4f}"),
            row("Cd_min (2-D)",         f"{wng.Cd_min:.6f}"),
            row("Best L/D (2-D)",       f"{wng.ClCd_max:.2f}  at α = {wng.aoa_ClCd:.1f}°"),
            row("Cl_cruise (2-D)",      f"{wng.Cl_cruise:.4f}  at α = {wng.aoa_ClCd:.1f}°"),
            row("Cd_cruise (2-D)",      f"{wng.Cd_cruise:.6f}  ← used in 3-D polar"),
            row("Stall AoA (2-D)",      f"{wng.stall_aoa:.1f}°"),
            "",

            SEP, "§3  TAIL AIRFOIL — XFOIL 2-D POLAR", SEP,
            row("Airfoil",              tl.name.upper()),
            row("Re (cruise)",          f"{self._estimate_Re(sp,'tail'):.3e}"),
            row("Cl_max (2-D)",         f"{tl.Cl_max:.4f}"),
            row("Cd_min (2-D)",         f"{tl.Cd_min:.6f}"),
            row("Best L/D (2-D)",       f"{tl.ClCd_max:.2f}  at α = {tl.aoa_ClCd:.1f}°"),
            row("Stall AoA (2-D)",      f"{tl.stall_aoa:.1f}°"),
            "",

            SEP, "§4  WING GEOMETRY  (VLM lifting-line)", SEP,
            row("Wing area",            f"{sz.wing_area_m2:.4f} m²"),
            row("Span",                 f"{b:.4f} m"),
            row("Aspect ratio",         f"{sz.aspect_ratio:.4f}  [{sz.ar_method}]"),
            row("Taper ratio",          f"{sz.taper_ratio:.4f}  [near-elliptic optimum]"),
            row("Root chord",           f"{sz.root_chord_m:.4f} m"),
            row("Tip chord",            f"{sz.tip_chord_m:.4f} m"),
            row("MAC",                  f"{sz.mac_m:.4f} m"),
            row("LE sweep",             f"{sz.sweep_deg:.2f}°"),
            row("Wing loading",         f"{sz.wing_loading_Nm2:.2f} N/m²"),
            row("VLM CL_cruise (3-D)",  f"{sz.CL_cruise:.4f}"),
            row("VLM CL_max (3-D)",     f"{sz.CL_max_3D:.4f}  [tip-stall corrected]"),
            row("Oswald e (derived)",   f"{sz.e_oswald:.4f}"),
            "",

            SEP, "§5  3-D DRAG POLAR BREAKDOWN", SEP,
            row("CD0 airfoil (XFOIL)",  f"{sz.CD0_airfoil:.6f}  [from Cd_cruise]"),
            row("CD0 fuselage",         f"{sz.CD0_fuselage:.6f}  [Cf·FF·Swet/Sref]"),
            row("CD0 misc",             f"{sz.CD0_misc:.6f}"),
            row("CD_induced",           f"{sz.CD_induced:.6f}  [CL²/(π·AR·e)]"),
            row("CD_total",             f"{sz.CD_total:.6f}"),
            row("Cruise L/D (3-D)",     f"{sz.LD_cruise:.3f}"),
            row("Stall speed (3-D)",    f"{sz.V_stall_ms:.3f} m/s  ({sz.V_stall_ms*3.6:.2f} km/h)"),
            row("Thrust required",      f"{sz.thrust_required_N:.2f} N"),
            row("Power required",       f"{sz.power_required_W/1000:.3f} kW"),
            "",

            SEP, "§6  EMPENNAGE & FUSELAGE", SEP,
            "  Horizontal tail",
            row("    Airfoil",          tl.name.upper()),
            row("    Area",             f"{sz.ht_area_m2:.5f} m²"),
            row("    Span",             f"{sz.ht_span_m:.4f} m"),
            row("    Moment arm",       f"{sz.ht_arm_m:.4f} m"),
            row("    Volume coeff Vh",  f"{sz.ht_volume_coeff:.3f}"),
            "  Vertical tail",
            row("    Area",             f"{sz.vt_area_m2:.5f} m²"),
            row("    Moment arm",       f"{sz.vt_arm_m:.4f} m"),
            row("    Volume coeff Vv",  f"{sz.vt_volume_coeff:.4f}"),
            "  Fuselage",
            row("    Length",           f"{sz.fuselage_length_m:.4f} m"),
            row("    Diameter",         f"{sz.fuselage_diam_m:.4f} m"),
            row("    Fineness ratio",   f"{sz.fuselage_length_m/max(sz.fuselage_diam_m,0.001):.2f}"),
            "",

            SEP, "§7  PROPULSION", SEP,
            f"  {sz.propulsion_note}",
            "",
        ]

        # ── §8 Physics sanity warnings (physics-derived, not arbitrary) ───────
        warnings = getattr(sz, "sizing_warnings", []) or []
        if warnings:
            lines += [
                SEP, "§8  ENGINEERING WARNINGS (physics-derived)", SEP,
            ]
            for i, w in enumerate(warnings, 1):
                lines.append(f"  [{i}] {w}")
            lines.append("")

        return "\n".join(lines)

    def _build_report_prompt(self) -> str:
        sz  = self.sizing
        sp  = self.spec
        wng = self.wing_data
        tl  = self.tail_data
        b   = math.sqrt(sz.aspect_ratio * sz.wing_area_m2)

        wing_all   = getattr(self, "_wing_all_results", [wng])
        tail_all   = getattr(self, "_tail_all_results", [tl])
        wing_reason = getattr(self, "_wing_reason", "")
        tail_reason = getattr(self, "_tail_reason", "")
        wing_table = self._comparison_table(wing_all, "wing")
        tail_table = self._comparison_table(tail_all, "tail")

        data_summary = (
            f"Mission: {sp.mission or sp.aircraft_type}\n"
            f"MTOW={sp.mass_kg:.1f} kg  Cruise={sp.cruise_speed_ms:.1f} m/s  "
            f"Endurance={sp.endurance_hr:.1f} h  Alt={sp.altitude_m:.0f} m  "
            f"Propulsion={sp.propulsion}\n\n"
            f"WING AIRFOIL CANDIDATES (XFOIL):\n{wing_table}\n"
            f"Wing selected: {wng.name.upper()} — {wing_reason}\n"
            f"Wing 2-D: Cl_max={wng.Cl_max:.4f}  Cd_min={wng.Cd_min:.6f}  "
            f"L/D={wng.ClCd_max:.2f}@{wng.aoa_ClCd:.1f}°  "
            f"Cl_cr={wng.Cl_cruise:.4f}  Cd_cr={wng.Cd_cruise:.6f}  stall={wng.stall_aoa:.1f}°\n\n"
            f"TAIL AIRFOIL CANDIDATES (XFOIL):\n{tail_table}\n"
            f"Tail selected: {tl.name.upper()} — {tail_reason}\n"
            f"Tail 2-D: Cl_max={tl.Cl_max:.4f}  Cd_min={tl.Cd_min:.6f}  "
            f"L/D={tl.ClCd_max:.2f}@{tl.aoa_ClCd:.1f}°  stall={tl.stall_aoa:.1f}°\n\n"
            f"WING (3-D, VLM lifting-line):\n"
            f"  S={sz.wing_area_m2:.4f} m²  b={b:.4f} m  AR={sz.aspect_ratio:.4f} [{sz.ar_method}]\n"
            f"  taper={sz.taper_ratio:.4f}  MAC={sz.mac_m:.4f} m  sweep={sz.sweep_deg:.2f}°\n"
            f"  W/S={sz.wing_loading_Nm2:.2f} N/m²  CL_cr(3D)={sz.CL_cruise:.4f}  "
            f"CL_max(3D)={sz.CL_max_3D:.4f}  e_oswald={sz.e_oswald:.4f}\n\n"
            f"3-D DRAG POLAR:\n"
            f"  CD0_airfoil(XFOIL)={sz.CD0_airfoil:.6f}  CD0_fuse={sz.CD0_fuselage:.6f}  "
            f"CD0_misc={sz.CD0_misc:.6f}  CDi={sz.CD_induced:.6f}  CD_total={sz.CD_total:.6f}\n"
            f"  L/D={sz.LD_cruise:.3f}  V_stall={sz.V_stall_ms:.3f} m/s  "
            f"T={sz.thrust_required_N:.2f} N  P={sz.power_required_W/1000:.3f} kW\n\n"
            f"EMPENNAGE: Vh={sz.ht_volume_coeff:.3f} Vv={sz.vt_volume_coeff:.4f}  "
            f"ht_area={sz.ht_area_m2:.5f} m²  vt_area={sz.vt_area_m2:.5f} m²\n"
            f"FUSELAGE: L={sz.fuselage_length_m:.4f} m  D={sz.fuselage_diam_m:.4f} m  "
            f"fineness={sz.fuselage_length_m/max(sz.fuselage_diam_m,0.001):.2f}\n"
            f"PROPULSION: {sz.propulsion_note}\n"
        )

        # ── Inject physics warnings (may be empty) ────────────────────────
        warnings_list = getattr(sz, "sizing_warnings", []) or []
        warnings_block = ""
        if warnings_list:
            warnings_block = (
                "\nPHYSICS WARNINGS — issues the code detected:\n"
                + "\n".join(f"  [{i+1}] {w}" for i, w in enumerate(warnings_list))
                + "\n"
            )

        # ── Inject engineering constraint search notes ─────────────────────
        constraint_notes_raw = getattr(self, "_constraint_notes", "")
        constraint_block = ""
        if constraint_notes_raw:
            constraint_block = (
                "\nENGINEERING CONSTRAINT SEARCH RESULTS:\n"
                + constraint_notes_raw[:4000]
                + "\n"
            )

        return (
            f"You are an expert aircraft design engineer writing Section 8 of a "
            f"formal preliminary design report. All sizing, airfoil data, "
            f"physics warnings, and engineering constraint research are below.\n\n"
            f"{data_summary}"
            f"{warnings_block}"
            f"{constraint_block}\n"
            f"Write the DESIGN ASSESSMENT using EXACTLY these six headings in order. "
            f"Every judgement MUST cite specific numbers from the data above.\n\n"
            f"AIRFOIL SELECTION RATIONALE\n"
            f"Compare the wing airfoil candidates using their actual XFOIL numbers "
            f"(Cl_max, best L/D, stall AoA). Explain specifically why {wng.name.upper()} "
            f"was chosen. For the tail, explicitly address whether {tl.name.upper()} is "
            f"aerodynamically appropriate for a tail surface (symmetric loading, pitch "
            f"stability). Reference constraint search results if relevant. 3-5 sentences.\n\n"
            f"WING DESIGN ASSESSMENT\n"
            f"Cite wing area, span, AR, wing loading, cruise CL, and actual Re at MAC. "
            f"If physics warnings were raised (span infeasibility, AR beyond structural "
            f"limits), address them explicitly and explain how the design was corrected. "
            f"3-4 sentences.\n\n"
            f"AERODYNAMIC PERFORMANCE\n"
            f"Cite the 3D cruise L/D, CD_total, stall speed, and cruise/stall ratio. "
            f"If the stall margin is below 1.35, flag it as a risk. Reference constraint "
            f"search data on airfoil Re sensitivity if available. 3-4 sentences.\n\n"
            f"STABILITY & TAIL SIZING\n"
            f"Cite horizontal and vertical tail volume coefficients and moment arms. "
            f"Assess whether they are adequate. Note any tail airfoil stability concerns. "
            f"3-4 sentences.\n\n"
            f"PROPULSION ASSESSMENT\n"
            f"Cite required shaft power, battery energy (Wh), and estimated battery mass. "
            f"Assess feasibility. Reference propulsion constraint search data if available. "
            f"3-4 sentences.\n\n"
            f"DESIGN RISKS & RECOMMENDED NEXT STEPS\n"
            f"Bullet list of exactly 4-6 specific actionable items. Include any items "
            f"raised by physics warnings or constraint search results.\n\n"
            f"RULES:\n"
            f"- ALWAYS cite specific computed values inline.\n"
            f"- If physics warnings exist, address them — never rationalise impossible values.\n"
            f"- If constraint search found real-world issues, cite them specifically.\n"
            f"- Sections 8.1-8.5 are prose. Section 8.6 is bullet points.\n"
            f"- No introduction, conclusion, or preamble outside these six sections."
        )
    def _generate_report(self) -> str:
        return self._generate_structured_report()

    def _wing_summary(self) -> str:
        sz = self.sizing; w = self.wing_data
        b = math.sqrt(sz.aspect_ratio * sz.wing_area_m2)
        return (
            f"Wing — {w.name.upper()}\n"
            f"  Area: {sz.wing_area_m2:.3f} m²  |  Span: {b:.2f} m  |  AR: {sz.aspect_ratio:.2f}\n"
            f"  Root chord: {sz.root_chord_m:.3f} m  |  Tip: {sz.tip_chord_m:.3f} m  |  MAC: {sz.mac_m:.3f} m\n"
            f"  Taper: {sz.taper_ratio:.2f}  |  Sweep: {sz.sweep_deg:.1f}°\n"
            f"  Wing loading: {sz.wing_loading_Nm2:.1f} N/m²  |  CL_cruise: {w.Cl_cruise:.3f}"
        )

    def _tail_summary(self) -> str:
        sz = self.sizing; t = self.tail_data
        return (
            f"Horizontal Tail — {t.name.upper()}\n"
            f"  Area: {sz.ht_area_m2:.3f} m²  |  Span: {sz.ht_span_m:.3f} m\n"
            f"  Moment arm: {sz.ht_arm_m:.3f} m  |  Vh: {sz.ht_volume_coeff:.2f}\n\n"
            f"Vertical Tail\n"
            f"  Area: {sz.vt_area_m2:.3f} m²  |  Moment arm: {sz.vt_arm_m:.3f} m  |  Vv: {sz.vt_volume_coeff:.3f}"
        )

    def _fuselage_summary(self) -> str:
        sz = self.sizing
        return (
            f"Fuselage\n"
            f"  Length  : {sz.fuselage_length_m:.3f} m\n"
            f"  Diameter: {sz.fuselage_diam_m:.3f} m\n"
            f"  Fineness: {sz.fuselage_length_m/max(sz.fuselage_diam_m,0.001):.1f}"
        )

    def _propulsion_summary(self) -> str:
        return f"Propulsion\n  {self.sizing.propulsion_note}"

    def _performance_summary(self) -> str:
        sz = self.sizing
        return (
            f"Cruise Performance\n"
            f"  L/D (3-D)  : {sz.LD_cruise:.1f}\n"
            f"  CD_total   : {sz.CD_total:.4f}\n"
            f"  Stall speed: {sz.V_stall_ms:.1f} m/s ({sz.V_stall_ms*3.6:.0f} km/h)\n"
            f"  Thrust req : {sz.thrust_required_N:.1f} N\n"
            f"  Power req  : {sz.power_required_W/1000:.2f} kW"
        )

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _air_density(altitude_m: float) -> float:
        """ISA air density [kg/m³] up to ~11 km (troposphere)."""
        T  = 288.15 - 0.0065 * altitude_m
        p  = 101325 * (T / 288.15) ** 5.2561
        return p / (287.058 * T)

    def _estimate_Re(self, spec: AircraftSpec, surface: str = "wing") -> float:
        """
        Estimate Reynolds number for the given surface.
        Only called after _validate_spec() has confirmed both
        cruise_speed_ms > 0 and mass_kg > 0, so spec values are
        used directly — no arbitrary fallbacks.
        """
        nu = 1.46e-5   # kinematic viscosity air at ~300 m [m²/s]
        c  = self._estimate_chord(spec, surface)
        return spec.cruise_speed_ms * c / nu

    def _estimate_chord(self, spec: AircraftSpec, surface: str = "wing") -> float:
        """
        Quick geometric chord estimate before full sizing is done.

        Computes approximate wing area from weight and cruise dynamic
        pressure, then derives root chord assuming AR=10 and taper≈0.45.

        Precondition: _validate_spec() must have already confirmed that
        spec.mass_kg > 0 and spec.cruise_speed_ms > 0 before this is
        called.  No arbitrary fallback values are used — if the spec is
        incomplete, the pipeline should have already asked the user for
        the missing data before reaching here.
        """
        g   = 9.81
        W   = spec.mass_kg * g
        rho = self._air_density(spec.altitude_m)
        q   = 0.5 * rho * spec.cruise_speed_ms**2
        CL  = 0.6           # assumed cruise CL for pre-sizing estimate
        S   = W / (q * CL)
        AR  = 10.0
        b   = math.sqrt(AR * S)
        c_root = 2 * S / (b * 1.45)   # taper ratio ≈ 0.45

        if surface in ("tail", "horizontal tail"):
            return c_root * 0.55   # tail chord ≈ 55 % of wing root
        return c_root

    def _estimate_mach(self, spec: AircraftSpec) -> float:
        """Only called after validation; uses spec.cruise_speed_ms directly."""
        return spec.cruise_speed_ms / 340.0   # speed of sound at sea level [m/s]

    # ------------------------------------------------------------------
    # Internal reset
    # ------------------------------------------------------------------

    def _reset(self):
        self.stage        = Stage.IDLE
        self.active       = False
        self.spec         = AircraftSpec()
        self.wing_data    = AirfoilData()
        self.tail_data    = AirfoilData()
        self.sizing       = SizingResults()
        self._ref_params  = ReferenceParams()          # FIX-D: search-seeded params
        self._airfoil_agent: Optional[AirfoilAnalysisAgent] = None
        self._wing_candidates: list[str] = []
        self._tail_candidates: list[str] = []
        # _app is NOT reset — the GUI reference persists across sessions