#!/usr/bin/env python3

import threading
import time
import serial
import serial.tools.list_ports
import logging
import os
import random
import math
import numpy as np
from typing import Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import subprocess
import signal
import sys
from collections import defaultdict, deque
import json
import heapq  # Added for scheduler

import json, os, logging, sys

# set up logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'templates')
#_last_mtime = None

def load_config():
    #global _last_mtime, _cfg, _sound_cfg, _params
    #mtime = os.path.getmtime(CONFIG_PATH)
    #if _last_mtime is None or mtime > _last_mtime:
    logging.info("Reloading config.json (mtime changed)")
    with open(CONFIG_PATH, 'r') as f:
        _cfg = json.load(f)
    _sound_cfg = _cfg['sound_model']
    _params = SoundParams(_sound_cfg)
    #_last_mtime = mtime

try:
    with open(CONFIG_PATH, 'r') as f:
        _cfg = json.load(f)
    # sanity check that your section exists
    if 'sound_model' not in _cfg:
        raise KeyError("'sound_model' section missing in config")
    _sound_cfg = _cfg['sound_model']
except FileNotFoundError:
    logger.error("Config file not found at %s", CONFIG_PATH)
    sys.exit(1)
except (json.JSONDecodeError, KeyError) as e:
    logger.error("Error parsing config: %s", e)
    sys.exit(2)
else:
    logger.info("Successfully loaded configuration from %s", CONFIG_PATH)
    # optionally print out which sub-keys you have loaded:
    logger.debug("sound_model keys: %s", list(_sound_cfg.keys()))


# Helper for accessing sound-model parameters

# Helper for accessing sound-model parameters (expanded with more accessors)
class SoundParams:
    def __init__(self, cfg):
        self.hammer = cfg.get('hammer', {})
        self.string = cfg.get('string', {})
        self.coupling = cfg.get('coupling', {})
        self.soundboard = cfg.get('soundboard', {})
        self.case = cfg.get('case', {})
        self.extended = cfg.get('extended', {})
        self.transient = cfg.get('transient', {})
        self.polyphony = cfg.get('polyphony', {})

    def felt_density(self):
        r = self.hammer.get('felt_density', {'min':0.1,'max':1.0})
        return random.uniform(r.get('min',0.1), r.get('max',1.0))

    def hardness_layers(self):
        return self.hammer.get('hardness_layers', [])

    def age_compression_rate(self):
        return self.hammer.get('age_compression_rate', 0.01)

    def asymmetry_factor(self):
        return self.hammer.get('asymmetry_factor', 0.0)

    def base_tension(self):
        return self.string.get('base_tension', 180.0)

    def pitch_drift_per_deg(self):
        return self.string.get('temperature_pitch_drift_cents_per_deg', 0.12)

    def inharmonicity_coef(self, register):
        coefs = self.string.get('inharmonicity_coefs', {})
        return coefs.get(register, 0.00012)

    def coupling_strengths(self):
        return self.coupling

    def soundboard_modes(self):
        return self.soundboard.get('modes', [])

    def case_modes(self):
        return self.case.get('modes', [])

    def harmonic_threshold(self):
        return self.extended.get('harmonic_detection_threshold', 0.01)

    def preparation_effect(self, name):
        return self.extended.get('preparation_effects', {}).get(name, 1.0)

    def attack_phases(self):
        return self.transient.get('attack_phases', [])

    def decay_rates(self):
        return self.transient.get('harmonic_decay_rates', [])

    def max_active_notes(self):
        return self.polyphony.get('max_active_notes', 32)

    def capture_resonance(self):
        return self.polyphony.get('resonance_capture', True)


# MIDI engine for per-note pitch-bend via channel pooling
class MidiEngine:
    def __init__(self, virtual_port_name='PianoPi', max_channels=64, pitch_bend_range=2.0):
        self.virtual_port_name = virtual_port_name
        self.max_channels = int(max_channels)
        self.pitch_bend_range = float(pitch_bend_range)
        self.base_channel = 0
        self.channels = list(range(self.base_channel, min(256, self.base_channel + self.max_channels)))
        self.channel_pool = deque(self.channels)
        self.note_channel_map = {}
        self.channel_note_map = {}
        try:
            self.output = mido.open_output(self.virtual_port_name, virtual=True)
            logger.info("Created virtual MIDI out port: %s", self.virtual_port_name)
        except Exception as e:
            logger.warning("Could not create virtual MIDI port: %s", e)
            self.output = None

        # set pitch bend range on init
        self._set_pitch_bend_range_all(self.pitch_bend_range)

    def _set_pitch_bend_range_all(self, semitones):
        pb = int(round(semitones))
        for ch in range(0, min(128, len(self.channels))):
            try:
                self.safe_send(mido.Message('control_change', channel=ch, control=101, value=0))
                self.safe_send(mido.Message('control_change', channel=ch, control=100, value=0))
                self.safe_send(mido.Message('control_change', channel=ch, control=6, value=pb))
                self.safe_send(mido.Message('control_change', channel=ch, control=38, value=0))
                self.safe_send(mido.Message('control_change', channel=ch, control=101, value=127))
            except Exception:
                pass

    def safe_send(self, msg):
        if self.output:
            try:
                self.output.send(msg)
            except Exception as e:
                logger.debug("MIDI send error: %s", e)

    def allocate_channel_for_note(self, note):
        if note in self.note_channel_map:
            return self.note_channel_map[note]
        if self.channel_pool:
            ch = self.channel_pool.popleft()
            self.note_channel_map[note] = ch
            self.channel_note_map[ch] = note
            return ch
        # reuse a random channel if pool exhausted
        ch = random.choice(self.channels)
        prev_note = self.channel_note_map.get(ch)
        if prev_note is not None:
            self.note_channel_map.pop(prev_note, None)
        self.note_channel_map[note] = ch
        self.channel_note_map[ch] = note
        return ch

    def release_channel(self, note):
        ch = self.note_channel_map.pop(note, None)
        if ch is not None:
            self.channel_note_map.pop(ch, None)
            if ch in self.channels:
                self.channel_pool.append(ch)

    def send_note_on(self, note, velocity, pitch_bend_cents=0.0, brightness_cc=None):
        ch = self.allocate_channel_for_note(note)
        # set pitch bend
        self.send_pitch_bend(ch, pitch_bend_cents)
        # set brightness CC74 if provided
        if brightness_cc is not None:
            self.safe_send(mido.Message('control_change', channel=ch, control=74, value=int(clamp(brightness_cc,0,127))))
        self.safe_send(mido.Message('note_on', channel=ch, note=note, velocity=int(clamp(velocity,0,127))))

    def send_note_off(self, note):
        ch = self.note_channel_map.get(note)
        if ch is None:
            self.safe_send(mido.Message('note_off', channel=0, note=note, velocity=0))
            return
        self.safe_send(mido.Message('note_off', channel=ch, note=note, velocity=0))
        self.release_channel(note)

    def send_pitch_bend(self, channel, cents):
        pb_range = max(0.1, self.pitch_bend_range)
        semitone_shift = cents / 100.0
        ratio = clamp(semitone_shift / pb_range, -1.0, 1.0)
        raw = int(round(ratio * 8191))
        self.safe_send(mido.Message('pitchwheel', channel=channel, pitch=raw))

    def send_cc(self, channel, cc, value):
        self.safe_send(mido.Message('control_change', channel=channel, control=cc, value=int(clamp(value,0,127))))

# Internal audio engine (optional) - simple additive / partials + inharmonicity
class InternalAudioEngine:
    def __init__(self, sample_rate=44100, blocksize=512):
        try:
            import sounddevice as sd_local
            import numpy as np_local
        except Exception as e:
            logger.warning("Internal audio engine dependencies missing: %s", e)
            raise
        self.sd = sd_local
        self.np = np_local
        self.sr = int(sample_rate)
        self.blocksize = int(blocksize)
        self.voices = {}
        self.lock = threading.RLock()
        self.stream = self.sd.OutputStream(samplerate=self.sr, channels=2, blocksize=self.blocksize, callback=self._callback)
        self.stream.start()
        logger.info("Internal audio engine started (sr=%s block=%s)", self.sr, self.blocksize)

    def _callback(self, outdata, frames, time_info, status):
        buf = self.np.zeros(frames, dtype=self.np.float32)
        with self.lock:
            to_remove = []
            for note, voice_list in list(self.voices.items()):
                newlist = []
                for v in voice_list:
                    chunk = v.render(frames)
                    buf += chunk
                    if not v.is_dead():
                        newlist.append(v)
                if newlist:
                    self.voices[note] = newlist
                else:
                    to_remove.append(note)
            for n in to_remove:
                self.voices.pop(n, None)
        stereo = self.np.vstack([buf, buf]).T * 0.7
        stereo = self.np.clip(stereo, -1.0, 1.0)
        outdata[:] = stereo.astype(self.np.float32)

    def note_on(self, note, vel, params):
        with self.lock:
            v = GranularPianoVoice(note, vel, params, self.sr)
            self.voices.setdefault(note, []).append(v)

    def note_off(self, note):
        with self.lock:
            if note in self.voices:
                for v in self.voices[note]:
                    v.note_off()
_params = SoundParams(_sound_cfg)

# MIDI libraries
try:
    import mido
    import rtmidi
    import numpy as np
    from scipy import signal as scipy_signal
    from scipy.spatial.distance import euclidean
except ImportError:
    print("Installing required libraries...")
    subprocess.run([sys.executable, "-m", "pip", "install", "mido", "python-rtmidi", "numpy", "scipy"])
    import mido
    import rtmidi
    import numpy as np
    from scipy import signal as scipy_signal
    from scipy.spatial.distance import euclidean


class DeviceType(Enum):
    MASTER_BASS_PEDALS = "MASTER_BASS_PEDALS"
    SLAVE_TREBLE = "SLAVE_TREBLE"
    CONTROLLER = "CONTROL"

@dataclass
class KeyEvent:
    device_id: str
    key_number: int
    section: str
    contact: str  # "make" or "break"
    action: str   # "press" or "release"
    velocity: int
    timestamp: float


@dataclass
class PedalEvent:
    device_id: str
    pedal_number: int
    pedal_name: str
    action: str   # "press" or "release"
    timestamp: float


@dataclass
class HammerState:
    """Represents the physical state of a piano hammer"""
    position: float  # 0.0 (rest) to 1.0 (strike)
    velocity: float
    felt_density: float  # Affects tone color
    wear_level: float    # Affects consistency
    temperature: float   # Affects felt properties

@dataclass
class VolumeEvent:
    """Volume control event"""
    device_id: str
    raw_value: int          # 0-4095 from ESP32
    volume_percent: float   # 0-100% calculated
    muted: bool
    timestamp: float
    
@dataclass
class StringState:
    """Represents the physical state of a piano string"""
    tension: float       # Current tension (affects pitch)
    temperature: float   # Affects tuning
    age: float          # Affects inharmonicity
    coupling_strength: float  # How well it couples with soundboard
    last_strike_time: float
    decay_coefficient: float


class AdvancedKeyActionEngine:
    """Ultra-realistic key action modeling with escapement and after-touch"""
    
    def __init__(self):
        self.key_states = {}  # key_id -> detailed state
        self.escapement_points = self._build_escapement_map()
        self.key_weights = self._build_advanced_key_weights()
        self.action_noise_samples = self._generate_action_noise()
        
        # Physical parameters
        self.escapement_threshold = 0.85  # Point where hammer escapes
        self.aftertouch_depth = 0.15     # Additional travel after escapement
        self.letoff_variations = {}      # Per-key regulation variations
        
    def _build_escapement_map(self) -> Dict[int, Dict]:
        """Build escapement characteristics for each key"""
        escapement_map = {}
        
        for note in range(21, 109):
            if note <= 32:  # Bass - heavier action, later escapement
                escapement_point = 0.88
                resistance_curve = "exponential"
                letoff_distance = 2.5  # mm
            elif note <= 60:  # Mid-range - standard action
                escapement_point = 0.85
                resistance_curve = "sigmoid"
                letoff_distance = 2.0
            else:  # Treble - lighter action, earlier escapement
                escapement_point = 0.82
                resistance_curve = "linear"
                letoff_distance = 1.5
            
            # Add slight randomization for realism
            variation = random.uniform(-0.02, 0.02)
            escapement_point += variation
            
            escapement_map[note] = {
                'escapement_point': max(0.75, min(0.95, escapement_point)),
                'resistance_curve': resistance_curve,
                'letoff_distance': letoff_distance,
                'regulation_variation': random.uniform(0.95, 1.05)
            }
            
        return escapement_map
    
    def _build_advanced_key_weights(self) -> Dict[int, Dict]:
        weights = {}
        
        for note in range(21, 109):
            # Base weight distribution (unchanged)
            if note <= 28:
                base_weight = 50.0 + (28 - note) * 2.0  # grams
            elif note <= 48:
                base_weight = 45.0 + (48 - note) * 0.5
            elif note <= 60:
                base_weight = 42.0 + (60 - note) * 0.2
            elif note <= 72:
                base_weight = 38.0 - (note - 60) * 0.3
            else:
                base_weight = 35.0 - (note - 72) * 0.4
            
            base_weight = max(25.0, min(70.0, base_weight))
            
            # Add resistance_profile: a list of resistance values at different positions (0-1)
            # Example: starts low, increases to base_weight, then drops slightly (simulating escapement letoff)
            resistance_profile = [
                base_weight * (0.5 + i * 0.1) for i in range(8)  # Build up to ~1.3x base
            ] + [base_weight * 0.7, base_weight * 0.5]  # Drop after escapement
            
            weights[note] = {
                'base_weight': base_weight,
                'resistance_profile': resistance_profile
            }
        return weights
    
    def _generate_action_noise(self) -> Dict[str, np.ndarray]:
        """Generate realistic mechanical noise samples"""
        sample_rate = 44100
        
        # Key click (escapement)
        t_click = np.linspace(0, 0.01, int(sample_rate * 0.01))
        click_noise = np.random.normal(0, 0.1, len(t_click)) * np.exp(-t_click * 300)
        
        # Key thump (bottom out)
        t_thump = np.linspace(0, 0.02, int(sample_rate * 0.02))
        thump_noise = np.random.normal(0, 0.2, len(t_thump)) * np.exp(-t_thump * 150)
        
        # Key release click
        t_release = np.linspace(0, 0.005, int(sample_rate * 0.005))
        release_noise = np.random.normal(0, 0.05, len(t_release)) * np.exp(-t_release * 400)
        
        return {
            'escapement_click': click_noise,
            'bottom_thump': thump_noise,
            'release_click': release_noise
        }
    
    def process_key_motion(self, key_id: str, contact: str, timestamp: float, 
                          midi_note: int) -> Dict:
        """Process realistic key motion with escapement modeling"""
        
        if key_id not in self.key_states:
            self.key_states[key_id] = {
                'position': 0.0,
                'velocity': 0.0,
                'last_update': timestamp,
                'escapement_triggered': False,
                'aftertouch_active': False,
                'resistance_level': 0.0
            }
        
        key_state = self.key_states[key_id]
        dt = timestamp - key_state['last_update']
        key_state['last_update'] = timestamp
        
        # Get key characteristics
        escapement_data = self.escapement_points.get(midi_note, self.escapement_points[60])
        weight_data = self.key_weights.get(midi_note, self.key_weights[60])
        
        # Calculate position based on contact
        if contact == "make":
            target_position = 0.5  # Key partially down
        elif contact == "break":
            target_position = 1.0  # Key fully down
        else:
            target_position = 0.0  # Key up
        
        # Simulate realistic motion with physics
        position_diff = target_position - key_state['position']
        
        # Apply resistance profile
        progress = key_state['position']
        resistance_idx = min(len(weight_data['resistance_profile']) - 1, 
                           int(progress * len(weight_data['resistance_profile'])))
        current_resistance = weight_data['resistance_profile'][resistance_idx]
        
        # Calculate velocity with resistance
        force_factor = 1.0 / (1.0 + current_resistance * 2.0)
        key_state['velocity'] = position_diff * force_factor / max(dt, 0.001)
        key_state['position'] = min(1.0, max(0.0, key_state['position'] + 
                                           key_state['velocity'] * dt))
        
        # Check for escapement
        if (not key_state['escapement_triggered'] and 
            key_state['position'] >= escapement_data['escapement_point']):
            key_state['escapement_triggered'] = True
            key_state['resistance_level'] = 0.3  # Sudden resistance drop
        
        # After-touch simulation
        if key_state['position'] >= 0.9:
            key_state['aftertouch_active'] = True
            aftertouch_depth = (key_state['position'] - 0.9) / 0.1
        else:
            aftertouch_depth = 0.0
        
        return {
            'position': key_state['position'],
            'velocity': abs(key_state['velocity']),
            'escapement_triggered': key_state['escapement_triggered'],
            'aftertouch_depth': aftertouch_depth,
            'resistance_level': current_resistance,
            'mechanical_noise': self._generate_mechanical_noise(key_state, midi_note)
        }
    
    def _generate_mechanical_noise(self, key_state: Dict, midi_note: int) -> List[Tuple[str, float]]:
        """Generate appropriate mechanical noises"""
        noises = []
        
        if key_state['escapement_triggered'] and key_state['position'] > 0.8:
            intensity = min(1.0, key_state['velocity'] * 2.0)
            noises.append(('escapement_click', intensity * 0.3))
        
        if key_state['position'] >= 0.95:
            intensity = key_state['velocity']
            noises.append(('bottom_thump', intensity * 0.2))
        
        if key_state['position'] < 0.1 and key_state['velocity'] > 0.1:
            noises.append(('release_click', 0.1))
        
        return noises


class AdvancedHammerModelingEngine:
    """Ultra-realistic hammer modeling with voicing and wear simulation"""
    
    def __init__(self):
        self.hammer_states = {}  # note -> HammerState
        self.voicing_profiles = self._build_voicing_profiles()
        self.hammer_wear_patterns = defaultdict(float)
        self.temperature = 20.0  # Celsius
        self.humidity = 50.0     # Percent
        
    def _build_voicing_profiles(self) -> Dict[int, Dict]:
        """Build comprehensive hammer voicing profiles"""
        profiles = {}
        
        for note in range(21, 109):
            if note <= 32:  # Bass hammers - large, soft felt
                felt_density = 0.6 + random.uniform(-0.1, 0.1)
                strike_area = 3.5  # cm²
                hardness_gradient = [0.4, 0.6, 0.8]  # Soft to hard layers
            elif note <= 60:  # Mid-range - balanced
                felt_density = 0.75 + random.uniform(-0.1, 0.1)
                strike_area = 2.8
                hardness_gradient = [0.5, 0.75, 0.9]
            else:  # Treble - small, harder felt
                felt_density = 0.9 + random.uniform(-0.05, 0.05)
                strike_area = 1.2
                hardness_gradient = [0.7, 0.85, 0.95]
            
            # Age-related variations
            age_factor = random.uniform(0.0, 1.0)  # 0 = new, 1 = old
            felt_density *= (1.0 - age_factor * 0.2)  # Felt compresses with age
            
            profiles[note] = {
                'felt_density': felt_density,
                'strike_area': strike_area,
                'hardness_gradient': hardness_gradient,
                'age_factor': age_factor,
                'groove_depth': age_factor * 0.5,  # String grooves in felt
                'asymmetry': random.uniform(0.0, 0.1)  # Slight felt asymmetry
            }
            
        return profiles
    
    def calculate_hammer_strike(self, note: int, velocity: int, 
                              key_motion_data: Dict, timestamp: float) -> Dict:
        """Calculate realistic hammer strike characteristics"""
        
        if note not in self.hammer_states:
            profile = self.voicing_profiles[note]
            self.hammer_states[note] = HammerState(
                position=0.0,
                velocity=0.0,
                felt_density=profile['felt_density'],
                wear_level=profile['age_factor'],
                temperature=self.temperature
            )
        
        hammer = self.hammer_states[note]
        profile = self.voicing_profiles[note]
        
        # Calculate hammer velocity from key motion
        hammer_velocity = velocity * (1.0 + key_motion_data['velocity'] * 0.3)
        
        # Apply escapement effects
        if key_motion_data['escapement_triggered']:
            # Hammer continues with momentum after escapement
            momentum_factor = 0.85 + random.uniform(-0.05, 0.05)
            hammer_velocity *= momentum_factor
        
        # Calculate strike characteristics
        strike_data = self._calculate_strike_physics(
            hammer, profile, hammer_velocity, key_motion_data
        )
        
        # Update wear patterns
        self._update_hammer_wear(note, hammer_velocity, timestamp)
        
        return strike_data
    
    def _calculate_strike_physics(self, hammer: HammerState, profile: Dict, 
                                 velocity: float, key_motion: Dict) -> Dict:
        """Calculate detailed strike physics"""
        
        # Base impact characteristics
        impact_duration = 0.8 + (1.0 - hammer.felt_density) * 1.2  # Milliseconds
        contact_area = profile['strike_area'] * (1.0 + velocity / 200.0)
        
        # Felt compression modeling
        compression_ratio = min(0.7, velocity / 150.0)  # How much felt compresses
        effective_hardness = hammer.felt_density * (1.0 + compression_ratio)
        
        # Multiple contact points (felt layers)
        strike_phases = []
        for i, hardness in enumerate(profile['hardness_gradient']):
            phase_delay = i * 0.1  # Milliseconds between layers
            phase_intensity = hardness * (1.0 - i * 0.2)
            strike_phases.append({
                'delay': phase_delay,
                'intensity': phase_intensity,
                'duration': impact_duration * (1.0 - i * 0.3)
            })
        
        # Harmonicity effects from felt condition
        inharmonicity_factor = 1.0 + hammer.wear_level * 0.02
        
        # Temperature effects on felt
        temp_factor = 1.0 + (hammer.temperature - 20.0) * 0.001
        effective_hardness *= temp_factor
        
        return {
            'impact_duration': impact_duration,
            'effective_hardness': effective_hardness,
            'contact_area': contact_area,
            'strike_phases': strike_phases,
            'inharmonicity_factor': inharmonicity_factor,
            'felt_compression': compression_ratio,
            'asymmetry_factor': profile['asymmetry'],
            'groove_effect': profile['groove_depth']
        }
    
    def _update_hammer_wear(self, note: int, velocity: float, timestamp: float):
        """Update hammer wear patterns over time"""
        wear_increment = velocity / 10000.0  # Very gradual wear
        self.hammer_wear_patterns[note] += wear_increment
        
        if note in self.voicing_profiles:
            # Wear affects felt density
            wear_effect = min(0.3, self.hammer_wear_patterns[note])
            self.voicing_profiles[note]['felt_density'] *= (1.0 - wear_effect * 0.5)
            self.voicing_profiles[note]['groove_depth'] += wear_increment * 0.1


class ComprehensiveStringResonanceEngine:
    """Enhanced string resonance with physical string modeling"""
    
    def __init__(self):
        self.string_states = {}  # note -> StringState
        self.string_coupling_map = self._build_advanced_coupling_map()
        self.soundboard_model = SoundboardResonanceModel()
        self.case_resonance = CaseResonanceModel()
        self.temperature = 20.0
        self.humidity = 50.0
        
        # Physical constants
        self.string_tensions = self._calculate_string_tensions()
        self.inharmonicity_coefficients = self._calculate_inharmonicity()
        
    def _build_advanced_coupling_map(self) -> Dict[int, List[Tuple[int, float, str]]]:
        """Build comprehensive string coupling with coupling types"""
        coupling = {}
        
        for fundamental in range(21, 109):
            coupled_notes = []
            
            # 1. Harmonic series (most important)
            for harmonic in range(2, 16):  # Extended harmonics
                harmonic_freq = 440.0 * (2 ** ((fundamental - 69) / 12.0)) * harmonic
                harmonic_note = 69 + 12 * math.log2(harmonic_freq / 440.0)
                
                if 21 <= harmonic_note <= 108:
                    strength = 0.4 / math.sqrt(harmonic)
                    coupled_notes.append((int(harmonic_note), strength, "harmonic"))
            
            # 2. Sub-harmonics
            for sub in range(2, 8):
                sub_freq = 440.0 * (2 ** ((fundamental - 69) / 12.0)) / sub
                sub_note = 69 + 12 * math.log2(sub_freq / 440.0)
                
                if 21 <= sub_note <= 108:
                    strength = 0.2 / sub
                    coupled_notes.append((int(sub_note), strength, "sub_harmonic"))
            
            # 3. Interval resonances (perfect consonances)
            intervals = {
                'octave': [12, -12],
                'perfect_fifth': [7, -7, 19, -19],
                'perfect_fourth': [5, -5, 17, -17],
                'major_third': [4, -4, 16, -16]
            }
            
            for interval_type, semitones in intervals.items():
                for semitone in semitones:
                    interval_note = fundamental + semitone
                    if 21 <= interval_note <= 108:
                        if interval_type == 'octave':
                            strength = 0.3
                        elif interval_type == 'perfect_fifth':
                            strength = 0.15
                        else:
                            strength = 0.08
                        coupled_notes.append((interval_note, strength, interval_type))
            
            # 4. Adjacent string coupling (physical proximity)
            for adj in [-2, -1, 1, 2]:
                adj_note = fundamental + adj
                if 21 <= adj_note <= 108:
                    strength = 0.1 / abs(adj)
                    coupled_notes.append((adj_note, strength, "adjacent"))
            
            # 5. Soundboard coupling regions
            soundboard_region = self._get_soundboard_region(fundamental)
            for other_note in range(21, 109):
                if other_note != fundamental:
                    other_region = self._get_soundboard_region(other_note)
                    if soundboard_region == other_region:
                        distance = abs(other_note - fundamental)
                        if distance <= 12:  # Within an octave
                            strength = 0.05 / (1 + distance * 0.1)
                            coupled_notes.append((other_note, strength, "soundboard"))
            
            coupling[fundamental] = coupled_notes
        
        return coupling
    
    def _get_soundboard_region(self, note: int) -> str:
        """Determine which soundboard region a note primarily excites"""
        if note <= 32:
            return "bass_bridge"
        elif note <= 52:
            return "tenor_bridge"
        elif note <= 72:
            return "mid_soundboard"
        else:
            return "treble_soundboard"
    
    def _calculate_string_tensions(self) -> Dict[int, float]:
        """Calculate realistic string tensions"""
        tensions = {}
        
        for note in range(21, 109):
            # Base tension calculation (Newtons)
            freq = 440.0 * (2 ** ((note - 69) / 12.0))
            
            if note <= 32:  # Bass - thick wound strings
                string_length = 1.8 - (note - 21) * 0.02  # meters
                linear_density = 0.05  # kg/m
            elif note <= 60:  # Mid - wound strings
                string_length = 1.2 - (note - 32) * 0.01
                linear_density = 0.003
            else:  # Treble - plain wire
                string_length = 0.6 - (note - 60) * 0.005
                linear_density = 0.0005
            
            tension = (2 * string_length * freq) ** 2 * linear_density
            tensions[note] = tension
        
        return tensions
    
    def _calculate_inharmonicity(self) -> Dict[int, float]:
        """Calculate inharmonicity coefficients for each string"""
        coefficients = {}
        
        for note in range(21, 109):
            if note <= 32:  # Bass - high inharmonicity
                coeff = 0.0005 + (32 - note) * 0.00002
            elif note <= 60:  # Mid - moderate
                coeff = 0.0002 + (60 - note) * 0.000005
            else:  # Treble - low inharmonicity
                coeff = 0.0001 - (note - 60) * 0.000001
            
            coefficients[note] = max(0.00001, coeff)
        
        return coefficients
    
    def trigger_advanced_resonance(self, note: int, velocity: int, 
                                  hammer_data: Dict, timestamp: float,
                                  pedal_states: Dict) -> List[Tuple[int, int, float, Dict]]:
        """Trigger comprehensive resonance modeling"""
        
        # Initialise string state if needed
        if note not in self.string_states:
            self.string_states[note] = StringState(
                tension=self.string_tensions[note],
                temperature=self.temperature,
                age=random.uniform(0.0, 1.0),
                coupling_strength=1.0,
                last_strike_time=timestamp,
                decay_coefficient=self._calculate_decay_coefficient(note)
            )
        
        string_state = self.string_states[note]
        
        # Update string tension based on temperature
        temp_factor = 1.0 + (self.temperature - 20.0) * 0.0002
        current_tension = string_state.tension * temp_factor
        
        # Calculate pitch deviation from tension change
        tension_ratio = current_tension / self.string_tensions[note]
        pitch_deviation = (tension_ratio - 1.0) * 600  # cents
        
        # Generate resonant notes
        resonant_responses = []
        
        if note in self.string_coupling_map:
            for coupled_note, base_strength, coupling_type in self.string_coupling_map[note]:
                
                # Calculate coupling strength with multiple factors
                coupling_strength = self._calculate_advanced_coupling_strength(
                    note, coupled_note, velocity, base_strength, 
                    coupling_type, hammer_data, pedal_states, timestamp
                )
                
                if coupling_strength > 0.01:
                    resonant_velocity = int(coupling_strength * velocity * 0.7)
                    
                    if resonant_velocity >= 5:
                        # Calculate propagation delay
                        delay = self._calculate_propagation_delay(
                            note, coupled_note, coupling_type
                        )
                        
                        # Generate resonance characteristics
                        resonance_data = {
                            'coupling_type': coupling_type,
                            'source_note': note,
                            'pitch_deviation': pitch_deviation * (coupling_strength * 0.5),
                            'inharmonicity': self.inharmonicity_coefficients[coupled_note],
                            'decay_rate': self._calculate_resonant_decay(
                                coupled_note, coupling_type, pedal_states
                            )
                        }
                        
                        resonant_responses.append((
                            coupled_note, resonant_velocity, delay, resonance_data
                        ))
        
        # Limit number of resonant notes to prevent thread explosion
        MAX_RESONANT_NOTES = 10
        if len(resonant_responses) > MAX_RESONANT_NOTES:
            resonant_responses = resonant_responses[:MAX_RESONANT_NOTES]
        
        # Update soundboard and case resonance
        self.soundboard_model.add_excitation(note, velocity, timestamp)
        self.case_resonance.add_excitation(note, velocity, timestamp)
        
        return resonant_responses
    
    def _calculate_advanced_coupling_strength(self, fundamental: int, resonant: int,
                                            velocity: int, base_strength: float,
                                            coupling_type: str, hammer_data: Dict,
                                            pedal_states: Dict, timestamp: float) -> float:
        """Calculate coupling strength with comprehensive factors"""
        
        strength = base_strength
        
        # Velocity dependence
        velocity_factor = min(1.2, velocity / 80.0)
        strength *= velocity_factor
        
        # Hammer characteristics influence
        if coupling_type == "harmonic":
            # Harder felt enhances harmonics
            hardness_factor = 0.8 + hammer_data.get('effective_hardness', 0.7) * 0.4
            strength *= hardness_factor
        
        # Pedal influence
        if pedal_states.get('sustain', False):
            strength *= 1.4  # Increased coupling with sustain
        
        if pedal_states.get('sostenuto', False) and resonant in pedal_states.get('sostenuto_notes', set()):
            strength *= 1.2
        
        # String age effects
        if fundamental in self.string_states:
            age_factor = self.string_states[fundamental].age
            if coupling_type == "harmonic":
                strength *= (1.0 + age_factor * 0.3)  # Older strings = more harmonics
            else:
                strength *= (1.0 - age_factor * 0.1)  # Less coupling overall
        
        # Environmental factors
        humidity_factor = 1.0 + (self.humidity - 50.0) * 0.002
        strength *= humidity_factor
        
        # Soundboard coupling enhancement
        if coupling_type == "soundboard":
            soundboard_factor = self.soundboard_model.get_coupling_factor(
                fundamental, resonant
            )
            strength *= soundboard_factor
        
        return strength
    
    def _calculate_propagation_delay(self, source: int, target: int, 
                                   coupling_type: str) -> float:
        """Calculate realistic wave propagation delays"""
        
        if coupling_type == "harmonic":
            # Harmonic coupling is nearly instantaneous
            return random.uniform(0.0001, 0.0003)
        
        elif coupling_type == "soundboard":
            # Soundboard transmission
            distance = abs(target - source) * 0.03  # Approximate distance in meters
            speed = 1200  # m/s in wood
            return distance / speed + random.uniform(0.0002, 0.0008)
        
        elif coupling_type == "adjacent":
            # String-to-string coupling
            return random.uniform(0.0005, 0.002)
        
        else:
            # Other coupling types
            return random.uniform(0.0003, 0.001)
    
    def _calculate_resonant_decay(self, note: int, coupling_type: str,
                                pedal_states: Dict) -> float:
        """Calculate decay rate for resonant notes"""
        
        base_decay = 0.3 + (note - 21) * 0.005  # Higher notes decay faster
        
        if coupling_type == "harmonic":
            decay = base_decay * 0.8  # Harmonics sustain well
        elif coupling_type == "soundboard":
            decay = base_decay * 1.2  # Soundboard coupling decays faster
        else:
            decay = base_decay
        
        # Pedal effects
        if pedal_states.get('sustain', False):
            decay *= 0.3  # Much slower decay with sustain
        
        return decay
    
    def _calculate_decay_coefficient(self, note: int) -> float:
        """Calculate string decay coefficient"""
        if note <= 32:
            return 0.15  # Bass strings sustain longer
        elif note <= 60:
            return 0.25
        else:
            return 0.4   # Treble strings decay faster


class SoundboardResonanceModel:
    """Advanced soundboard resonance modeling"""
    
    def __init__(self):
        self.resonance_modes = self._calculate_soundboard_modes()
        self.current_excitations = {}  # frequency -> amplitude
        self.mode_coupling = self._build_mode_coupling()

        self.temperature = 20.0  # Celsius
        self.humidity = 50.0     # Percent
        self.air_pressure = 1013.25  # hPa
        
        self.temperature_drift_rate = 0.001  # Per degree per minute
        self.humidity_drift_rate = 0.0005    # Per percent per minute
        
        self.last_update = time.time()
        self.cumulative_drift = defaultdict(float) 
        
    def update_environment(self, temp_change: float = None, 
                          humidity_change: float = None, 
                          pressure_change: float = None):
        """Update environmental conditions"""
        current_time = time.time()
        dt = (current_time - self.last_update) / 60.0  # Minutes
        
        if temp_change is not None:
            self.temperature += temp_change
        else:
            # Simulate gradual temperature drift
            self.temperature += random.gauss(0, 0.1) * dt
        
        if humidity_change is not None:
            self.humidity += humidity_change
        else:
            # Simulate gradual humidity drift
            self.humidity += random.gauss(0, 0.5) * dt
        
        self.humidity = max(10, min(90, self.humidity))
        
        # Update pitch drift for all notes
        self._update_pitch_drift(dt)
        
        self.last_update = current_time
        
    def _calculate_soundboard_modes(self) -> List[Dict]:
        """Calculate soundboard resonant modes"""
        modes = []
        
        # Primary bending modes
        for i in range(1, 12):  # 11 primary modes
            frequency = 60 + i * 85 + random.uniform(-10, 10)  # Hz
            damping = 0.02 + i * 0.003
            amplitude = 1.0 / (i * 0.8)
            
            modes.append({
                'frequency': frequency,
                'damping': damping,
                'amplitude': amplitude,
                'phase': random.uniform(0, 2 * math.pi),
                'q_factor': 15 + i * 3
            })
        
        return modes
    
    def _build_mode_coupling(self) -> Dict[int, List[Tuple[int, float]]]:
        """Build coupling between soundboard modes and piano notes"""
        coupling = {}
        
        for note in range(21, 109):
            note_freq = 440.0 * (2 ** ((note - 69) / 12.0))
            coupled_modes = []
            
            for mode_idx, mode in enumerate(self.resonance_modes):
                # Coupling strength based on frequency proximity
                freq_diff = abs(note_freq - mode['frequency'])
                coupling_strength = 1.0 / (1.0 + freq_diff / 50.0)
                
                if coupling_strength > 0.1:
                    coupled_modes.append((mode_idx, coupling_strength))
            
            coupling[note] = coupled_modes
        
        return coupling
    
    def add_excitation(self, note: int, velocity: int, timestamp: float):
        """Add excitation to soundboard"""
        note_freq = 440.0 * (2 ** ((note - 69) / 12.0))
        excitation_amplitude = velocity / 127.0
        
        self.current_excitations[note_freq] = {
            'amplitude': excitation_amplitude,
            'timestamp': timestamp,
            'decay_rate': 0.3
        }
    
    def get_coupling_factor(self, note1: int, note2: int) -> float:
        """Get coupling factor between two notes via soundboard"""
        if note1 not in self.mode_coupling or note2 not in self.mode_coupling:
            return 1.0
        
        # Find common modes
        modes1 = {mode_idx for mode_idx, _ in self.mode_coupling[note1]}
        modes2 = {mode_idx for mode_idx, _ in self.mode_coupling[note2]}
        common_modes = modes1 & modes2
        
        if not common_modes:
            return 0.8
        
        # Calculate coupling through common modes
        coupling_sum = 0.0
        for mode_idx in common_modes:
            strength1 = next(s for m, s in self.mode_coupling[note1] if m == mode_idx)
            strength2 = next(s for m, s in self.mode_coupling[note2] if m == mode_idx)
            coupling_sum += strength1 * strength2
        
        return min(2.0, 1.0 + coupling_sum)

    def _update_pitch_drift(self, dt: float):
        """Update cumulative pitch drift"""
        for note in range(21, 109):
            # Temperature effect (strings expand/contract)
            temp_effect = (self.temperature - 20.0) * self.temperature_drift_rate * dt
            
            # Humidity effect (soundboard swells/shrinks)
            humidity_effect = (self.humidity - 50.0) * self.humidity_drift_rate * dt
            
            # Total drift
            total_drift = temp_effect + humidity_effect
            
            # Different strings drift differently
            if note <= 32:  # Bass strings more stable
                total_drift *= 0.7
            elif note >= 80:  # Treble strings less stable
                total_drift *= 1.3
            
            self.cumulative_drift[note] += total_drift
            
            # Limit drift
            self.cumulative_drift[note] = max(-50, min(50, self.cumulative_drift[note]))
    
    def get_pitch_deviation(self, note: int) -> float:
        """Get current pitch deviation for a note in cents"""
        return self.cumulative_drift.get(note, 0.0)
    
    def get_timbre_modification(self, note: int) -> Dict:
        """Get timbre modifications due to environment"""
        temp_factor = (self.temperature - 20.0) / 30.0  # Normalized
        humidity_factor = (self.humidity - 50.0) / 40.0  # Normalized
        
        return {
            'brightness_change': temp_factor * 0.1,  # Warmer = brighter
            'sustain_change': -humidity_factor * 0.15,  # Higher humidity = less sustain
            'attack_sharpness': temp_factor * 0.05,
            'harmonic_content': temp_factor * 0.08
        }

class CaseResonanceModel:
    """Piano case resonance modeling"""
    
    def __init__(self):
        self.case_modes = self._calculate_case_modes()
        self.rim_coupling = {}  # Note coupling to rim resonances
        
    def _calculate_case_modes(self) -> List[Dict]:
        """Calculate case resonant frequencies"""
        modes = []
        
        # Typical grand piano case resonances
        case_frequencies = [85, 120, 180, 220, 280, 350, 420]
        
        for freq in case_frequencies:
            modes.append({
                'frequency': freq + random.uniform(-5, 5),
                'damping': 0.05,
                'amplitude': 0.3,
                'q_factor': 8
            })
        
        return modes
    
    def add_excitation(self, note: int, velocity: int, timestamp: float):
        """Add excitation to case"""
        # Case resonance primarily affects bass notes
        if note <= 40:
            case_factor = 1.0 - (note - 21) / 19.0  # Stronger for lower notes
            self.rim_coupling[note] = case_factor * velocity / 127.0


class UltraAdvancedPedalSystem:
    """Enhanced pedal system with mechanical modeling"""
    
    def __init__(self, midi_sender_callback):
        self.midi_sender = midi_sender_callback
        
        # Continuous pedal positions and physics
        self.pedal_positions = {'sustain': 0.0, 'sostenuto': 0.0, 'soft': 0.0}
        self.pedal_velocities = {'sustain': 0.0, 'sostenuto': 0.0, 'soft': 0.0}
        self.pedal_springs = {'sustain': 0.8, 'sostenuto': 0.6, 'soft': 0.7}  # Spring constants
        
        # Mechanical characteristics
        self.friction_coefficients = {'sustain': 0.12, 'sostenuto': 0.08, 'soft': 0.10}
        self.catch_positions = {'sustain': 0.15, 'sostenuto': 0.12, 'soft': 0.18}
        self.engagement_curves = self._build_engagement_curves()
        
        # Advanced pedal states
        self.damper_lift_curve = self._build_damper_lift_curve()
        self.una_corda_shift_curve = self._build_una_corda_curve()
        self.sostenuto_captured_dampers = set()
        
        # Mechanical noise generation
        self.pedal_noise_samples = self._generate_pedal_noise()
        
        # Pedal regulation (slight variations)
        self.regulation_variations = {
            pedal: random.uniform(0.95, 1.05) for pedal in ['sustain', 'sostenuto', 'soft']
        }
        
        self.logger = logging.getLogger("AdvancedPedals")
    
    def _build_engagement_curves(self) -> Dict[str, Callable]:
        """Build non-linear engagement curves for each pedal"""
        
        def sustain_curve(position: float) -> float:
            """Sustain pedal - gradual engagement with sharp cutoff"""
            if position < 0.2:
                return 0.0
            elif position < 0.4:
                return (position - 0.2) * 2.5  # Gradual start
            else:
                return 1.0  # Full engagement
        
        def sostenuto_curve(position: float) -> float:
            """Sostenuto - more linear engagement"""
            return min(1.0, max(0.0, (position - 0.1) / 0.8))
        
        def soft_curve(position: float) -> float:
            """Una corda - progressive shift"""
            if position < 0.1:
                return 0.0
            else:
                return min(1.0, (position - 0.1) ** 0.7)  # Exponential curve
        
        return {
            'sustain': sustain_curve,
            'sostenuto': sostenuto_curve,
            'soft': soft_curve
        }
    
    def _build_damper_lift_curve(self) -> np.ndarray:
        """Build realistic damper lift characteristics"""
        # Different dampers lift at different pedal positions
        positions = np.linspace(0, 1, 88)  # For each note
        lift_curve = np.zeros(88)
        
        for i in range(88):
            note = i + 21
            if note <= 32:  # Bass - lift early
                lift_position = 0.15 + i * 0.002
            elif note <= 60:  # Mid - standard lift
                lift_position = 0.25 + (i - 11) * 0.001
            else:  # Treble - lift later
                lift_position = 0.35 + (i - 39) * 0.0005
            
            lift_curve[i] = min(0.6, lift_position)
        
        return lift_curve
    
    def _build_una_corda_curve(self) -> Dict[int, float]:
        """Build una corda hammer shift characteristics"""
        shift_amounts = {}
        
        for note in range(21, 109):
            if note <= 40:  # Bass - no shift (only one string)
                shift_amounts[note] = 0.0
            elif note <= 65:  # Two strings - partial shift
                shift_amounts[note] = 0.6
            else:  # Three strings - full shift
                shift_amounts[note] = 1.0
        
        return shift_amounts
    
    def _generate_pedal_noise(self) -> Dict[str, np.ndarray]:
        """Generate realistic pedal mechanism noises"""
        sample_rate = 44100
        
        # Spring creak
        t_creak = np.linspace(0, 0.2, int(sample_rate * 0.2))
        spring_creak = np.random.normal(0, 0.02, len(t_creak)) * np.sin(2 * np.pi * 8 * t_creak)
        
        # Felt brush (dampers on strings)
        t_brush = np.linspace(0, 0.1, int(sample_rate * 0.1))
        felt_brush = np.random.normal(0, 0.015, len(t_brush)) * np.exp(-t_brush * 20)
        
        # Mechanism click
        t_click = np.linspace(0, 0.05, int(sample_rate * 0.05))
        mech_click = np.random.normal(0, 0.01, len(t_click)) * np.exp(-t_click * 40)
        
        return {
            'spring_creak': spring_creak,
            'felt_brush': felt_brush,
            'mechanism_click': mech_click
        }
    
    def handle_advanced_pedal_event(self, pedal_name: str, action: str,
                                   current_held_notes: Set[int], timestamp: float) -> List[Dict]:
        """Handle pedal with comprehensive physical modeling"""
        
        # Calculate pedal physics
        physics_data = self._calculate_pedal_physics(pedal_name, action, timestamp)
        
        # Generate mechanical effects
        mechanical_effects = []
        
        if pedal_name == 'sustain':
            effects = self._handle_sustain_advanced(action, current_held_notes, physics_data)
        elif pedal_name == 'sostenuto':
            effects = self._handle_sostenuto_advanced(action, current_held_notes, physics_data)
        elif pedal_name == 'soft':
            effects = self._handle_una_corda_advanced(action, physics_data)
        else:
            effects = []
        
        mechanical_effects.extend(effects)
        
        return mechanical_effects
    
    def _calculate_pedal_physics(self, pedal_name: str, action: str, timestamp: float) -> Dict:
        """Calculate pedal physics with spring/damper model"""
        
        current_pos = self.pedal_positions[pedal_name]
        spring_constant = self.pedal_springs[pedal_name]
        friction = self.friction_coefficients[pedal_name]
        regulation = self.regulation_variations[pedal_name]
        
        # Target position
        target_pos = 1.0 * regulation if action == "press" else 0.0
        
        # Spring force calculation
        spring_force = spring_constant * (target_pos - current_pos)
        
        # Friction effects
        if abs(self.pedal_velocities[pedal_name]) > 0.01:
            friction_force = -friction * np.sign(self.pedal_velocities[pedal_name])
        else:
            friction_force = 0.0
        
        # Update velocity and position
        dt = 0.01  # Assumed time step
        acceleration = spring_force + friction_force
        self.pedal_velocities[pedal_name] += acceleration * dt
        
        # Damping
        self.pedal_velocities[pedal_name] *= 0.85
        
        # Position update
        new_position = current_pos + self.pedal_velocities[pedal_name] * dt
        self.pedal_positions[pedal_name] = max(0.0, min(1.0, new_position))
        
        return {
            'position': self.pedal_positions[pedal_name],
            'velocity': abs(self.pedal_velocities[pedal_name]),
            'spring_force': spring_force,
            'engagement_level': self.engagement_curves[pedal_name](self.pedal_positions[pedal_name])
        }
    
    def _handle_sustain_advanced(self, action: str, current_notes: Set[int],
                               physics_data: Dict) -> List[Dict]:
        """Advanced sustain pedal with individual damper modeling"""
        effects = []
        position = physics_data['position']
        engagement = physics_data['engagement_level']
        
        # Individual damper control
        for note in range(21, 109):
            note_idx = note - 21
            damper_lift_threshold = self.damper_lift_curve[note_idx]
            
            if position >= damper_lift_threshold:
                # Damper lifted
                damper_lift_amount = min(1.0, (position - damper_lift_threshold) / 0.3)
                
                # Send continuous damper control
                cc_value = int(damper_lift_amount * 127)
                effects.append({
                    'type': 'damper_lift',
                    'note': note,
                    'amount': damper_lift_amount,
                    'cc_value': cc_value
                })
                
                # Mechanical noise for damper lift
                if physics_data['velocity'] > 0.2:
                    effects.append({
                        'type': 'mechanical_noise',
                        'sound': 'felt_brush',
                        'intensity': physics_data['velocity'] * 0.1
                    })
        
        # Global sustain CC
        self.midi_sender(64, int(engagement * 127))
        
        return effects
    
    def _handle_sostenuto_advanced(self, action: str, current_notes: Set[int],
                                 physics_data: Dict) -> List[Dict]:
        """Advanced sostenuto with selective damper capture"""
        effects = []
        
        if action == "press" and physics_data['engagement_level'] > 0.3:
            # Capture currently held dampers
            self.sostenuto_captured_dampers = current_notes.copy()
            
            for note in self.sostenuto_captured_dampers:
                effects.append({
                    'type': 'sostenuto_capture',
                    'note': note,
                    'capture_level': physics_data['engagement_level']
                })
        
        elif action == "release":
            # Release captured dampers gradually
            for note in self.sostenuto_captured_dampers:
                effects.append({
                    'type': 'sostenuto_release',
                    'note': note,
                    'release_rate': 1.0 - physics_data['position']
                })
            
            if physics_data['position'] < 0.1:
                self.sostenuto_captured_dampers.clear()
        
        # Sostenuto CC
        self.midi_sender(66, int(physics_data['engagement_level'] * 127))
        
        return effects
    
    def _handle_una_corda_advanced(self, action: str, physics_data: Dict) -> List[Dict]:
        """Advanced una corda with progressive hammer shift"""
        effects = []
        shift_amount = physics_data['engagement_level']
        
        # Calculate tone modification for each register
        for note in range(21, 109):
            if note in self.una_corda_shift_curve:
                max_shift = self.una_corda_shift_curve[note]
                current_shift = shift_amount * max_shift
                
                # Tone color change
                brightness_reduction = current_shift * 0.3  # Darker tone
                volume_reduction = current_shift * 0.15     # Softer
                
                effects.append({
                    'type': 'una_corda_effect',
                    'note': note,
                    'shift_amount': current_shift,
                    'brightness_factor': 1.0 - brightness_reduction,
                    'volume_factor': 1.0 - volume_reduction
                })
        
        # Una corda CC
        self.midi_sender(67, int(shift_amount * 127))
        
        return effects
    
    def get_damper_state(self, note: int) -> Dict:
        """Get current damper state for a note"""
        sustain_pos = self.pedal_positions['sustain']
        note_idx = note - 21
        
        if note_idx < len(self.damper_lift_curve):
            lift_threshold = self.damper_lift_curve[note_idx]
            is_lifted = sustain_pos >= lift_threshold
            lift_amount = max(0.0, min(1.0, (sustain_pos - lift_threshold) / 0.3))
        else:
            is_lifted = False
            lift_amount = 0.0
        
        # Check sostenuto
        sostenuto_held = (note in self.sostenuto_captured_dampers and 
                         self.pedal_positions['sostenuto'] > 0.3)
        
        return {
            'lifted': is_lifted or sostenuto_held,
            'lift_amount': lift_amount,
            'sostenuto_captured': sostenuto_held
        }


class ExtendedTechniquesEngine:
    """Engine for extended piano techniques and preparations"""
    
    def __init__(self):
        self.prepared_notes = {}  # note -> preparation_type
        self.harmonic_nodes = self._calculate_harmonic_nodes()
        self.silent_key_detections = {}  # For catching harmonics
        self.string_muting = {}  # Manual string muting
        
    def _calculate_harmonic_nodes(self) -> Dict[int, List[float]]:
        """Calculate harmonic node positions for each string"""
        nodes = {}
        
        for note in range(21, 109):
            note_nodes = []
            
            # Calculate node positions for harmonics 2-8
            for harmonic in range(2, 9):
                for node in range(1, harmonic):
                    position = node / harmonic  # Position along string length
                    note_nodes.append(position)
            
            nodes[note] = sorted(note_nodes)
        
        return nodes
    
    def detect_silent_key_press(self, note: int, key_position: float,
                               velocity: float, timestamp: float) -> Optional[Dict]:
        """Detect silent key presses for harmonic generation"""
        
        # Silent press detection (key pressed slowly without hammer strike)
        if velocity < 15 and key_position > 0.7:
            self.silent_key_detections[note] = {
                'timestamp': timestamp,
                'position': key_position,
                'harmonic_ready': True
            }
            
            # Check for harmonic excitation from other notes
            return self._check_harmonic_excitation(note, timestamp)
        
        return None
    
    def _check_harmonic_excitation(self, silent_note: int, timestamp: float) -> Optional[Dict]:
        """Check if any playing notes can excite harmonics on silent key"""
        
        harmonic_responses = []
        
        # Look for fundamental frequencies that match harmonics of the silent note
        silent_freq = 440.0 * (2 ** ((silent_note - 69) / 12.0))
        
        # Check recently struck notes
        for active_note, strike_data in getattr(self, 'recent_strikes', {}).items():
            if timestamp - strike_data['timestamp'] < 2.0:  # Within 2 seconds
                
                active_freq = 440.0 * (2 ** ((active_note - 69) / 12.0))
                
                # Check if active note frequency matches harmonic of silent note
                for harmonic in range(2, 9):
                    harmonic_freq = silent_freq * harmonic
                    
                    if abs(active_freq - harmonic_freq) / harmonic_freq < 0.02:  # Within 2%
                        # Harmonic match found
                        harmonic_strength = strike_data['velocity'] / 127.0 / harmonic
                        
                        if harmonic_strength > 0.1:
                            harmonic_responses.append({
                                'type': 'natural_harmonic',
                                'fundamental_note': active_note,
                                'harmonic_note': silent_note,
                                'harmonic_number': harmonic,
                                'strength': harmonic_strength
                            })
        
        if harmonic_responses:
            return {
                'type': 'harmonic_excitation',
                'responses': harmonic_responses
            }
        
        return None
    
    def add_preparation(self, note: int, preparation_type: str, **params) -> Dict:
        """Add preparation to a string"""
        
        preparation_data = {
            'type': preparation_type,
            'timestamp': time.time(),
            'parameters': params
        }
        
        if preparation_type == 'mute':
            # Rubber mute between strings
            preparation_data['mute_position'] = params.get('position', 0.5)
            preparation_data['mute_material'] = params.get('material', 'rubber')
            
        elif preparation_type == 'bolt':
            # Metal bolt on string
            preparation_data['bolt_position'] = params.get('position', 0.3)
            preparation_data['bolt_size'] = params.get('size', 'medium')
            
        elif preparation_type == 'paper':
            # Paper through strings
            preparation_data['paper_thickness'] = params.get('thickness', 'thin')
            preparation_data['weave_pattern'] = params.get('pattern', 'straight')
            
        elif preparation_type == 'coin':
            # Coins on strings
            preparation_data['coin_type'] = params.get('coin_type', 'penny')
            preparation_data['coin_position'] = params.get('position', 0.4)
        
        self.prepared_notes[note] = preparation_data
        
        return preparation_data
    
    def calculate_preparation_effect(self, note: int, velocity: int,
                                   hammer_data: Dict) -> Dict:
        """Calculate how preparation affects the sound"""
        
        if note not in self.prepared_notes:
            return {'modified': False}
        
        prep = self.prepared_notes[note]
        prep_type = prep['type']
        
        effect = {
            'modified': True,
            'preparation_type': prep_type,
            'velocity_modification': 1.0,
            'pitch_modification': 0.0,  # cents
            'timbre_changes': {},
            'additional_sounds': []
        }
        
        if prep_type == 'mute':
            # Rubber mute - shortens sustain, muffles tone
            effect['velocity_modification'] = 0.7
            effect['timbre_changes'] = {
                'sustain_reduction': 0.6,
                'brightness_reduction': 0.4,
                'attack_modification': 'softer'
            }
            
        elif prep_type == 'bolt':
            # Metal bolt - creates buzzing, pitch bending
            bolt_size_factor = {'small': 0.8, 'medium': 1.0, 'large': 1.2}[prep['bolt_size']]
            
            effect['pitch_modification'] = random.uniform(-20, 20) * bolt_size_factor
            effect['timbre_changes'] = {
                'buzz_intensity': 0.3 * bolt_size_factor,
                'inharmonicity_increase': 0.4,
                'attack_modification': 'metallic'
            }
            effect['additional_sounds'].append({
                'type': 'metallic_buzz',
                'intensity': velocity / 127.0 * 0.4
            })
            
        elif prep_type == 'paper':
            # Paper - creates rattling, filtering
            thickness_factor = {'thin': 0.7, 'medium': 1.0, 'thick': 1.3}[prep['paper_thickness']]
            
            effect['velocity_modification'] = 0.8
            effect['timbre_changes'] = {
                'rattle_intensity': 0.2 * thickness_factor,
                'frequency_filtering': 'high_cut',
                'attack_modification': 'papery'
            }
            effect['additional_sounds'].append({
                'type': 'paper_rattle',
                'intensity': velocity / 127.0 * 0.3
            })
            
        elif prep_type == 'coin':
            # Coins - metallic ringing, pitch shifts
            effect['velocity_modification'] = 1.2  # Can amplify
            effect['timbre_changes'] = {
                'metallic_ring': 0.5,
                'pitch_instability': 0.3,
                'attack_modification': 'bright_metallic'
            }
            effect['additional_sounds'].append({
                'type': 'coin_ring',
                'pitch': note + random.uniform(-2, 2),  # Slightly detuned
                'intensity': velocity / 127.0 * 0.6
            })
        
        return effect
    
    def handle_resonance_capture(self, pedal_pressed: bool, active_resonances: Dict,
                                timestamp: float) -> Dict:
        """Handle resonance capture when pedal is pressed"""
        
        if pedal_pressed:
            # Capture current resonance state
            captured_resonances = {}
            
            for note, resonance_data in active_resonances.items():
                if resonance_data.get('amplitude', 0) > 0.1:
                    # Capture this resonance for extension
                    captured_resonances[note] = {
                        'captured_amplitude': resonance_data['amplitude'],
                        'capture_timestamp': timestamp,
                        'decay_rate': resonance_data.get('decay_rate', 0.3) * 0.5,  # Slower decay
                        'extended': True
                    }
            
            return {
                'type': 'resonance_capture',
                'captured_notes': captured_resonances,
                'capture_timestamp': timestamp
            }
        
        return {'type': 'no_capture'}
    
    def generate_prepared_sound_events(self, note: int, velocity: int,
                                     preparation_effect: Dict, timestamp: float) -> List[Dict]:
        """Generate additional sound events from preparations"""
        
        events = []
        
        for additional_sound in preparation_effect.get('additional_sounds', []):
            sound_type = additional_sound['type']
            intensity = additional_sound['intensity']
            
            if sound_type == 'metallic_buzz':
                # Generate buzz frequency
                buzz_freq = 440.0 * (2 ** ((note - 69) / 12.0)) * 1.5  # Slightly higher
                
                events.append({
                    'type': 'prepared_sound',
                    'sound_type': 'buzz',
                    'frequency': buzz_freq,
                    'intensity': intensity,
                    'duration': 0.3 + intensity * 0.5,
                    'delay': 0.01
                })
                
            elif sound_type == 'paper_rattle':
                # Generate rattle events
                for i in range(int(intensity * 10)):
                    events.append({
                        'type': 'prepared_sound',
                        'sound_type': 'rattle',
                        'frequency': random.uniform(200, 800),
                        'intensity': intensity * 0.3,
                        'duration': 0.05,
                        'delay': i * 0.02
                    })
                    
            elif sound_type == 'coin_ring':
                # Generate coin ringing
                coin_pitch = additional_sound.get('pitch', note)
                
                events.append({
                    'type': 'prepared_sound',
                    'sound_type': 'metallic_ring',
                    'note': coin_pitch,
                    'intensity': intensity,
                    'duration': 1.5 + intensity * 2.0,
                    'delay': 0.005
                })
        
        return events


class RealisticVelocityEngine:
    """Ultra-realistic velocity calculation based on actual piano mechanics"""
    
    def __init__(self):
        self.contact_times: Dict[str, Dict[str, float]] = {}
        self.key_weights = self._build_key_weight_map()
        self.velocity_curves = self._build_velocity_curves()
        
        # Enhanced parameters
        self.min_audible_velocity = 8
        self.max_velocity = 127
        self.optimal_time_window = (0.8, 12.0)
        
        # Human factors
        self.finger_fatigue_factor = 0.98
        self.hand_position_variance = 0.05
        self.key_depression_depth = {}
        
        # Performance tracking
        self.recent_velocities = deque(maxlen=20)
        self.playing_intensity = 0.5
        self.finger_independence = self._model_finger_characteristics()
        
    def _build_key_weight_map(self) -> Dict[int, float]:
        """Build realistic key weight distribution across keyboard"""
        weights = {}
        for note in range(21, 109):
            if note <= 28:
                weight = 1.4 + (28 - note) * 0.02
            elif note <= 48:
                weight = 1.2 + (48 - note) * 0.01
            elif note <= 60:
                weight = 1.0 + (60 - note) * 0.005
            elif note <= 72:
                weight = 1.0 - (note - 60) * 0.003
            elif note <= 84:
                weight = 0.92 - (note - 72) * 0.008
            else:
                weight = 0.75 - (note - 84) * 0.005
            
            weights[note] = max(0.5, min(1.6, weight))
        return weights
    
    def _build_velocity_curves(self) -> Dict[str, Callable]:
        """Build different velocity response curves for different registers"""
        def bass_curve(time_diff_ms: float) -> float:
            if time_diff_ms < 0.5:
                return 0.95
            elif time_diff_ms < 1.5:
                return 0.85 - (time_diff_ms - 0.5) * 0.15
            else:
                return max(0.1, 0.7 * np.exp(-time_diff_ms / 8.0))
        
        def middle_curve(time_diff_ms: float) -> float:
            if time_diff_ms < 0.3:
                return 0.98
            elif time_diff_ms < 1.0:
                return 0.9 - (time_diff_ms - 0.3) * 0.2
            else:
                return max(0.08, 0.75 * np.exp(-time_diff_ms / 10.0))
        
        def treble_curve(time_diff_ms: float) -> float:
            if time_diff_ms < 0.2:
                return 0.99
            elif time_diff_ms < 0.8:
                return 0.95 - (time_diff_ms - 0.2) * 0.3
            else:
                return max(0.05, 0.8 * np.exp(-time_diff_ms / 6.0))
        
        return {
            'bass': bass_curve,
            'middle': middle_curve, 
            'treble': treble_curve
        }
    
    def _model_finger_characteristics(self) -> Dict[str, float]:
        """Model individual finger characteristics"""
        return {
            'thumb': {'strength': 1.0, 'independence': 0.9, 'timing_variance': 0.02},
            'index': {'strength': 0.95, 'independence': 0.95, 'timing_variance': 0.015},
            'middle': {'strength': 1.0, 'independence': 1.0, 'timing_variance': 0.01},
            'ring': {'strength': 0.8, 'independence': 0.7, 'timing_variance': 0.025},
            'pinky': {'strength': 0.6, 'independence': 0.6, 'timing_variance': 0.035}
        }
    
    def calculate_realistic_velocity(self, key_id: str, contact: str, timestamp: float,
                                   midi_note: int, section: str, finger_hint: str = None) -> int:
        """Calculate ultra-realistic velocity with advanced modeling"""
        
        if key_id not in self.contact_times:
            self.contact_times[key_id] = {}
        
        self.contact_times[key_id][contact] = timestamp
        
        if contact != "break" or "make" not in self.contact_times[key_id]:
            return 64
        
        # Calculate time difference
        make_time = self.contact_times[key_id]["make"]
        break_time = self.contact_times[key_id]["break"]
        time_diff_ms = abs(break_time - make_time) * 1000
        
        del self.contact_times[key_id]
        
        # Determine register and curve
        if midi_note <= 47:
            curve_func = self.velocity_curves['bass']
            register = 'bass'
        elif midi_note <= 72:
            curve_func = self.velocity_curves['middle']
            register = 'middle'
        else:
            curve_func = self.velocity_curves['treble']
            register = 'treble'
        
        # Apply velocity curve
        velocity_ratio = curve_func(time_diff_ms)
        
        # Apply key weight
        key_weight = self.key_weights.get(midi_note, 1.0)
        velocity_ratio *= key_weight
        
        # Calculate base velocity
        base_velocity = int(self.min_audible_velocity + 
                           velocity_ratio * (self.max_velocity - self.min_audible_velocity))
        
        # Apply advanced modulations
        final_velocity = self._apply_advanced_modulations(
            base_velocity, midi_note, time_diff_ms, register, finger_hint
        )
        
        # Update tracking
        self.recent_velocities.append(final_velocity)
        self._update_playing_intensity()
        
        return max(self.min_audible_velocity, min(self.max_velocity, final_velocity))
    
    def _apply_advanced_modulations(self, base_velocity: int, midi_note: int, 
                                   time_diff_ms: float, register: str, finger_hint: str = None) -> int:
        """Apply advanced realistic modulations"""
        velocity = base_velocity
        
        # Finger-specific modulations
        if finger_hint and finger_hint in self.finger_independence:
            finger_data = self.finger_independence[finger_hint]
            
            # Strength variation
            strength_factor = 0.9 + finger_data['strength'] * 0.2
            velocity = int(velocity * strength_factor)
            
            # Independence affects timing consistency
            if finger_data['independence'] < 0.8:
                timing_error = random.gauss(0, finger_data['timing_variance'] * 1000)  # ms
                if abs(timing_error) > 2:  # Significant timing error
                    velocity_adjustment = -abs(timing_error) * 2  # Reduce velocity
                    velocity = max(velocity // 2, velocity + int(velocity_adjustment))
        
        # Micro-velocity variations (human inconsistency)
        if random.random() < 0.4:
            micro_var = random.gauss(0, 2.5)
            velocity += int(micro_var)
        
        # Playing intensity influence
        intensity_factor = 0.9 + (self.playing_intensity * 0.2)
        velocity = int(velocity * intensity_factor)
        
        # Register-specific characteristics
        if register == 'bass' and velocity > 100:
            velocity = min(127, int(velocity * 1.1))
        elif register == 'treble' and velocity < 30:
            velocity = max(25, velocity)
        
        # Fatigue simulation
        if len(self.recent_velocities) > 10:
            avg_recent = sum(list(self.recent_velocities)[-10:]) / 10
            if avg_recent > 90:
                fatigue_reduction = random.uniform(0.02, 0.08)
                velocity = int(velocity * (1.0 - fatigue_reduction))
        
        return velocity
    
    def _update_playing_intensity(self):
        """Update overall playing intensity"""
        if len(self.recent_velocities) >= 5:
            recent_avg = sum(list(self.recent_velocities)[-5:]) / 5
            target_intensity = recent_avg / 127.0
            self.playing_intensity = (self.playing_intensity * 0.8 + target_intensity * 0.2)


class TemporalDynamicEngine:
    """Engine for temporal and dynamic realism enhancements"""
    
    def __init__(self):
        self.attack_transients = self._model_attack_transients()
        self.polyphonic_interactions = PolyphonicInteractionModel()
        self.environmental_drift = EnvironmentalDriftModel()
        
    def _model_attack_transients(self) -> Dict[int, Dict]:
        """Model complex attack transients for each note"""
        transients = {}
        
        for note in range(21, 109):
            freq = 440.0 * (2 ** ((note - 69) / 12.0))
            
            # Attack phases
            if note <= 32:  # Bass
                phases = [
                    {'type': 'hammer_thunk', 'delay': 0.0, 'duration': 2.0, 'amplitude': 0.3},
                    {'type': 'string_excitation', 'delay': 0.5, 'duration': 3.0, 'amplitude': 1.0},
                    {'type': 'soundboard_response', 'delay': 1.0, 'duration': 8.0, 'amplitude': 0.8}
                ]
            elif note <= 60:  # Middle
                phases = [
                    {'type': 'hammer_thunk', 'delay': 0.0, 'duration': 1.2, 'amplitude': 0.2},
                    {'type': 'string_excitation', 'delay': 0.3, 'duration': 2.5, 'amplitude': 1.0},
                    {'type': 'soundboard_response', 'delay': 0.8, 'duration': 5.0, 'amplitude': 0.6}
                ]
            else:  # Treble
                phases = [
                    {'type': 'hammer_thunk', 'delay': 0.0, 'duration': 0.8, 'amplitude': 0.15},
                    {'type': 'string_excitation', 'delay': 0.2, 'duration': 1.5, 'amplitude': 1.0},
                    {'type': 'soundboard_response', 'delay': 0.5, 'duration': 3.0, 'amplitude': 0.4}
                ]
            
            transients[note] = {
                'phases': phases,
                'harmonic_decay_rates': self._calculate_harmonic_decay(note, freq),
                'inharmonicity': 0.0001 + (note - 21) * 0.000005  # Increases with pitch
            }
        
        return transients
    
    def _calculate_harmonic_decay(self, note: int, fundamental_freq: float) -> List[float]:
        """Calculate decay rates for each harmonic"""
        decay_rates = []
        
        for harmonic in range(1, 17):  # First 16 harmonics
            harmonic_freq = fundamental_freq * harmonic
            
            # Higher harmonics decay faster
            base_decay = 0.3 + harmonic * 0.08
            
            # Register effects
            if note <= 32:  # Bass
                decay_rate = base_decay * 0.8  # Slower decay
            elif note >= 72:  # Treble
                decay_rate = base_decay * 1.4  # Faster decay
            else:
                decay_rate = base_decay
            
            decay_rates.append(decay_rate)
        
        return decay_rates
    
    def process_attack_transient(self, note: int, velocity: int, 
                                hammer_data: Dict, timestamp: float) -> List[Dict]:
        """Process complex attack transient"""
        
        transient_data = self.attack_transients[note]
        transient_events = []
        
        velocity_factor = velocity / 127.0
        
        for phase in transient_data['phases']:
            phase_amplitude = phase['amplitude'] * velocity_factor
            
            # Hammer characteristics affect attack
            if phase['type'] == 'hammer_thunk':
                hardness_factor = hammer_data.get('effective_hardness', 0.7)
                phase_amplitude *= hardness_factor * 0.5 + 0.5
            
            elif phase['type'] == 'string_excitation':
                # Main tone - affected by hammer condition
                felt_factor = 1.0 - hammer_data.get('felt_compression', 0.0) * 0.2
                phase_amplitude *= felt_factor
            
            elif phase['type'] == 'soundboard_response':
                # Soundboard resonance
                resonance_factor = 0.8 + velocity_factor * 0.4
                phase_amplitude *= resonance_factor
            
            if phase_amplitude > 0.01:
                transient_events.append({
                    'type': 'attack_phase',
                    'phase_type': phase['type'],
                    'note': note,
                    'delay': phase['delay'] / 1000.0,  # Convert to seconds
                    'duration': phase['duration'] / 1000.0,
                    'amplitude': phase_amplitude,
                    'timestamp': timestamp
                })
        
        return transient_events


class PolyphonicInteractionModel:
    """Model interactions between simultaneous notes"""
    
    def __init__(self):
        self.active_notes = {}  # note -> {amplitude, phase, start_time}
        self.string_tension_effects = {}
        self.soundboard_loading = 0.0  # Current soundboard energy
        
    def add_note(self, note: int, velocity: int, timestamp: float):
        """Add note to polyphonic model"""
        amplitude = velocity / 127.0
        
        self.active_notes[note] = {
            'amplitude': amplitude,
            'phase': 0.0,
            'start_time': timestamp,
            'original_pitch': note
        }
        
        # Update soundboard loading
        self.soundboard_loading += amplitude * 0.1
        self.soundboard_loading = min(1.0, self.soundboard_loading)
        
        # Calculate interactions
        interactions = self._calculate_note_interactions(note, amplitude)
        
        return interactions
    
    def remove_note(self, note: int, timestamp: float):
        """Remove note from polyphonic model"""
        if note in self.active_notes:
            amplitude = self.active_notes[note]['amplitude']
            del self.active_notes[note]
            
            # Update soundboard loading
            self.soundboard_loading -= amplitude * 0.1
            self.soundboard_loading = max(0.0, self.soundboard_loading)
    
    def _calculate_note_interactions(self, new_note: int, new_amplitude: float) -> List[Dict]:
        """Calculate how new note interacts with existing notes"""
        interactions = []
        
        for existing_note, note_data in self.active_notes.items():
            if existing_note == new_note:
                continue
            
            # String tension coupling
            tension_effect = self._calculate_tension_coupling(new_note, existing_note, new_amplitude)
            if abs(tension_effect) > 0.01:
                interactions.append({
                    'type': 'tension_coupling',
                    'source_note': new_note,
                    'affected_note': existing_note,
                    'pitch_shift': tension_effect,  # cents
                    'amplitude_change': tension_effect * 0.1
                })
            
            # Beat frequencies
            beat_freq = self._calculate_beat_frequency(new_note, existing_note)
            if 0.5 < beat_freq < 10.0:  # Audible beats
                interactions.append({
                    'type': 'beat_frequency',
                    'note1': new_note,
                    'note2': existing_note,
                    'beat_frequency': beat_freq,
                    'modulation_depth': min(new_amplitude, note_data['amplitude']) * 0.3
                })
            
            # Soundboard competition
            competition_factor = self._calculate_soundboard_competition(new_note, existing_note)
            if competition_factor > 0.1:
                interactions.append({
                    'type': 'soundboard_competition',
                    'dominant_note': new_note if new_amplitude > note_data['amplitude'] else existing_note,
                    'suppressed_note': existing_note if new_amplitude > note_data['amplitude'] else new_note,
                    'suppression_amount': competition_factor
                })
        
        return interactions
    
    def _calculate_tension_coupling(self, note1: int, note2: int, amplitude: float) -> float:
        """Calculate pitch shift due to string tension coupling"""
        # Closer notes have stronger coupling
        semitone_distance = abs(note2 - note1)
        
        if semitone_distance > 12:
            return 0.0  # Too far apart
        
        # Coupling strength decreases with distance
        coupling_strength = 1.0 / (1.0 + semitone_distance * 0.5)
        
        # Amplitude affects coupling
        tension_change = amplitude * coupling_strength * 0.02
        
        # Convert to pitch shift (cents)
        pitch_shift = tension_change * 5.0  # Approximate conversion
        
        return pitch_shift
    
    def _calculate_beat_frequency(self, note1: int, note2: int) -> float:
        """Calculate beat frequency between two notes"""
        freq1 = 440.0 * (2 ** ((note1 - 69) / 12.0))
        freq2 = 440.0 * (2 ** ((note2 - 69) / 12.0))
        
        return abs(freq1 - freq2)
    
    def _calculate_soundboard_competition(self, note1: int, note2: int) -> float:
        """Calculate soundboard energy competition"""
        # Notes in same register compete more
        register1 = note1 // 12
        register2 = note2 // 12
        
        if abs(register1 - register2) > 2:
            return 0.0
        
        # Competition based on soundboard loading
        competition = self.soundboard_loading * 0.3
        
        return min(0.5, competition)


class EnvironmentalDriftModel:
    """Model environmental effects on tuning and timbre"""
    
    def __init__(self):
        self.temperature = 20.0  # Celsius
        self.humidity = 50.0     # Percent
        self.air_pressure = 1013.25  # hPa
        
        self.temperature_drift_rate = 0.001  # Per degree per minute
        self.humidity_drift_rate = 0.0005    # Per percent per minute
        
        self.last_update = time.time()
        self.cumulative_drift = defaultdict(float)  # note -> cents drift
        
    def update_environment(self, temp_change: float = None, 
                          humidity_change: float = None, 
                          pressure_change: float = None):
        """Update environmental conditions"""
        current_time = time.time()
        dt = (current_time - self.last_update) / 60.0  # Minutes
        
        if temp_change is not None:
            self.temperature += temp_change
        else:
            # Simulate gradual temperature drift
            self.temperature += random.gauss(0, 0.1) * dt
        
        if humidity_change is not None:
            self.humidity += humidity_change
        else:
            # Simulate gradual humidity drift
            self.humidity += random.gauss(0, 0.5) * dt
        
        self.humidity = max(10, min(90, self.humidity))
        
        # Update pitch drift for all notes
        self._update_pitch_drift(dt)
        
        self.last_update = current_time
    
    def _update_pitch_drift(self, dt: float):
        """Update cumulative pitch drift"""
        for note in range(21, 109):
            # Temperature effect (strings expand/contract)
            temp_effect = (self.temperature - 20.0) * self.temperature_drift_rate * dt
            
            # Humidity effect (soundboard swells/shrinks)
            humidity_effect = (self.humidity - 50.0) * self.humidity_drift_rate * dt
            
            # Total drift
            total_drift = temp_effect + humidity_effect
            
            # Different strings drift differently
            if note <= 32:  # Bass strings more stable
                total_drift *= 0.7
            elif note >= 80:  # Treble strings less stable
                total_drift *= 1.3
            
            self.cumulative_drift[note] += total_drift
            
            # Limit drift
            self.cumulative_drift[note] = max(-50, min(50, self.cumulative_drift[note]))
    
    def get_pitch_deviation(self, note: int) -> float:
        """Get current pitch deviation for a note in cents"""
        return self.cumulative_drift.get(note, 0.0)
    
    def get_timbre_modification(self, note: int) -> Dict:
        """Get timbre modifications due to environment"""
        temp_factor = (self.temperature - 20.0) / 30.0  # Normalized
        humidity_factor = (self.humidity - 50.0) / 40.0  # Normalized
        
        return {
            'brightness_change': temp_factor * 0.1,  # Warmer = brighter
            'sustain_change': -humidity_factor * 0.15,  # Higher humidity = less sustain
            'attack_sharpness': temp_factor * 0.05,
            'harmonic_content': temp_factor * 0.08
        }

def set_pi_volume(volume_percent: float, muted: bool = False):
    """Set Raspberry Pi audio volume using amixer"""
    try:
        if muted or volume_percent == 0:
            # Mute the audio
            subprocess.run(['amixer', 'sset', 'PCM', 'mute'], 
                         check=True, capture_output=True)
            print("🔇 System audio muted")
        else:
            # Unmute first, then set volume
            subprocess.run(['amixer', 'sset', 'PCM', 'unmute'], 
                         check=True, capture_output=True)
            
            # Set volume (amixer expects 0-100%)
            volume_str = f"{int(volume_percent)}%"
            subprocess.run(['amixer', 'sset', 'PCM', volume_str], 
                         check=True, capture_output=True)
            print(f"🔊 System volume set to {volume_percent:.1f}%")
            
    except subprocess.CalledProcessError as e:
        # Try alternative methods if PCM doesn't work
        try_alternative_volume_control(volume_percent, muted)
    except FileNotFoundError:
        logging.error("amixer not found - install alsa-utils: sudo apt install alsa-utils")

def try_alternative_volume_control(volume_percent: float, muted: bool):
    """Try alternative volume control methods"""
    try:
        # Method 1: Try 'Master' instead of 'PCM'
        if muted or volume_percent == 0:
            subprocess.run(['amixer', 'sset', 'Master', 'mute'], check=True, capture_output=True)
        else:
            subprocess.run(['amixer', 'sset', 'Master', 'unmute'], check=True, capture_output=True)
            volume_str = f"{int(volume_percent)}%"
            subprocess.run(['amixer', 'sset', 'Master', volume_str], check=True, capture_output=True)
        print(f"✓ Volume control using 'Master' - {volume_percent:.1f}%")
        
    except subprocess.CalledProcessError:
        try:
            # Method 2: Use PulseAudio if available
            if muted or volume_percent == 0:
                subprocess.run(['pactl', 'set-sink-mute', '@DEFAULT_SINK@', '1'], 
                             check=True, capture_output=True)
            else:
                subprocess.run(['pactl', 'set-sink-mute', '@DEFAULT_SINK@', '0'], 
                             check=True, capture_output=True)
                # PulseAudio expects volume as percentage
                subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', f"{int(volume_percent)}%"], 
                             check=True, capture_output=True)
            print(f"✓ Volume control using PulseAudio - {volume_percent:.1f}%")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.error("Could not control volume with amixer or pactl")

# Optional: Get current system volume
def get_current_pi_volume():
    """Get current Pi volume for initialization"""
    try:
        result = subprocess.run(['amixer', 'sget', 'PCM'], 
                              capture_output=True, text=True, check=True)
        # Parse amixer output to extract volume percentage
        # This is a simplified parser - amixer output can vary
        lines = result.stdout.split('\n')
        for line in lines:
            if '[' in line and '%' in line:
                # Extract percentage from something like "[50%]"
                start = line.find('[') + 1
                end = line.find('%')
                if start > 0 and end > start:
                    return float(line[start:end])
    except:
        pass
    return 50.0  # Default fallback

class DelayedEventScheduler:
    """Centralized scheduler for delayed events"""
    def __init__(self):
        self.queue = []
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def schedule(self, delay_seconds, callback, args=()):
        event_time = time.time() + delay_seconds
        with self.lock:
            heapq.heappush(self.queue, (event_time, callback, args))

    def run(self):
        while self.running:
            now = time.time()
            events_to_run = []
            with self.lock:
                while self.queue and self.queue[0][0] <= now:
                    event = heapq.heappop(self.queue)
                    events_to_run.append(event)
            for event in events_to_run:
                _, callback, args = event
                try:
                    callback(*args)
                except Exception as e:
                    logger.error(f"Scheduled event error: {e}")
            time.sleep(0.01)  # 10ms sleep to prevent busy-waiting

class PianoSystem:
    
    def __init__(self):
        self.logger = logging.getLogger("UltraRealisticPiano")
        # Use the central config and sound_params
        self.config = _cfg
        self.sound_params = _params
        self.setup_logging()

        # Enhanced realistic components
        self.key_action_engine   = AdvancedKeyActionEngine()
        self.hammer_engine       = AdvancedHammerModelingEngine()
        self.velocity_engine     = RealisticVelocityEngine()
        self.resonance_engine    = ComprehensiveStringResonanceEngine()
        self.temporal_engine     = TemporalDynamicEngine()
        self.extended_techniques = ExtendedTechniquesEngine()

        # Centralized scheduler for delayed events
        self.scheduler = DelayedEventScheduler()

        # Original components
        self.key_mapper          = RealisticPianoKeyMapper()
        self.expression_manager  = PianoExpressionManager()

        # Audio system
        sf_path = self.config.get("audio", {}) \
                             .get("fluidsynth", {}) \
                             .get("soundfont_path")
        self.fluidsynth = EnhancedFluidSynthController(soundfont_path=sf_path)

        # Device management
        self.monitors         = []                # type: List[ESP32Monitor]
        self.midi_out         = None
        self.running          = False

        # Enhanced tracking
        self.active_notes         = {}            # type: Dict[int, Dict]
        self.current_held_notes   = set()         # type: Set[int]
        self.pedal_states         = {
            'sustain': False,
            'sostenuto': False,
            'soft': False
        }

        # Performance analytics
        self.note_statistics = {
            'total_notes': 0,
            'velocity_histogram': defaultdict(int),
            'timing_accuracy': deque(maxlen=100),
            'pedal_usage': defaultdict(int),
            'chord_detections': 0,
            'extended_techniques': defaultdict(int),
        }

        # Advanced timing
        self.last_note_time         = 0.0
        self.chord_detection_window = 0.05
        self.pending_chord_notes    = []

        # Environmental simulation: schedule periodic updates
        self.environmental_updates = threading.Timer(60.0, self._update_environment)
    
    def get_default_config(self) -> dict:
        """Enhanced default configuration"""
        return {
            "midi": {"virtual_port_name": "UltraRealisticPiano", "channel": 0},
            "devices": {"auto_detect": True, "manual_mapping": {}},
            "audio": {
                "fluidsynth": {
                    "enabled": True,
                    "soundfont_path": "/home/admin/piano/sounds/UprightPianoKW-20220221.sf2"
                }
            },
            "logging": {"level": "INFO"},
            "piano": {
                "ultra_realistic_mode": True,
                "physical_modeling": True,
                "advanced_acoustics": True,
                "temporal_dynamics": True,
                "extended_techniques": True,
                "environmental_simulation": True,
                "pedal_physics": True
            }
        }
    
    def setup_logging(self):
        """Enhanced logging setup"""
        log_level = self.config.get("logging", {}).get("level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('ultra_realistic_piano.log')
            ]
        )
        
        # Performance logging
        perf_logger = logging.getLogger("Performance")
        perf_handler = logging.FileHandler('piano_performance.log')
        perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
    
    def start(self):
        """Start the enhanced piano system"""
        self.logger.info("Starting system")
        
        # Start the scheduler
        self.scheduler.start()
        
        # Initialise FluidSynth
        if not self.fluidsynth.start():
            self.logger.warning("FluidSynth failed to start")
        
        time.sleep(2.5)
        
        # Setup MIDI
        if not self.setup_midi():
            return False
        
        # Initialise pedal system
        self.pedal_system = UltraAdvancedPedalSystem(self.send_midi_cc)
        
        # Find and setup devices
        devices = self.find_esp32_devices()
        if not devices:
            self.logger.error("No ESP32 devices found!")
            return False
        
        # Create monitors
        for port, device_type in devices.items():
            volume_cb = self.handle_volume_change if device_type == DeviceType.CONTROLLER  else None
            
            monitor = ESP32Monitor(
                port=port,
                device_type=device_type,
                key_callback=self.handle_enhanced_key_event,
                pedal_callback=self.handle_enhanced_pedal_event,
                volume_callback=volume_cb,  # Pass callback
                config=self.config
            )
            
            if monitor.connect():
                monitor.start_monitoring()
                self.monitors.append(monitor)
        
        if not self.monitors:
            self.logger.error("No ESP32 devices connected!")
            return False
        
        self.running = True
        
        # Start background processes
        self._start_background_processes()

        self.logger.info("🎹 System active")
        return True
    
    def _start_background_processes(self):
        def background_worker():
            last_resonance = 0
            last_environment = 0
            last_performance = 0
            
            while self.running:
                current_time = time.time()
                
                # Resonance updates (50Hz)
                if current_time - last_resonance >= 0.02:
                    try:
                        self.resonance_engine.soundboard_model.update_environment()
                        last_resonance = current_time
                    except Exception as e:
                        self.logger.error(f"Resonance update error: {e}")
                
                # Environmental updates (every 30 seconds)
                if current_time - last_environment >= 30:
                    try:
                        self._update_environment()
                        last_environment = current_time
                    except Exception as e:
                        self.logger.error(f"Environmental update error: {e}")
                
                # Performance analysis (every 1 second)
                if current_time - last_performance >= 1:
                    try:
                        self._Analyse_enhanced_performance()
                        last_performance = current_time
                    except Exception as e:
                        self.logger.error(f"Performance analysis error: {e}")
                
                time.sleep(0.01)  # Reduce CPU usage
        
        # Start only one background thread
        threading.Thread(target=background_worker, daemon=True).start()
    
    def handle_volume_change(self, event: VolumeEvent):
        print(f"Volume: {event.volume_percent:.1f}% "
              f"({'MUTED' if event.muted else 'UNMUTED'})")
        
        try:
            set_pi_volume(event.volume_percent, muted=event.muted)
        except Exception as e:
            logging.error(f"Failed to set system volume: {e}")
            
    def handle_enhanced_key_event(self, event: KeyEvent):
        """Handle key events with full enhancement processing"""
        current_time = time.time()
        
        # Convert to MIDI note
        midi_note = self.key_mapper.get_midi_note(event.key_number, event.section)
        
        # Process key action physics
        key_motion_data = self.key_action_engine.process_key_motion(
            f"{event.device_id}_{event.key_number}",
            event.contact,
            event.timestamp,
            midi_note
        )
        
        # Calculate realistic velocity
        velocity = self.velocity_engine.calculate_realistic_velocity(
            f"{event.device_id}_{event.key_number}",
            event.contact,
            event.timestamp,
            midi_note,
            event.section
        )
        
        # Apply pedal effects
        if hasattr(self, 'pedal_system'):
            soft_effect = self.pedal_system.pedal_positions.get('soft', 0.0)
            velocity = int(velocity * (1.0 - soft_effect * 0.4))
        
        if event.action == "press" and event.contact == "break":
            self._trigger_enhanced_note(midi_note, velocity, current_time, key_motion_data)
        elif event.action == "release" and event.contact == "make":
            self._release_enhanced_note(midi_note, velocity, current_time)
    
    def _trigger_enhanced_note(self, midi_note: int, velocity: int, 
                              timestamp: float, key_motion_data: Dict):
        """Trigger note with full enhancement processing"""
        
        # Process hammer strike
        hammer_data = self.hammer_engine.calculate_hammer_strike(
            midi_note, velocity, key_motion_data, timestamp
        )
        
        # Check for extended techniques
        extended_effects = self._check_extended_techniques(midi_note, velocity, key_motion_data)
        
        # Process attack transients
        attack_events = self.temporal_engine.process_attack_transient(
            midi_note, velocity, hammer_data, timestamp
        )
        
        # Environmental effects
        pitch_deviation = self.temporal_engine.environmental_drift.get_pitch_deviation(midi_note)
        timbre_mods = self.temporal_engine.environmental_drift.get_timbre_modification(midi_note)
        
        # Apply pitch deviation
        if abs(pitch_deviation) > 1.0:  # Significant deviation
            # Send pitch bend (simplified - would need proper implementation)
            self.logger.debug(f"Pitch deviation: {pitch_deviation:.2f} cents")
        
        # Main note trigger
        final_velocity = self._apply_timbre_modifications(velocity, timbre_mods)
        
        # Check for chord detection
        if self._is_chord_note(timestamp):
            self.pending_chord_notes.append((midi_note, final_velocity, timestamp, hammer_data))
            # Schedule chord processing with scheduler
            self.scheduler.schedule(self.chord_detection_window, self._process_enhanced_chord)
            return
        
        # Single note processing
        self._send_enhanced_note(midi_note, final_velocity, timestamp, hammer_data, attack_events)
        
        # Process extended technique effects
        if extended_effects:
            self._process_extended_effects(extended_effects, timestamp)
    
    def _check_extended_techniques(self, midi_note: int, velocity: int, 
                                  key_motion_data: Dict) -> Optional[Dict]:
        """Check for extended piano techniques"""
        
        # Silent key press detection
        if velocity < 20 and key_motion_data['position'] > 0.6:
            return self.extended_techniques.detect_silent_key_press(
                midi_note, key_motion_data['position'], velocity, time.time()
            )
        
        # Check for preparations
        if midi_note in self.extended_techniques.prepared_notes:
            return self.extended_techniques.calculate_preparation_effect(
                midi_note, velocity, key_motion_data
            )
        
        return None
    
    def _apply_timbre_modifications(self, velocity: int, timbre_mods: Dict) -> int:
        """Apply environmental timbre modifications"""
        modified_velocity = velocity
        
        # Brightness change
        brightness_change = timbre_mods.get('brightness_change', 0.0)
        if brightness_change != 0.0:
            modified_velocity = int(velocity * (1.0 + brightness_change))
        
        # Attack sharpness
        attack_change = timbre_mods.get('attack_sharpness', 0.0)
        if attack_change > 0.0:
            modified_velocity = min(127, int(modified_velocity * (1.0 + attack_change)))
        
        return max(1, min(127, modified_velocity))
    
    def _is_chord_note(self, timestamp: float) -> bool:
        """Enhanced chord detection"""
        return (timestamp - self.last_note_time) < self.chord_detection_window
    
    def _process_enhanced_chord(self):
        """Process chord with enhanced voicing and timing"""
        if not self.pending_chord_notes:
            return
        
        if len(self.pending_chord_notes) >= 2:
            self.note_statistics['chord_detections'] += 1
            self.logger.debug(f"Enhanced chord detected: {len(self.pending_chord_notes)} notes")
            
            # Sort by pitch for natural voicing
            self.pending_chord_notes.sort(key=lambda x: x[0])
            
            # Apply chord-specific processing
            for i, (note, velocity, timestamp, hammer_data) in enumerate(self.pending_chord_notes):
                # Natural timing spread (bass first, treble last)
                chord_delay = i * 0.003  # 3ms per note
                
                # Voice balancing
                if i == 0:  # Bass note - strongest
                    voice_factor = 1.0
                elif i == len(self.pending_chord_notes) - 1:  # Melody note
                    voice_factor = 0.98
                else:  # Inner voices
                    voice_factor = 0.92
                
                adjusted_velocity = int(velocity * voice_factor)
                
                # Process attack transients for chord
                attack_events = self.temporal_engine.process_attack_transient(
                    note, adjusted_velocity, hammer_data, timestamp
                )
                
                # Schedule note with scheduler
                self.scheduler.schedule(chord_delay, self._send_enhanced_note, 
                                      (note, adjusted_velocity, timestamp, hammer_data, attack_events))
        else:
            # Single note after all
            note, velocity, timestamp, hammer_data = self.pending_chord_notes[0]
            attack_events = self.temporal_engine.process_attack_transient(
                note, velocity, hammer_data, timestamp
            )
            self._send_enhanced_note(note, velocity, timestamp, hammer_data, attack_events)
        
        self.pending_chord_notes.clear()
    
    def _send_enhanced_note(self, midi_note: int, velocity: int, timestamp: float,
                           hammer_data: Dict, attack_events: List[Dict]):
        """Send note with comprehensive enhancement processing"""
        
        # Add to polyphonic interaction model
        polyphonic_effects = self.temporal_engine.polyphonic_interactions.add_note(
            midi_note, velocity, timestamp
        )
        
        # Process comprehensive string resonance
        resonant_responses = self.resonance_engine.trigger_advanced_resonance(
            midi_note, velocity, hammer_data, timestamp, self.pedal_states
        )
        
        # Main note
        self.send_midi_note_on(midi_note, velocity)
        
        # Store comprehensive note data
        self.active_notes[midi_note] = {
            'velocity': velocity,
            'start_time': timestamp,
            'hammer_data': hammer_data,
            'attack_events': attack_events,
            'resonant_notes': [],
            'polyphonic_effects': polyphonic_effects,
            'environmental_mods': self.temporal_engine.environmental_drift.get_timbre_modification(midi_note)
        }
        
        # Process attack transient events
        for event in attack_events:
            self._process_attack_event(event)
        
        # Send resonant notes with advanced timing using scheduler
        for resonant_note, resonant_velocity, delay, resonance_data in resonant_responses:
            self.scheduler.schedule(delay, self._send_enhanced_resonant_note,
                                  (resonant_note, resonant_velocity, midi_note, resonance_data))
        
        # Apply polyphonic effects
        for effect in polyphonic_effects:
            self._apply_polyphonic_effect(effect)
        
        # Update statistics
        self.note_statistics['total_notes'] += 1
        self.note_statistics['velocity_histogram'][velocity // 10] += 1
        self.last_note_time = timestamp
        self.current_held_notes.add(midi_note)
        
        self.logger.info(f"🎵 Enhanced Note ON: {midi_note} vel:{velocity} "
                        f"(+{len(resonant_responses)} resonant, +{len(attack_events)} transients)")
    
    def _process_attack_event(self, event: Dict):
        """Process individual attack transient events"""
        event_type = event.get('phase_type')
        
        if event_type == 'hammer_thunk':
            # Low-frequency percussive sound
            # Could trigger additional MIDI events or samples
            self.logger.debug(f"Hammer thunk: {event['amplitude']:.2f}")
        
        elif event_type == 'string_excitation':
            # Main string resonance - already handled by main note
            pass
        
        elif event_type == 'soundboard_response':
            # Soundboard resonance could add reverb or resonant filtering
            self.logger.debug(f"Soundboard response: {event['amplitude']:.2f}")
    
    def _send_enhanced_resonant_note(self, note: int, velocity: int, 
                                   source_note: int, resonance_data: Dict):
        """Send enhanced resonant note with characteristics"""
        
        # Apply pitch deviation if significant
        pitch_deviation = resonance_data.get('pitch_deviation', 0.0)
        if abs(pitch_deviation) > 5.0:  # More than 5 cents
            # Would implement micro-tuning here
            self.logger.debug(f"Resonant note {note} pitch deviation: {pitch_deviation:.1f} cents")
        
        # Apply inharmonicity effects
        inharmonicity = resonance_data.get('inharmonicity', 0.0)
        if inharmonicity > 0.001:
            # Slightly adjust velocity for inharmonic content
            velocity = int(velocity * (1.0 + inharmonicity * 50))
        
        self.send_midi_note_on(note, velocity)
        
        # Store reference
        if source_note in self.active_notes:
            self.active_notes[source_note]['resonant_notes'].append(note)
        
        # Auto-release with enhanced decay using scheduler
        decay_rate = resonance_data.get('decay_rate', 0.3)
        release_time = 1.0 / decay_rate + random.uniform(0.1, 0.3)
        
        self.scheduler.schedule(release_time, self.send_midi_note_off, (note, 20))
        
        coupling_type = resonance_data.get('coupling_type', 'unknown')
        self.logger.debug(f"🌊 Enhanced Resonance: {note} vel:{velocity} "
                         f"({coupling_type} from {source_note})")
    
    def _apply_polyphonic_effect(self, effect: Dict):
        """Apply polyphonic interaction effects"""
        effect_type = effect['type']
        
        if effect_type == 'tension_coupling':
            # Slight pitch bend due to string tension
            affected_note = effect['affected_note']
            pitch_shift = effect['pitch_shift']
            
            self.logger.debug(f"Tension coupling: Note {affected_note} shifted {pitch_shift:.2f} cents")
        
        elif effect_type == 'beat_frequency':
            # Beat frequency between notes
            beat_freq = effect['beat_frequency']
            modulation_depth = effect['modulation_depth']
            
            self.logger.debug(f"Beat frequency: {beat_freq:.1f} Hz, depth: {modulation_depth:.2f}")
        
        elif effect_type == 'soundboard_competition':
            # One note suppresses another
            dominant = effect['dominant_note']
            suppressed = effect['suppressed_note']
            suppression = effect['suppression_amount']
            
            self.logger.debug(f"Soundboard competition: {dominant} suppresses {suppressed} by {suppression:.2f}")
    
    def _release_enhanced_note(self, midi_note: int, velocity: int, timestamp: float):
        """Release note with enhanced processing"""
        
        # Check damper state
        damper_state = self.pedal_system.get_damper_state(midi_note)
        
        if damper_state['lifted']:
            self.logger.debug(f"Note {midi_note} sustained by damper")
            return
        
        # Calculate enhanced release characteristics
        if midi_note in self.active_notes:
            note_data = self.active_notes[midi_note]
            hold_duration = timestamp - note_data['start_time']
            
            # Enhanced release velocity calculation
            base_release_velocity = self._calculate_release_velocity(
                velocity, hold_duration, note_data.get('hammer_data', {})
            )
            
            # Apply environmental effects to release
            env_mods = note_data.get('environmental_mods', {})
            sustain_change = env_mods.get('sustain_change', 0.0)
            
            if sustain_change != 0.0:
                # Modify release based on environmental sustain change
                release_velocity = int(base_release_velocity * (1.0 + sustain_change))
            else:
                release_velocity = base_release_velocity
            
            # Release main note
            self.send_midi_note_off(midi_note, release_velocity)
            
            # Release associated resonant notes with natural timing
            for i, resonant_note in enumerate(note_data.get('resonant_notes', [])):
                release_delay = i * 0.01 + random.uniform(0.01, 0.05)
                self.scheduler.schedule(release_delay, self.send_midi_note_off, 
                                     (resonant_note, max(15, release_velocity // 2)))
            
            # Remove from polyphonic model
            self.temporal_engine.polyphonic_interactions.remove_note(midi_note, timestamp)
            
            del self.active_notes[midi_note]
            self.logger.info(f"🎵 Enhanced Note OFF: {midi_note} rel_vel:{release_velocity}")
        
        self.current_held_notes.discard(midi_note)
    
    def _calculate_release_velocity(self, press_velocity: int, hold_duration: float, 
                                   hammer_data: Dict) -> int:
        """Calculate enhanced release velocity"""
        
        if hold_duration < 0.1:  # Staccato
            release_velocity = min(90, press_velocity + 15)
        elif hold_duration > 5.0:  # Very long held note
            release_velocity = max(10, press_velocity // 5)
        else:  # Normal release
            # Factor in hammer characteristics
            felt_hardness = hammer_data.get('effective_hardness', 0.7)
            hardness_factor = 0.8 + felt_hardness * 0.4
            
            release_velocity = max(15, int((press_velocity // 3) * hardness_factor))
        
        return release_velocity
    
    def handle_enhanced_pedal_event(self, event: PedalEvent):
        """Handle pedal events with enhanced processing"""
        if not hasattr(self, 'pedal_system'):
            return
        
        # Update pedal states
        self.pedal_states[event.pedal_name] = (event.action == "press")
        
        # Process with enhanced pedal system
        mechanical_effects = self.pedal_system.handle_advanced_pedal_event(
            event.pedal_name, 
            event.action, 
            self.current_held_notes.copy(), 
            event.timestamp
        )
        
        # Process mechanical effects
        for effect in mechanical_effects:
            self._process_pedal_effect(effect)
        
        # Check for resonance capture
        if event.pedal_name == 'sustain' and event.action == "press":
            # Capture any decaying resonances
            active_resonances = {note: data for note, data in self.active_notes.items()}
            capture_result = self.extended_techniques.handle_resonance_capture(
                True, active_resonances, event.timestamp
            )
            
            if capture_result['type'] == 'resonance_capture':
                self.logger.info(f"🎭 Resonance capture: {len(capture_result['captured_notes'])} notes extended")
                self.note_statistics['extended_techniques']['resonance_capture'] += 1
        
        # Update statistics
        self.note_statistics['pedal_usage'][event.pedal_name] += 1
        
        self.logger.info(f"🦶 Enhanced Pedal {event.pedal_name}: {event.action}")
    
    def _process_pedal_effect(self, effect: Dict):
        """Process individual pedal mechanical effects"""
        effect_type = effect['type']
        
        if effect_type == 'damper_lift':
            note = effect['note']
            amount = effect['amount']
            self.logger.debug(f"Damper lift: Note {note}, amount {amount:.2f}")
        
        elif effect_type == 'mechanical_noise':
            sound = effect['sound']
            intensity = effect['intensity']
            self.logger.debug(f"Pedal noise: {sound}, intensity {intensity:.2f}")
        
        elif effect_type == 'una_corda_effect':
            note = effect['note']
            shift = effect['shift_amount']
            brightness = effect['brightness_factor']
            volume = effect['volume_factor']
            
            # Could apply these effects to active notes
            self.logger.debug(f"Una corda: Note {note}, shift {shift:.2f}, "
                            f"brightness {brightness:.2f}, volume {volume:.2f}")
    
    def _process_extended_effects(self, extended_effects: Dict, timestamp: float):
        """Process extended piano technique effects"""
        
        if extended_effects.get('type') == 'harmonic_excitation':
            responses = extended_effects.get('responses', [])
            
            for response in responses:
                harmonic_note = response['harmonic_note']
                strength = response['strength']
                harmonic_num = response['harmonic_number']
                
                # Trigger harmonic
                harmonic_velocity = int(strength * 80)  # Scale to MIDI velocity
                self.send_midi_note_on(harmonic_note, harmonic_velocity)
                
                # Auto-release harmonic with scheduler
                release_time = 0.5 + strength * 2.0
                self.scheduler.schedule(release_time, self.send_midi_note_off, 
                                     (harmonic_note, 10))
                
                self.logger.info(f"🎭 Natural harmonic: {harmonic_note} (harmonic {harmonic_num})")
                self.note_statistics['extended_techniques']['natural_harmonics'] += 1
        
        elif extended_effects.get('modified', False):
            prep_type = extended_effects.get('preparation_type')
            if prep_type:
                self.logger.info(f"🎭 Prepared piano effect: {prep_type}")
                self.note_statistics['extended_techniques'][prep_type] += 1
                
                # Generate additional prepared sounds
                additional_sounds = self.extended_techniques.generate_prepared_sound_events(
                    0, 64, extended_effects, timestamp  # Simplified parameters
                )
                
                for sound_event in additional_sounds:
                    self._trigger_prepared_sound(sound_event)
    
    def _trigger_prepared_sound(self, sound_event: Dict):
        """Trigger additional sounds from preparations"""
        sound_type = sound_event['type']
        delay = sound_event.get('delay', 0.0)
        
        def play_prepared_sound():
            if sound_type == 'buzz':
                # Metallic buzz - could use a specific MIDI note or sample
                freq = sound_event['frequency']
                intensity = sound_event['intensity']
                duration = sound_event['duration']
                
                # Convert frequency to approximate MIDI note
                midi_note = int(69 + 12 * math.log2(freq / 440.0))
                velocity = int(intensity * 127)
                
                self.send_midi_note_on(midi_note, velocity)
                self.scheduler.schedule(duration, self.send_midi_note_off, (midi_note, 20))
            
            elif sound_type == 'metallic_ring':
                note = sound_event['note']
                intensity = sound_event['intensity']
                duration = sound_event['duration']
                
                velocity = int(intensity * 127)
                self.send_midi_note_on(note, velocity)
                self.scheduler.schedule(duration, self.send_midi_note_off, (note, 30))
        
        if delay > 0:
            self.scheduler.schedule(delay, play_prepared_sound)
        else:
            play_prepared_sound()
    
    def _update_environment(self):
        """Update environmental conditions"""
        # Simulate gradual environmental changes
        temp_change = random.gauss(0, 0.5)  # ±0.5°C variation
        humidity_change = random.gauss(0, 2.0)  # ±2% humidity variation
        
        self.temporal_engine.environmental_drift.update_environment(
            temp_change, humidity_change
        )
        
        # Update string and soundboard temperatures
        new_temp = self.temporal_engine.environmental_drift.temperature
        self.resonance_engine.temperature = new_temp
        self.hammer_engine.temperature = new_temp
        
        self.logger.debug(f"Environment: {new_temp:.1f}°C, "
                         f"{self.temporal_engine.environmental_drift.humidity:.1f}% RH")
    
    def _Analyse_enhanced_performance(self):
        """Analyse enhanced performance patterns"""
        if len(self.velocity_engine.recent_velocities) < 5:
            return
        
        # Analyse velocity patterns
        recent_velocities = list(self.velocity_engine.recent_velocities)
        avg_velocity = sum(recent_velocities) / len(recent_velocities)
        velocity_variance = np.var(recent_velocities)
        
        # Detect playing style
        if avg_velocity > 95 and velocity_variance > 500:
            playing_style = "fortissimo"
        elif avg_velocity > 80 and velocity_variance > 300:
            playing_style = "forte"
        elif avg_velocity < 40 and velocity_variance < 80:
            playing_style = "pianissimo"
        elif avg_velocity < 60 and velocity_variance < 150:
            playing_style = "piano"
        else:
            playing_style = "mezzo"
        
        # Calculate chord ratio
        chord_ratio = 0.0
        if self.note_statistics['total_notes'] > 0:
            chord_ratio = self.note_statistics['chord_detections'] / max(1, self.note_statistics['total_notes'] // 3)
        
        # Extended techniques usage
        extended_usage = sum(self.note_statistics['extended_techniques'].values())
        
        # Log comprehensive performance insights
        perf_logger = logging.getLogger("Performance")
        perf_logger.info(f"Style: {playing_style}, Avg Vel: {avg_velocity:.1f}, "
                        f"Variance: {velocity_variance:.1f}, Chords: {chord_ratio:.2f}, "
                        f"Extended: {extended_usage}, Total: {self.note_statistics['total_notes']}")
    
    def add_preparation(self, note: int, preparation_type: str, **params) -> Dict:
        """Add a preparation to a string (public API)"""
        result = self.extended_techniques.add_preparation(note, preparation_type, **params)
        self.logger.info(f"🎭 Added preparation: {preparation_type} on note {note}")
        return result
    
    def remove_preparation(self, note: int) -> bool:
        """Remove preparation from a string"""
        if note in self.extended_techniques.prepared_notes:
            prep_type = self.extended_techniques.prepared_notes[note]['type']
            del self.extended_techniques.prepared_notes[note]
            self.logger.info(f"🎭 Removed preparation: {prep_type} from note {note}")
            return True
        return False
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'active_notes': len(self.active_notes),
            'held_notes': len(self.current_held_notes),
            'pedal_positions': getattr(self.pedal_system, 'pedal_positions', {}),
            'environment': {
                'temperature': self.temporal_engine.environmental_drift.temperature,
                'humidity': self.temporal_engine.environmental_drift.humidity
            },
            'statistics': dict(self.note_statistics),
            'preparations': len(self.extended_techniques.prepared_notes),
            'polyphonic_notes': len(self.temporal_engine.polyphonic_interactions.active_notes),
            'thread_count': threading.active_count()
        }
    
    def find_esp32_devices(self) -> Dict[str, DeviceType]:
        """Find ESP32 devices"""
        devices = {}
        device_config = self.config.get("devices", {})
        manual_mapping = device_config.get("manual_mapping", {})
        auto_detect = device_config.get("auto_detect", True)
        
        if manual_mapping and not auto_detect:
            for port, device_type_str in manual_mapping.items():
                try:
                    device_type = DeviceType(device_type_str)
                    if os.path.exists(port):
                        devices[port] = device_type
                except ValueError:
                    continue
        else:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                if any(x in port.description for x in ['USB', 'ESP32']) or any(x in port.device for x in ['ttyUSB', 'ttyACM']):
                    device_type = self.identify_device(port.device)
                    if device_type:
                        devices[port.device] = device_type
        
        return devices
    
    def identify_device(self, port: str) -> Optional[DeviceType]:
        """Identify device type"""
        try:
            test_conn = serial.Serial(port=port, baudrate=115200, timeout=2.0)
            time.sleep(1)
            
            for _ in range(10):
                if test_conn.in_waiting > 0:
                    line = test_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        try:
                            data = json.loads(line)
                            device_id = data.get('device')
                            if device_id == "MASTER_BASS_PEDALS":
                                test_conn.close()
                                return DeviceType.MASTER_BASS_PEDALS
                            elif device_id == "SLAVE_TREBLE":
                                test_conn.close()
                                return DeviceType.SLAVE_TREBLE
                            elif device_id == "CONTROLLER":
                                return DeviceType.CONTROL
                        except json.JSONDecodeError:
                            continue
                time.sleep(0.1)
            
            test_conn.close()
            return None
        except Exception:
            return None
    
    def setup_midi(self) -> bool:
        """Setup MIDI"""
        try:
            time.sleep(1)
            available_ports = mido.get_output_names()
            
            target_port = None
            for port in available_ports:
                if any(x in port.lower() for x in ["fluidsynth", "fluid", "synth"]):
                    target_port = port
                    break
            
            if target_port:
                self.midi_out = mido.open_output(target_port)
                self.logger.info(f"✓ Connected MIDI to: {target_port}")
                return True
            else:
                self.logger.error(f"No MIDI port found. Available: {available_ports}")
                return False
        except Exception as e:
            self.logger.error(f"MIDI setup failed: {e}")
            return False
    
    def send_midi_note_on(self, note: int, velocity: int):
        """Send MIDI note on"""
        if self.midi_out:
            msg = mido.Message('note_on', channel=0, note=note, velocity=velocity)
            self.midi_out.send(msg)
    
    def send_midi_note_off(self, note: int, velocity: int = 0):
        """Send MIDI note off"""
        if self.midi_out:
            msg = mido.Message('note_off', channel=0, note=note, velocity=velocity)
            self.midi_out.send(msg)
    
    def send_midi_cc(self, cc_number: int, value: int):
        """Send MIDI control change"""
        if self.midi_out:
            msg = mido.Message('control_change', channel=0, control=cc_number, value=value)
            self.midi_out.send(msg)
    
    def stop(self):
        """Stop the enhanced system"""
        self.logger.info("Stopping system")
        self.running = False
        
        # Stop scheduler
        self.scheduler.stop()
        
        # Stop monitors
        for monitor in self.monitors:
            monitor.stop_monitoring()
        
        # Release all notes
        for note in list(self.active_notes.keys()):
            self.send_midi_note_off(note, 40)
        
        # Reset pedals
        if hasattr(self, 'pedal_system'):
            for cc in [64, 66, 67]:
                self.send_midi_cc(cc, 0)
        
        if self.midi_out:
            self.midi_out.close()
        
        self.fluidsynth.stop()
        
        # Log final statistics
        total_notes = self.note_statistics['total_notes']
        chords = self.note_statistics['chord_detections']
        extended = sum(self.note_statistics['extended_techniques'].values())
        
        self.logger.info(f"Session complete: {total_notes} notes, {chords} chords")
    
# Support classes
class ESP32Monitor:
    """ESP32 device monitor"""
    def __init__(self, port: str, device_type: DeviceType, 
                 key_callback: Callable[[KeyEvent], None],
                 pedal_callback: Callable[[PedalEvent], None],
                 config: dict = None,
                 volume_callback: Callable[[VolumeEvent], None] = None):
        self.port = port
        self.device_type = device_type
        self.key_callback = key_callback
        self.pedal_callback = pedal_callback
        self.volume_callback = volume_callback
        self.serial_conn: Optional[serial.Serial] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None,
        self.config = config or {},
        self.logger = logging.getLogger(f"ESP32-{device_type.value}")
        
    def connect(self) -> bool:
        try:
            self.serial_conn = serial.Serial(
                port=self.port, baudrate=115200, timeout=1.0,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            time.sleep(2)
            self.logger.info(f"Connected to {self.device_type.value} on {self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.port}: {e}")
            return False
    
    def start_monitoring(self):
        if self.serial_conn and not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
    
    def stop_monitoring(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.serial_conn:
            self.serial_conn.close()
    
    def _monitor_loop(self):
        buffer = ""
        while self.running and self.serial_conn:
            try:
                if self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    
                    # Ensure we only work with string data
                    if isinstance(data, bytes):
                        try:
                            decoded = data.decode('utf-8', errors='ignore')
                            buffer += decoded
                        except UnicodeDecodeError:
                            buffer = ""
                    elif isinstance(data, str):
                        buffer += data
                    
                    # Process complete lines (only if buffer is string)
                    if isinstance(buffer, str) and '\n' in buffer:
                        lines = buffer.split('\n')
                        # Process all complete lines except last
                        for line in lines[:-1]:
                            self._process_message(line.strip())
                        buffer = lines[-1]  # Keep last incomplete line
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Serial error: {e}")
                buffer = ""  # Reset buffer on error
                time.sleep(1)
    
    def _process_message(self, message: str):
        try:
            # Skip non-JSON messages (like status floats)
            if not (message.startswith('{') and message.endswith('}')):
                self.logger.debug(f"Skipping non-JSON message: {message}")
                return
                
            data = json.loads(message)
            self._handle_parsed_data(data)
        except json.JSONDecodeError:
            self.logger.warning(f"JSON decode error on message: {message}")
        except Exception as e:
            self.logger.error(f"Error processing message: {message}, error: {e}")

    def _handle_parsed_data(self, data: dict):
        # Add type guard for dictionary
        if not isinstance(data, dict):
            self.logger.warning(f"Unexpected data type: {type(data)}")
            return
            
        msg_type = data.get('type')
        
        if msg_type == 'key':
            try:
                # Extract timestamp safely
                timestamp = data.get('timestamp', 0)
                if isinstance(timestamp, (int, float)):
                    timestamp_sec = timestamp / 1000.0
                else:
                    timestamp_sec = 0.0
                    self.logger.warning(f"Invalid timestamp: {timestamp}")
                
                key_event = KeyEvent(
                    device_id=str(data.get('device', '')),
                    key_number=int(data.get('key_number', 0)),
                    section=str(data.get('section', '')),
                    contact=str(data.get('contact', '')),
                    action=str(data.get('action', '')),
                    velocity=64,  # Default velocity
                    timestamp=timestamp_sec
                )
                self.key_callback(key_event)
            except (TypeError, ValueError) as e:
                self.logger.error(f"Key event error: {e} - Data: {data}")
                
        elif msg_type == 'pedal':
            if 'pedal_name' in data and 'action' in data:
                pedal_event = PedalEvent(
                    device_id=data.get('device', ''),
                    pedal_number=data.get('pedal_number', 0),
                    pedal_name=data['pedal_name'],
                    action=data['action'],
                    timestamp=data.get('timestamp', 0) / 1000.0
                )
                self.pedal_callback(pedal_event)
            else:
                self.logger.warning("Malformed pedal event: missing fields")
                
        elif msg_type == 'volume' and self.volume_callback:
            # Process volume events ONLY from volume control device
            volume_event = VolumeEvent(
                device_id=data.get('device', ''),
                raw_value=data['raw_value'],
                volume_percent=self._raw_to_percent(data['raw_value']),
                muted=data.get('muted', False),
                timestamp=data.get('timestamp', 0) / 1000.0
            )
            self.volume_callback(volume_event)
                
        elif msg_type == 'status':
            self.logger.info(f"Device status: {data.get('status')} - {data.get('message')}")
                            
    def _raw_to_percent(self, raw_value) -> float:
        """Convert raw ADC value to percentage using config"""
        try:
            # Ensure raw_value is numeric
            raw_value = float(raw_value)
        except (TypeError, ValueError):
            return 0.0
            
        # Rest of the method remains the same...
        vol_cfg = self.config.get("volume", {})
        min_raw = float(vol_cfg.get("min_raw", 0))
        max_raw = float(vol_cfg.get("max_raw", 4095))
        curve_type = vol_cfg.get("curve", "linear")
        
        # Apply calibration range
        clamped = max(min_raw, min(raw_value, max_raw))
        scaled = (clamped - min_raw) / (max_raw - min_raw)
        
        # Apply curve type
        if curve_type == "logarithmic":
            # Logarithmic curve: more precision at lower volumes
            return 100 * math.log10(1 + 9 * scaled)
        else:  # linear
            return 100 * scaled


class EnhancedFluidSynthController:
    """Enhanced FluidSynth controller"""
    def __init__(self, soundfont_path: str = None):
        self.process: Optional[subprocess.Popen] = None
        self.soundfont_path = soundfont_path or self._find_soundfont()
        self.logger = logging.getLogger("FluidSynth")
        
    def _find_soundfont(self) -> str:
        common_paths = [
            "/home/admin/piano/sounds/UprightPianoKW-20220221.sf2",
            "~/piano/sounds/UprightPianoKW-20220221.sf2",
            "/usr/share/sounds/sf2/FluidR3_GM.sf2"
        ]
        
        for path in common_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path
        return ""
    
    def start(self) -> bool:
        try:
            cmd = [
                "fluidsynth", "-a", "alsa", "-g", "0.65", "-r", "48000",
                "-c", "2", "-z", "512", "-i",
                "-o", "synth.polyphony=256", "-o", "synth.cpu-cores=2",
                "-o", "audio.period-size=128", "-o", "audio.periods=3",
                "-o", "synth.reverb.active=true", "-o", "synth.reverb.room-size=0.6",
                "-o", "synth.reverb.damping=0.3", "-o", "synth.reverb.width=0.8",
                "-o", "synth.reverb.level=0.12", "-o", "synth.chorus.active=true",
                "-o", "synth.chorus.nr=3", "-o", "synth.chorus.level=0.03",
                "-o", "synth.chorus.speed=0.4", "-o", "synth.chorus.depth=1.5",
                "-o", "synth.sample-rate=48000", "-o", "synth.interpolation=7",
                "-o", "synth.dynamic-sample-loading=true"
            ]
            
            if self.soundfont_path:
                cmd.append(self.soundfont_path)
            
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(1.5)
            self.logger.info("Started enhanced FluidSynth")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start FluidSynth: {e}")
            return False
    
    def stop(self):
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.write("quit\n")
                    self.process.stdin.flush()
                    self.process.stdin.close()
                self.process.wait(timeout=3.0)
            except:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                except:
                    self.process.kill()


class RealisticPianoKeyMapper:
    """Key mapper"""
    def __init__(self):
        self.bass_offset = 21
        self.treble_offset = 21
    
    def get_midi_note(self, key_number: int, section: str) -> int:
        if section == "bass":
            return min(108, max(21, self.bass_offset + key_number))
        elif section == "treble":
            return min(108, max(21, self.bass_offset + key_number))
        else:
            return 60


class PianoExpressionManager:
    """Expression manager"""
    def __init__(self):
        pass


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutting down system...")
    if 'system' in globals():
        system.stop()
    sys.exit(0)


def main():
    """Main entry point for piano system"""
    global system
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    system = PianoSystem()
    
    if system.start():
        print("🎹 Piano MIDI System")
        print("")
        print("Press Ctrl+C to stop.")
        
        #load_config()
        
        try:
            while system.running:
                #load_config()              # pick up any edits
                #system.process_events()    # use new values immediately
    
                time.sleep(5)
                # Print comprehensive performance stats
                status = system.get_system_status()
                print(f"🎼 Status: {status['active_notes']} active, "
                      f"{status['held_notes']} held, "
                      f"{status['preparations']} preparations, "
                      f"{status['polyphonic_notes']} polyphonic interactions, "
                      f"Threads: {status['thread_count']}")
                
                # Environmental status
                env = status['environment']
                print(f"🌡️ Environment: {env['temperature']:.1f}°C, {env['humidity']:.1f}% RH")

        except KeyboardInterrupt:
            pass
    else:
        print("Failed to start system")
        return 1
    
    system.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())