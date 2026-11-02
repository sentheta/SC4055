"""Student without privileged inputs: only what a player can see, feel or remember.

The teacher keeps its map-reading planner and labels every decision, but nothing derived
from the planner, the engine's object labels or its depth buffer reaches this model.
"""
import math
from collections import deque
import numpy as np
import torch
from PIL import Image
from torch import nn
from .full_actions import ACTIONS
from .full_environment import BASE_FEATURES
from .quantum import VariationalQuantumCircuit

SCHEMA = 'player-v1'
KEY_COLORS = ('blue', 'yellow', 'red')
# Read from the environment's feature vector: HUD values and the player's own motion.
HUD = ('health', 'ammo', 'armor', 'weapon', 'kills', 'items', 'yaw_cos', 'yaw_sin', 'speed', 'stuck',
       'coverage', 'time', 'damage', 'dx', 'dy', 'has_shotgun', 'has_chaingun')
HUD_INDEX = [BASE_FEATURES.index(n) for n in HUD]
PRIVILEGED = ('waypoint_cos', 'waypoint_sin', 'waypoint_distance', 'interaction_distance',
              'enemy_cos', 'enemy_sin', 'enemy_distance', 'enemy_visible',
              'depth_left', 'depth_midleft', 'depth_front', 'depth_midright', 'depth_right',
              'previous_waypoint_sin', 'previous_enemy_sin', 'previous_enemy_visible')
assert not set(HUD) & set(PRIVILEGED) and set(HUD) | set(PRIVILEGED) == set(BASE_FEATURES)
MEMORY = ('keys_blue', 'keys_yellow', 'keys_red', 'pos_x', 'pos_y', 'moved8_x', 'moved8_y',
          'moved32_x', 'moved32_y', 'uses', 'revisits')
PLAYER_NAMES = HUD + MEMORY + tuple(f'last_{i}' for i in range(len(ACTIONS)))
PLAYER_DIM = len(PLAYER_NAMES)
POS = PLAYER_NAMES.index('pos_x')
# Training-only targets for an auxiliary head; they are never inputs, so play stays player-view.
AUX_INDEX = [BASE_FEATURES.index(n) for n in PRIVILEGED[:13]]
FRAMES = 3


class PlayerView:
    """Turns an environment observation into the student's input, keeping episode memory."""

    def __init__(self, env, rgb=False):
        self.env, self.rgb = env, rgb

    def frame(self, obs):
        if not self.rgb:
            return obs.pixels
        return np.asarray(Image.fromarray(obs.frame).resize((96, 64), Image.BILINEAR))

    def reset(self):
        self.hidden = None; self.pos = np.zeros(2); self.trail = deque(maxlen=33); self.uses = 0
        self.cells = {}; self.frames = deque(maxlen=FRAMES)

    def update(self, obs):
        f = obs.features; n = len(BASE_FEATURES)
        # Dead reckoning from the player's own step-by-step motion, starting at the origin.
        self.pos += f[BASE_FEATURES.index('dx')]*32, f[BASE_FEATURES.index('dy')]*32
        self.trail.append(self.pos.copy())
        moved = lambda k: (self.pos-self.trail[max(0, len(self.trail)-1-k)])/512
        last = f[n:]
        if last[12] > .5 or last[13] > .5: self.uses += 1
        cell = tuple(np.floor(self.pos/64).astype(int))
        self.cells[cell] = self.cells.get(cell, 0)+1
        held = self.env.navigator.held  # the keys shown on the HUD
        vector = [*f[HUD_INDEX], *(float(c in held) for c in KEY_COLORS), *(self.pos/4096), *moved(8), *moved(32),
                  self.uses/32, min(self.cells[cell]/32, 4), *last]
        self.current = self.frame(obs); self.frames.append(self.current)
        return np.asarray(vector, dtype=np.float32)

    def stack(self):
        frames = list(self.frames)
        return np.stack([frames[0]]*(FRAMES-len(frames))+frames)


class PlayerModel(nn.Module):
    """Pixels and player features -> CNN and MLP -> 6 angles -> circuit -> action logits."""

    def __init__(self, kind='quantum', qubits=6, layers=2, hidden=256, bands=6, memory=False, aux=False, rgb=False):
        super().__init__()
        self.kind, self.qubits, self.layers, self.hidden, self.bands, self.rgb = kind, qubits, layers, hidden, bands, rgb
        hidden_vision = 64
        self.vision = nn.Sequential(
            nn.Conv2d(FRAMES*3 if rgb else FRAMES, 16, 5, stride=3), nn.ReLU(), nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(), nn.Flatten(), nn.Linear(32*4*7, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(64+PLAYER_DIM+4*bands, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 48), nn.Tanh())
        self.angles = nn.Sequential(nn.Linear(48, qubits), nn.Tanh())
        if kind == 'quantum':
            self.core = VariationalQuantumCircuit(qubits, layers)
        else:
            self.core = nn.Sequential(nn.Linear(qubits, 32), nn.Tanh(), nn.Linear(32, qubits), nn.Tanh())
        self.actor = nn.Linear(qubits, len(ACTIONS))
        self.classical_actor = nn.Linear(48, len(ACTIONS))
        # Memory: a GRU over the fused features, added residually so it can extend a memoryless model.
        self.gru = nn.GRU(48, 48, batch_first=True) if memory else None
        self.quantum_gain = nn.Parameter(torch.tensor(1.0))
        # RGB models predict what is visible from vision and the route (4 targets) from the fused map features.
        self.aux_head = nn.Linear(hidden_vision, len(AUX_INDEX)-4 if rgb else len(AUX_INDEX)) if aux else None
        self.aux_route = nn.Linear(48, 4) if aux and rgb else None

    def forward(self, pixels, features, return_aux=False, hidden=None, seq=1):
        # Fourier features let the network carve the map into small regions from (x, y).
        if pixels.dim() == 5:
            pixels = pixels.permute(0, 1, 4, 2, 3).flatten(1, 2)
        xy = features[:, POS:POS+2]*4096
        scales = 2.0**torch.arange(self.bands, device=xy.device)*(2*math.pi/4096)
        phase = (xy.unsqueeze(-1)*scales).flatten(1)
        seen = self.vision(pixels.float()/255)
        fused = self.fusion(torch.cat([seen, features, phase.sin(), phase.cos()], -1))
        if self.gru is not None:
            # Inputs are flattened (batch*seq, ...); seq is the sequence length.
            out, self.last_hidden = self.gru(fused.view(-1, seq, 48), hidden)
            fused = fused+out.reshape(-1, 48)
        a = self.angles(fused)
        measured = self.core(a*math.pi) if self.kind == 'quantum' else self.core(a)
        logits = self.classical_actor(fused)+self.quantum_gain*self.actor(measured)
        if not return_aux:
            return logits
        return logits, torch.cat([self.aux_route(fused), self.aux_head(seen)], -1) if self.rgb else self.aux_head(seen)


def save_player(path, model, metadata):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=SCHEMA, kind=model.kind, qubits=model.qubits, layers=model.layers, hidden=model.hidden,
                    bands=model.bands, aux=model.aux_head is not None, rgb=model.rgb, memory=model.gru is not None, names=list(PLAYER_NAMES), state_dict=model.state_dict(), metadata=metadata), path)


def load_player(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data['schema'] != SCHEMA or data['names'] != list(PLAYER_NAMES):
        raise ValueError('Incompatible player checkpoint')
    model = PlayerModel(data['kind'], data['qubits'], data['layers'], data['hidden'], data['bands'], aux=data.get('aux', False), rgb=data.get('rgb', False), memory=data.get('memory', False))
    model.load_state_dict(data['state_dict'])
    return model.eval()


def stack_indices(episode_ids, idx):
    """Indices of the frames before idx, staying inside each episode."""
    out = [idx]
    for k in range(1, FRAMES):
        prev = np.maximum(idx-k, 0)
        out.insert(0, np.where(episode_ids[prev] == episode_ids[idx], prev, out[0]))
    return np.stack(out, 1)


@torch.no_grad()
def act(model, view, temperature=0.):
    pixels = torch.from_numpy(view.stack()).unsqueeze(0)
    features = torch.from_numpy(view.vector).unsqueeze(0)
    logits = model(pixels, features, hidden=view.hidden)
    if model.gru is not None:
        view.hidden = model.last_hidden
    # Sampling lets a stuck student leave a loop that a deterministic argmax would repeat forever.
    return int(torch.multinomial((logits/temperature).softmax(-1), 1)) if temperature > 0 else int(logits.argmax(-1))
