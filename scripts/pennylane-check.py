"""Re-run a trained full-game hybrid with its circuit executed by PennyLane instead of the torch simulator.

  python scripts/pennylane-check.py parity CHECKPOINT DATASET
  python scripts/pennylane-check.py eval   CHECKPOINT  <full-eval args...>
"""
import sys
import numpy as np
import pennylane as qml
import torch
from torch import nn
from corridor_qnn import full_cli
from corridor_qnn.full_models import load_full


class PennyLaneCircuit(nn.Module):
    """Same circuit as corridor_qnn.quantum.VariationalQuantumCircuit, as a PennyLane QNode."""

    def __init__(self, weights, input_scale=None):
        super().__init__()
        self.input_scale = None if input_scale is None else input_scale.detach().clone().double()
        self.weights = nn.Parameter(weights.detach().clone().double())
        self.qubits = weights.shape[1]
        device = qml.device('default.qubit', wires=self.qubits)

        @qml.qnode(device, interface='torch', diff_method='backprop')
        def circuit(angles, weights):
            if self.input_scale is None:
                for wire in range(self.qubits):
                    qml.RY(angles[..., wire], wires=wire)
            for layer in range(weights.shape[0]):
                for wire in range(self.qubits):
                    if self.input_scale is not None:
                        qml.RY(self.input_scale[layer, wire] * angles[..., wire], wires=wire)
                    qml.RY(weights[layer, wire, 0], wires=wire)
                    qml.RZ(weights[layer, wire, 1], wires=wire)
                for control in range(self.qubits):
                    qml.CNOT(wires=[control, (control + 1) % self.qubits])
            return [qml.expval(qml.PauliZ(w)) for w in range(self.qubits)]

        self.circuit = circuit

    def forward(self, angles):
        out = self.circuit(angles.double(), self.weights)
        return torch.stack(out, -1).reshape(angles.shape[0], self.qubits).float()


def to_pennylane(model):
    model.core = PennyLaneCircuit(model.core.weights, getattr(model.core, 'input_scale', None))
    return model


def parity(checkpoint, dataset):
    torch_model = load_full(checkpoint, 'quantum')
    pl_model = to_pennylane(load_full(checkpoint, 'quantum'))
    print(qml.draw(pl_model.core.circuit)(torch.zeros(1, pl_model.core.qubits), pl_model.core.weights))
    angles = (torch.rand(512, torch_model.qubits) * 2 - 1) * np.pi
    with torch.no_grad():
        diff = (torch_model.core(angles) - pl_model.core(angles)).abs().max()
    print(f'random angles (512): max |<Z> torch - pennylane| = {diff:.2e}')
    data = np.load(dataset, allow_pickle=False)
    idx = np.random.default_rng(0).choice(len(data['labels']), 2000, replace=False)
    x = torch.from_numpy(data['images'][idx]); f = torch.from_numpy(data['features'][idx])
    with torch.no_grad():
        lt, vt = torch_model(x, f); lp, vp = pl_model(x, f)
    print(f'game states (2000): max |logit diff| = {(lt - lp).abs().max():.2e}, '
          f'max |value diff| = {(vt - vp).abs().max():.2e}, '
          f'argmax action agreement = {(lt.argmax(-1) == lp.argmax(-1)).float().mean():.4%}')


if __name__ == '__main__':
    mode, checkpoint = sys.argv[1], sys.argv[2]
    if mode == 'parity':
        parity(checkpoint, sys.argv[3])
    else:
        # Swap the circuit inside the unmodified full-eval command.
        original = full_cli.load_full
        full_cli.load_full = lambda path, kind=None: to_pennylane(original(path, kind))
        from corridor_qnn.__main__ import main
        sys.argv = ['corridor_qnn', 'full-eval', '--checkpoint', checkpoint] + sys.argv[3:]
        main()
