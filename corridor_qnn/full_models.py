"""Full-level hybrid policy. Legacy checkpoints require explicit partial transfer."""
import torch
from torch import nn
from .models import ActorCritic
from .quantum import VariationalQuantumCircuit
from .full_environment import FEATURE_DIM, FEATURE_NAMES
from .full_actions import ACTION_NAMES

SCHEMA = 'full-doom-v1'


class FullActorCritic(ActorCritic):
    def __init__(self, kind='quantum', qubits=6, layers=2, reupload=False, residual=True, zz=False):
        super().__init__(kind,qubits,layers,reupload)
        self.reupload = reupload
        self.residual = residual
        self.zz = zz
        self.fusion[0] = nn.Linear(48+FEATURE_DIM,64)
        outputs = qubits*(qubits+1)//2 if zz else qubits
        if zz:
            if kind=='quantum': self.core = VariationalQuantumCircuit(qubits,layers,reupload,True)
            else: self.core = nn.Sequential(nn.Linear(qubits,32),nn.Tanh(),nn.Linear(32,outputs),nn.Tanh())
            self.critic = nn.Sequential(nn.Linear(48+outputs,32),nn.Tanh(),nn.Linear(32,1))
        self.actor = nn.Linear(outputs,len(ACTION_NAMES))
        self.classical_actor = nn.Linear(48,len(ACTION_NAMES))


def save_full(path, model, metadata):
    from pathlib import Path
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema=SCHEMA,kind=model.kind,qubits=model.qubits,layers=model.layers,reupload=getattr(model,'reupload',False),
                    residual=getattr(model,'residual',True),zz=getattr(model,'zz',False),
                    feature_names=list(FEATURE_NAMES),action_names=list(ACTION_NAMES),
                    state_dict=model.state_dict(),metadata=metadata),path)


def load_full(path, kind=None):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('schema')!=SCHEMA or data.get('feature_names')!=list(FEATURE_NAMES) or data.get('action_names')!=list(ACTION_NAMES):
        raise ValueError('Incompatible full-game checkpoint; corridor models need --transfer-corridor')
    if kind and data['kind']!=kind: raise ValueError('Checkpoint policy kind mismatch')
    model=FullActorCritic(data['kind'],data['qubits'],data['layers'],data.get('reupload',False),data.get('residual',True),data.get('zz',False))
    model.load_state_dict(data['state_dict'])
    return model.eval()


def transfer_corridor(model, path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('schema')==SCHEMA:
        raise ValueError('Transfer expects a legacy corridor checkpoint')
    state=model.state_dict()
    # Copy only semantically compatible modules; feature fusion and action heads differ.
    prefixes=('vision.','angles.','core.','fusion.2.','critic.','quantum_gain')
    copied=[]
    for name,value in data['state_dict'].items():
        if name.startswith(prefixes) and name in state and value.shape==state[name].shape:
            state[name]=value; copied.append(name)
    model.load_state_dict(state)
    return copied
