from evtool.dvs import DvsFile
from evtool.utils import Player

# Load event data
data = DvsFile.load("/home/kuga/Workspace/cuke-emlb/results/mlpf/samples/demo/demo-01.pkl")
print(data['events'].shape)

# Load data into player and choose core
player = Player(data, core='matplotlib')

# View data
player.view("25ms", use_aps=False)
