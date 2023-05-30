import sys

from simulator.simulator import Simulator

# command line args
grid_id = int(sys.argv[1])
pol_type = "IM"

# define a simulator object
simulator = Simulator()
simulator.execute(0, grid_id, pol_type, None)
