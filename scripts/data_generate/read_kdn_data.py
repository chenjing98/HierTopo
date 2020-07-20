import pickle as pk
import numpy as np

topology = "geant2"
data_file = "../../data/" + topology + "/simulationResults.txt"
n_nodes = 14 if topology == "nsfnet" else 24

if topology == "nsfnet":
    graph_dict = {0:{1:{},2:{},3:{}}, 1:{0:{},2:{},7:{}}, 2:{0:{},1:{},5:{}}, 3:{0:{},4:{},8:{}},
                  4:{3:{},5:{},6:{}}, 5:{2:{},4:{},12:{},13:{}}, 6:{4:{},7:{}}, 7:{1:{},6:{},10:{}},
                  8:{3:{},9:{},11:{}}, 9:{8:{},10:{},12:{}}, 10:{7:{},9:{},11:{},13:{}},
                  11:{8:{},10:{},12:{}}, 12:{5:{},9:{},11:{}}, 13:{5:{},10:{}}}
elif topology == "geant2":
    graph_dict = {0:{1:{},2:{}}, 1:{0:{},3:{},6:{},9:{}}, 2:{0:{},3:{},4:{}}, 3:{1:{},2:{},5:{},6:{}},
                  4:{2:{},7:{}}, 5:{3:{},8:{}}, 6:{1:{},3:{},8:{},9:{}}, 7:{4:{},8:{},11:{}},
                  8:{5:{},6:{},7:{},11:{},12:{},17:{},18:{},20:{}}, 9:{1:{},6:{},10:{},12:{},13:{}},
                  10:{9:{},13:{}}, 11:{7:{},8:{},14:{},20:{}}, 12:{8:{},9:{},13:{},19:{},21:{}},
                  13:{9:{},10:{},12:{}}, 14:{11:{},15:{}}, 15:{14:{},16:{}}, 16:{15:{},17:{}},
                  17:{8:{},16:{},18:{}}, 18:{8:{},17:{},21:{}}, 19:{12:{},23:{}}, 20:{8:{},11:{}},
                  21:{12:{},18:{},22:{}}, 22:{21:{},23:{}}, 23:{19:{},22:{}}}
else:
    print('topology' + str(topology) + ' not recognized.')
    exit(1)

def safe_float(number):
    try:
        return float(number)
    except:
        return None

demands = []
pkts = []
output_demand_file = "../../data/" + topology + "/demand_{}.pkl".format(0)
output_packet_file = "../../data/" + topology + "/packet_{}.pkl".format(0)
output_topo_file = "../../data/" + topology + "/topology.pkl"
with open(output_topo_file, 'wb') as f0:
    pk.dump(graph_dict, f0)

lines = [line.rstrip('\n') for line in open(data_file)]
for i in range(len(lines)):
    val = lines[i].split(",")
    v = val[:(n_nodes * n_nodes * 3)]
    float_v = list(map(safe_float, v))
    np_v = np.array(float_v, dtype=np.float32).reshape(n_nodes*n_nodes, 3)
    demand = np_v[:,0].reshape((n_nodes, n_nodes))
    pkt = np_v[:,1].reshape((n_nodes, n_nodes))
    demands.append(demand)
    pkts.append(pkt)
    if (i+1) % 100 == 0 or i == (len(lines) - 1):
        output_demand_file = "../../data/" + topology + "/demand_{}.pkl".format(i+1)
        output_packet_file = "../../data/" + topology + "/packet_{}.pkl".format(i+1)
        with open(output_demand_file, 'wb') as f1:
            pk.dump(demands, f1)
        with open(output_packet_file, 'wb') as f2:
            pk.dump(pkts, f2)
        demands = []
        pkts = []
        print("{} samples loaded.".format(i+1))
