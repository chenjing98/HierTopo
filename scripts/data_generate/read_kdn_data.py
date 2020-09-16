import pickle as pk
import numpy as np

topology = "germany"
data_file = "../../data/" + topology + "/simulationResults.txt"

if topology == "nsfnet":
    n_nodes = 14
    graph_dict = {0:{1:{},2:{},3:{}}, 1:{0:{},2:{},7:{}}, 2:{0:{},1:{},5:{}}, 3:{0:{},4:{},8:{}},
                  4:{3:{},5:{},6:{}}, 5:{2:{},4:{},12:{},13:{}}, 6:{4:{},7:{}}, 7:{1:{},6:{},10:{}},
                  8:{3:{},9:{},11:{}}, 9:{8:{},10:{},12:{}}, 10:{7:{},9:{},11:{},13:{}},
                  11:{8:{},10:{},12:{}}, 12:{5:{},9:{},11:{}}, 13:{5:{},10:{}}}
elif topology == "geant2":
    n_nodes = 24
    graph_dict = {0:{1:{},2:{}}, 1:{0:{},3:{},6:{},9:{}}, 2:{0:{},3:{},4:{}}, 3:{1:{},2:{},5:{},6:{}},
                  4:{2:{},7:{}}, 5:{3:{},8:{}}, 6:{1:{},3:{},8:{},9:{}}, 7:{4:{},8:{},11:{}},
                  8:{5:{},6:{},7:{},11:{},12:{},17:{},18:{},20:{}}, 9:{1:{},6:{},10:{},12:{},13:{}},
                  10:{9:{},13:{}}, 11:{7:{},8:{},14:{},20:{}}, 12:{8:{},9:{},13:{},19:{},21:{}},
                  13:{9:{},10:{},12:{}}, 14:{11:{},15:{}}, 15:{14:{},16:{}}, 16:{15:{},17:{}},
                  17:{8:{},16:{},18:{}}, 18:{8:{},17:{},21:{}}, 19:{12:{},23:{}}, 20:{8:{},11:{}},
                  21:{12:{},18:{},22:{}}, 22:{21:{},23:{}}, 23:{19:{},22:{}}}
elif topology == "germany":
    n_nodes = 50
    graph_dict = {
        0:{29:{},46:{},48:{}}, 1:{34:{},47:{},49:{}}, 2:{8:{},31:{},37:{}}, 3:{11:{},20:{},31:{},32:{},43:{}},
        4:{5:{},22:{},35:{},44:{}}, 5:{4:{},21:{},22:{},25:{},32:{}}, 6:{7:{},22:{},38:{}}, 7:{6:{},15:{}},
        8:{2:{},11:{},13:{}}, 9:{16:{},23:{},33:{}}, 10:{14:{},25:{},35:{},44:{}}, 11:{3:{},8:{},13:{},31:{}},
        12:{14:{},29:{}}, 13:{8:{},25:{},31:{},49:{}}, 14:{10:{},12:{},48:{}}, 15:{7:{},27:{}},
        16:{9:{},18:{},19:{},28:{}}, 17:{24:{},30:{}}, 18:{16:{},19:{},25:{},49:{}}, 19:{16:{},18:{},25:{},44:{}},
        20:{3:{},43:{}}, 21:{5:{},22:{},27:{},43:{}}, 22:{4:{},5:{},6:{},39:{}}, 23:{9:{},24:{},28:{},42:{}},
        24:{17:{},23:{},33:{},42:{},45:{}}, 25:{5:{},10:{},13:{},18:{},19:{}}, 26:{30:{},34:{}}, 27:{15:{}, 21:{}, 43:{}},
        28:{16:{},23:{},29:{},44:{},46:{}}, 29:{0:{},12:{},28:{}}, 30:{17:{},26:{},45:{}}, 31:{2:{},3:{},11:{},13:{},32:{}},
        32:{3:{},5:{},31:{},43:{}}, 33:{9:{},24:{}}, 34:{1:{},26:{},37:{},40:{},41:{}}, 35:{10:{},39:{}},
        36:{38:{},48:{}}, 37:{2:{},34:{},41:{},49:{}}, 38:{6:{},36:{},39:{},48:{}}, 39:{22:{},35:{},38:{}},
        40:{34:{},41:{}}, 41:{34:{},37:{},40:{}}, 42:{23:{},24:{},46:{}}, 43:{3:{},20:{},21:{},27:{},32:{}},
        44:{4:{},10:{},19:{},28:{}}, 45:{24:{},30:{},47:{},49:{}}, 46:{0:{},28:{},42:{}}, 47:{1:{},45:{}}, 
        48:{0:{},14:{},36:{},38:{}}, 49:{1:{},13:{},18:{},37:{},45:{}}
        }
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
