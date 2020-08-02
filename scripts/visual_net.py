import numpy as np
import networkx as nx
import tkinter as tk
import json
import pickle as pk
from param_search.OptSearch import TopoOperator
from param_search.plotv import TopoSimulator

class Demo_gui(object):
    def __init__(self, root, networkParams):
        self.node_num = 8 # default
        self.networkParams = networkParams
        self.operator = TopoOperator(8)
        self.simulator = TopoSimulator()
        # GUI framework
        self.frame_main = tk.Frame(root,cursor='arrow',borderwidth=5)
        self.frame_main.pack()
        self.frame_l = tk.Frame(self.frame_main,borderwidth=5)
        self.frame_r = tk.Frame(self.frame_main,relief="groove",borderwidth=5)
        self.frame_l.pack(side='left')
        self.frame_r.pack(side='right')
        self.frame_l_t = tk.Frame(self.frame_l,borderwidth=5)
        self.frame_l_b = tk.Frame(self.frame_l,borderwidth=5)
        self.frame_l_t.pack(side='top')
        self.frame_l_b.pack(side='bottom')
        self.frame_l_t_l = tk.Frame(self.frame_l_t,borderwidth=5)
        self.frame_l_t_r = tk.Frame(self.frame_l_t,borderwidth=5)
        self.frame_l_t_l.pack(side='left')
        self.frame_l_t_r.pack(side='right')
        # input
        input_label = tk.Label(self.frame_l_t_l, text="Input dataset",font=('CMU Serif', 14),width=10, height=1)
        self.input_data = tk.Entry(self.frame_l_t_l,show=None,font=('CMU Typewriter Text', 14),width=10)
        button_demand = tk.Button(self.frame_l_t_r, bg="DimGrey",fg="white",text='show demand matrix', font=('CMU Serif Extra', 14), width=20, height=1, command=self.show_dm)
        button_topo = tk.Button(self.frame_l_t_r, bg="DarkGrey",fg="white",text='show graph', font=('CMU Serif Extra', 14), width=20, height=1, command=self.show_graph)
        param_input_label = tk.Label(self.frame_l_t_r, text="Input: policy parameters",font=('CMU Serif', 14),width=20, height=1)
        self.param_input_data_v = tk.Entry(self.frame_l_t_r,show=None,font=('CMU Typewriter Text', 14),width=10)
        self.param_input_data_i = tk.Entry(self.frame_l_t_r,show=None,font=('CMU Typewriter Text', 14),width=10)
        input_label.pack()
        self.input_data.pack()
        button_topo.pack()
        button_demand.pack()
        param_input_label.pack()
        self.param_input_data_v.pack()
        self.param_input_data_i.pack()

        # button
        self.button = tk.Button(self.frame_l_t_r, bg="Black",fg="white",text='act', font=('CMU Serif Extra', 14), width=20, height=1, command=self.invoke_policy)
        self.button.pack()

        # ouput
        self.demand_text = tk.Text(self.frame_l_t_l,bg="WhiteSmoke",font=('CMU Concrete', 10),width=40,height=10)
        self.text = tk.Text(self.frame_l_b,bg="WhiteSmoke",font=('CMU Concrete', 12),width=60)
        self.demand_text.pack()
        self.text.pack()

        # graph demonstration
        self.canvasWidth = 500
        self.canvasHeight = 500
        self.canvas = tk.Canvas(self.frame_r, width=self.canvasWidth, height=self.canvasHeight)
        self.canvas.pack()
        #self.canvas.grid(column=1, row=1, rowspan=4)
    
    def show_graph(self):
        self.text.delete(1.0, tk.END)
        self.canvas.delete("all")

        topo_var = self.input_data.get()
        if topo_var == "random8":
            self.node_num = 8
            file_topo = "../data/10000_8_4_topo_test.pk3"
            rand_num = np.random.randint(0,high=self.node_num,dtype=int)
            with open(file_topo, 'rb') as f2:
                self.topo = pk.load(f2)[rand_num]
            self.rectCenters = self.calcRectCenters(topo_var)
            self.lines = self.drawLines()
            self.rects = self.drawRectangles()
        elif topo_var == "nsfnet":
            self.node_num = 14
            file_topo = '../data/nsfnet/topology.pkl'
            with open(file_topo, 'rb') as f2:
                self.topo = pk.load(f2)
            self.rectCenters = self.calcRectCenters(topo_var)
            self.lines = self.drawLines()
            self.rects = self.drawRectangles()
        elif topo_var == "geant2":
            self.node_num = 24
            file_topo = '../data/geant2/topology.pkl'
            with open(file_topo, 'rb') as f2:
                self.topo = pk.load(f2)
            self.rectCenters = self.calcRectCenters(topo_var)
            self.lines = self.drawLines()
            self.rects = self.drawRectangles()
        else:
            self.text.insert(tk.END,"Invalid dataset option: {}\n".format(topo_var))
            self.text.update()

    def show_dm(self):
        self.text.delete(1.0, tk.END)
        self.demand_text.delete(1.0, tk.END)
        topo_var = self.input_data.get()
        if topo_var == "random8":
            file_demand_degree = '../data/10000_8_4_test.pk3'
            with open(file_demand_degree, 'rb') as f1:
                dataset = pk.load(f1)
            rand_num = np.random.randint(0,high=10000,dtype=int)
            self.demand = dataset[rand_num]["demand"]
            self.degree = dataset[rand_num]["allowed_degree"]
            self.demand_text.insert(tk.END, str(self.demand))
            self.demand_text.update()
        elif topo_var == "nsfnet":
            file_demand_degree = '../data/nsfnet/demand_100.pkl'
            with open(file_demand_degree, 'rb') as f1:
                dataset = pk.load(f1)
            rand_num = np.random.randint(0,high=100,dtype=int)
            self.demand = dataset[rand_num]
            self.degree = 4 * np.ones((self.node_num,), dtype=np.float32)
            self.demand_text.insert(tk.END, str(self.demand))
            self.demand_text.update()
        elif topo_var == "geant2":
            file_demand_degree = '../data/geant2/demand_100.pkl'
            with open(file_demand_degree, 'rb') as f1:
                dataset = pk.load(f1)
            rand_num = np.random.randint(0,high=100,dtype=int)
            self.demand = dataset[rand_num]
            self.degree = 8 * np.ones((self.node_num,), dtype=np.float32)
            self.demand_text.insert(tk.END, str(self.demand))
            self.demand_text.update()
        else:
            self.text.insert(tk.END,"Invalid topology option: {}\n".format(topo_var))
            self.text.update()

    def invoke_policy(self):
        param_v = float(self.param_input_data_v.get())
        param_i = float(self.param_input_data_i.get())
        self.simulator.reset(self.node_num)
        self.operator.reset(param_v, param_i, n_node=self.node_num)
        v = self.operator.predict(self.topo, self.demand)
        v_norm = self.normalization(v)
        self.text.insert(tk.END, "potential vector: {}\n".format(v_norm))
        act, rm_inds, new_topo = self.simulator.step_act(self.node_num, v, self.demand, self.topo, self.degree)
        if len(act) == 0:
            self.text.insert(tk.END,"No valid action.\n")
            self.text.update()
        else:
            for rm_ind in rm_inds:
                self.canvas.delete(self.lines[(rm_ind[0],rm_ind[1])])
            new_line = self.drawLine(act[0], act[1], active=True)
            act_tuple = (act[0], act[1]) if act[0]<act[1] else (act[1], act[0])
            self.lines[act_tuple] = new_line
        self.topo = new_topo

    def normalization(self, v):
        v_pos = v - min(v)
        if max(v) > 1e-7:
            return v/max(v)
        else:
            return (v+1e-7)/(max(v)+1e-7)

    def calcRectCenters(self, topo_var):
        """Compute the centers of the rectangles representing clients/routers"""
        rectCenters = {}
        gridSize = 5
        self.boxWidth = self.canvasWidth / gridSize
        self.boxHeight = self.canvasHeight / gridSize
        for i in range(self.node_num):
            gx,gy = self.networkParams[topo_var][str(i)]
            rectCenters[i] = (gx*self.boxWidth + self.boxWidth/2,
                                  gy*self.boxHeight + self.boxHeight/2)
        return rectCenters

    def drawLines(self):
        """draw lines corresponding to links"""
        lines = {}
        for i in range(self.node_num):
            for j in self.topo[i]:
                if j <= i:
                    continue
                line = self.drawLine(i, j)
                lines[(i,j)] = line
        return lines


    def drawLine(self, addr1, addr2, active=False):
        """draw a single line corresponding to one link"""
        linecolor = "orange" if active else "grey"
        center1, center2 = self.rectCenters[addr1], self.rectCenters[addr2]
        line = self.canvas.create_line(center1[0], center1[1], center2[0], center2[1],
                                       width=4, fill=linecolor)
        self.canvas.tag_lower(line)
        return line

    def drawRectangles(self):
        """draw rectangles corresponding to clients/routers"""
        rects = {}
        fill = "DodgerBlue2"
        for label in self.rectCenters:
            c = self.rectCenters[label]
            rect = self.canvas.create_rectangle(c[0]-self.boxWidth/6, c[1]-self.boxHeight/6,
                    c[0]+self.boxWidth/6, c[1]+self.boxHeight/6, fill=fill, activeoutline="green", activewidth=5)
            rects[label] = rect
            rectText = self.canvas.create_text(c[0], c[1], text=label, font=('consolas',18))
        return rects

def main():
    print("Demo")
    netCfgFilepath = "./locations.json"
    visualizeParams = json.load(open(netCfgFilepath))
    window = tk.Tk()
    window.geometry("1100x600")
    window.title("Topology adjustment demo")
    demo = Demo_gui(window, visualizeParams)
    window.mainloop()

if __name__ == "__main__":
    main()