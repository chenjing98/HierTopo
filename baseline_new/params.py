import argparse

parser = argparse.ArgumentParser(description='baseline')

parser.add_argument('-n','--node', type = int, help = 'define the number of node')
parser.add_argument('-b','--baseline', type = str, help = 'choose which method to use', choices = ['egotree','match'])
parser.add_argument('-p','--penalty', type = int, help = 'degree penalty')
parser.add_argument('-d','--maxdemand', type = int, help = 'the highest demand')
parser.add_argument('-i','--iters', type = int)
parser.add_argument('-c','--count', type = int, help = 'the number of edge in matching', default = 0)
parser.add_argument('-f','--folder', type = str, default= './data/')
args = parser.parse_args()
