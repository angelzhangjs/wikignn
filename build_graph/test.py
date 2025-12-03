from data_utils import load_graph

# data = torch.load('data/clean_graph.pyg_en_labeled.pt')
# input_dim = data.x.size(1)
# output_dim = (data.y.max().item() + 1)

if __name__ == '__main__':
    data = load_graph('graph_output/clean_graph.pyg_en_labeled.pt')
    input_dim = data.x.size(1)
    output_dim = (data.y.max().item() + 1)
    print(input_dim, output_dim)