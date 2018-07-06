import torch

def label_from_output(output_tensor, all_labels):
    top_n, top_i = output.topk(1)
    label = top_i[0].item()
    return all_labels[label], label
