from copy import deepcopy
from time import sleep

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import pairwise_distances
from scipy import sparse
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from utils import cutoff_matrix

def get_filtered_mat_list(adj_matrix, thresholds_array, ntokens):
    """
    Converts adjancency matrix with real weights into list of binary matricies.
    For each threshold, those weights of adjancency matrix, which are less than
    threshold, get "filtered out" (set to 0), remained weights are set to ones.

    Args:
        adj_matrix (np.array[float, float])
        thresholds_array (iterable[float])
        n_tokens (int)

    Returns:
        filtered_matricies (list[np.array[int, int]])
    """
    filtered_matricies = []
    for thr in thresholds_array:
        filtered_matrix = adj_matrix.copy()
        filtered_matrix = cutoff_matrix(filtered_matrix, ntokens)
        filtered_matrix[filtered_matrix <  thr] = 0
        filtered_matrix[filtered_matrix >= thr] = 1
        filtered_matricies.append(filtered_matrix.astype(np.int8))
    return filtered_matricies

def adj_m_to_nx_list(adj_matrix, thresholds_array, ntokens, no_mat_output=False):
    """
    Converts adjancency matrix into list of unweighted digraphs, using filtering
    process from previous function.

    Args:
        adj_matrix (np.array[float, float])
        thresholds_array (iterable[float])
        n_tokens (int)

    Returns:
        nx_graphs_list (list[nx.MultiDiGraph])
        filt_mat_list(list[np.array[int, int]])

    """
#     adj_matrix = adj_matrix[:length,:length]
    filt_mat_list = get_filtered_mat_list(adj_matrix, thresholds_array, ntokens)
    nx_graphs_list = []
    for mat in filt_mat_list:
        nx_graphs_list.append(nx.from_numpy_matrix(np.array(mat), \
                              create_using=nx.MultiDiGraph()))
    if no_mat_output:
        return nx_graphs_list, []
    else:
        return nx_graphs_list, filt_mat_list

def adj_ms_to_nx_lists(adj_matricies, \
                       thresholds_array, \
                       ntokens_array, \
                       verbose=True, \
                       no_mat_output=False):
    """
    Executes adj_m_to_nx_list for each matrix in adj_matricies array, arranges
    the results. If verbose==True, shows progress bar.

    Args:
        adj_matrix (np.array[float, float])
        thresholds_array (iterable[float])
        verbose (bool)

    Returns:
        nx_graphs_list (list[nx.MultiDiGraph])
        filt_mat_lists (list[list[np.array[int,int]]])
    """
    graph_lists = []
    filt_mat_lists = []

    iterable = range(len(adj_matricies))
    if verbose:
        iterable = tqdm(range(len(adj_matricies)),
                        desc="Calc graphs list")
    for i in iterable:
        g_list, filt_mat_list = adj_m_to_nx_list(adj_matricies[i],\
                                                 thresholds_array,\
                                                 ntokens_array[i], \
                                                 no_mat_output=no_mat_output)
        graph_lists.append(g_list)
        filt_mat_lists.append(filt_mat_lists)
    
    return graph_lists, filt_mat_lists

def count_stat(g_listt_j, function=nx.weakly_connected_components, cap=500):
    """
    Calculates stat (topological feature), using the function, which returns a
    generator (for example, generator of simple cycles in the DiGraph).

    Args:
        g_listt_j (list[nx.MultiDiGraph])
        function (function)
        cap (int)

    Returns:
        stat_amount (int)
    """
    stat_amount = 0
    for _ in function(g_listt_j):
        stat_amount += 1
        if stat_amount >= cap:
            break
    return stat_amount 

def count_weak_components(g_listt_j, cap=500):
    return count_stat(g_listt_j, 
                      function=nx.weakly_connected_components, 
                      cap=cap)

def count_strong_components(g_listt_j, cap=500):
    return count_stat(g_listt_j, 
                      function=nx.strongly_connected_components, 
                      cap=cap)

def count_simple_cycles(g_listt_j, cap=500):
    return count_stat(g_listt_j, function=nx.simple_cycles, cap=cap)

def dim_connected_components(graph_lists, strong=False, verbose=False, cap=500):
    """
    Calculates amount of connected components for each graph in list
    of lists of digraphs. If strong==True, calculates strongly connected
    components, otherwise calculates weakly connected components.
    If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        strong (bool)
        verbose (bool)

    Returns:
        w_lists (list[list[int])
    """
    w_lists = [] # len == len(w_graph_lists)
    iterable = range(len(graph_lists))
    if verbose:
        iterable = tqdm(range(len(graph_lists)),
                        desc="Calc weak comp")
    for i in iterable:
        g_list = graph_lists[i]
        w_cmp  = []
        for j in range(len(g_list)):
            if strong:
                w_cmp.append(count_strong_components(g_list[j], cap=cap))
            else:
                w_cmp.append(count_weak_components(g_list[j], cap=cap))
        w_lists.append(w_cmp)
    return w_lists

def dim_simple_cycles(graph_lists, verbose, cap=500):
    """
    Calculates amount of simple cycles for each graph in list
    of lists of digraphs. If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        c_lists (list[list[int])
    """
    c_lists = [] # len == len(pos_w_graph_lists)
    iterable = range(len(graph_lists))
    if verbose:
        iterable = tqdm(range(len(graph_lists)),
                        desc="Calc cycles")
    for i in iterable:
        g_list = graph_lists[i]
        c  = []
        for j in range (len(g_list)):
            c.append(count_simple_cycles(g_list[j], cap=cap))
        c_lists.append(c)
    return c_lists

def b0_b1(graph_lists, verbose):
    """
    Calculates first two Betti numbers for each graph in list of lists of 
    digraphs. If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        b0_lists (list[list[int])
        b1_lists (list[list[int])
    """
    b0_lists = []
    b1_lists = [] # len == len(pos_w_graph_lists)
    iterable = range(len(graph_lists))
    if verbose:
        iterable = tqdm(range(len(graph_lists)),
                        desc="Calc b0, b1")
    for i in iterable:
        g_list = graph_lists[i]
        b0 = []
        b1 = []
        for j in range (len(g_list)):
            g = nx.Graph(g_list[j].to_undirected()) 
            w = nx.number_connected_components(g)
            e = g.number_of_edges() 
            v = g.number_of_nodes()
            b0.append(w)
            b1.append(e - v + w)
            #print(b1)
        b0_lists.append(b0)
        b1_lists.append(b1)
    return b0_lists, b1_lists

def edges_f(graph_lists, verbose):
    """
    Calculates amount of edges for each graph in list
    of lists of digraphs. If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        e_lists (list[list[int])
    """
    e_lists = [] # len == len(pos_w_graph_lists)
    iterable = range(len(graph_lists))
    if verbose:
        iterable = tqdm(range(len(graph_lists)),
                        desc="Calc edges amount")
    for i in iterable:
        g_list = graph_lists[i]
        e  = []
        for j in range (len(g_list)):
            e.append(g_list[j].number_of_edges())
        e_lists.append(e)
    return e_lists

def v_degree_f(graph_lists, verbose):
    """
    Calculates amount of edges for each graph in list
    of lists of digraphs. If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        v_lists (list[list[int])
    """
    v_lists = [] # len == len(pos_w_graph_lists)
    iterable = range(len(graph_lists))
    if verbose:
        iterable = tqdm(range(len(graph_lists)),
                        desc="Calc average vertex degree")
    for i in iterable:
        g_list = graph_lists[i]
        v  = []
        for j in range (len(g_list)):
            #print(g_list[j])
            degrees = g_list[j].degree()
            degree_values = [v for k, v in degrees]
            sum_of_edges = sum(degree_values) / float(len(degree_values))
            v.append(sum_of_edges)
        v_lists.append(v)
    return v_lists

def H_1_statistics_by_thresholds(thresholds_array, \
                                 c_lists, \
                                 lang_list, \
                                 layer, \
                                 head):
    """
    Shows statistics of topological invariants from c_lists for each language
    from multi-language corpora, on the plot.

    Args:
        thresholds_array (iterable[float])
        c_lists (list[list[int]])
        lang_list (list[str])
        layer (int)
        head (int)

    Returns:
        None
    """

    T = len(thresholds_array)
    L = len(lang_list)
    fig, axs = plt.subplots(T, L, figsize=(16,16))

    colors = ['b', 'c', 'g', 'y', 'r', 'm']

    for l in range(L):
        current_color = colors[l % len(colors)]
        # Each language is assigned a color from colors list.
        max_amount = 0
        for t in range(T):
            current_data = np.array(c_lists[(layer, head)][l])
            axs[t, l].bar(current_data[l][t], \
                          color=current_color)
            axs[t, l].set_title(lang_list[k])
            max_amount = np.max(max_amount, current_data[l][t])
        for t in range(T):
            axs[t, l].set_ylim([0, max_amount * 1.1])

    plt.show()


def count_top_stats(adj_matricies, 
                    thresholds_array, 
                    ntokens_array, 
                    stats_to_count={"s", "e", "c", "v", "b0b1"}, 
                    stats_cap=500, 
                    sleep_time=0, 
                    verbose=False):
    """
    The main function for calculating topological invariants. Unites the 
    functional of all functions above.

    Args:
        adj_matricies (np.array[float, float, float, float, float])
        thresholds_array (list[float])
        stats_to_count (str)
        function_for_v (function)
        stats_cap (int)
        verbose (bool)

    Returns:
        stats_tuple_lists_array (np.array[float, float, float, float, float])
    """
    stats_tuple_lists_array = []

    for layer_of_interest in tqdm(range(adj_matricies.shape[1])):
        stats_tuple_lists_array.append([])
        for head_of_interest in range(adj_matricies.shape[2]):
            sleep(sleep_time)
            adj_ms = adj_matricies[:,layer_of_interest,head_of_interest,:,:]
            g_lists, _ = adj_ms_to_nx_lists(adj_ms,
                                            thresholds_array=thresholds_array,
                                            ntokens_array=ntokens_array,
                                            verbose=False)
            feat_lists = []
            if "s" in stats_to_count:
                feat_lists.append(dim_connected_components(g_lists,
                                                           strong=True,
                                                           verbose=False,
                                                           cap=stats_cap))
            if "w" in stats_to_count:
                feat_lists.append(dim_connected_components(g_lists,
                                                           strong=False, 
                                                           verbose=False, 
                                                           cap=stats_cap))
            if "e" in stats_to_count:
                feat_lists.append(edges_f(g_lists, verbose=False))
            if "v" in stats_to_count:
                feat_lists.append(v_degree_f(g_lists, verbose=False))
            if "c" in stats_to_count:
                feat_lists.append(dim_simple_cycles(g_lists,
                                                    verbose=False, 
                                                    cap=stats_cap))
            if "b0b1" in stats_to_count:
                b0_lists, b1_lists = b0_b1(g_lists, verbose=False) 
                feat_lists.append(b0_lists)
                feat_lists.append(b1_lists)
            stats_tuple_lists_array[-1].append(tuple(feat_lists))
            
    stats_tuple_lists_array = np.asarray(stats_tuple_lists_array, 
                                         dtype=np.float16)
    return stats_tuple_lists_array
