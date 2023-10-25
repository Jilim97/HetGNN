from NetworkAnalysis.Graph import Graph, from_df, positives_split, calculateZscores, getDriverScore, FITTERMAPPER, \
    filter_graph_by_LP_swaps, filter_graph_by_LP_swaps_fastest
# from NetworkAnalysis.DirectedInteractionNetwork import DirectedInteractionNetwork
from NetworkAnalysis.PyLouvain import PyLouvain, in_order

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, confusion_matrix

from networkx import to_dict_of_lists
from scipy.linalg import expm
from scipy.optimize import nnls
from itertools import product

import matplotlib.pyplot as plt
import numpy.matlib as matlib
import pandas as pd
import numpy as np
import networkx as nx
import scipy
import warnings
import ast
import random
import time


class UndirectedInteractionNetwork(Graph):
    """
    Class tp represent an undirected network and inherits from the base class Graph.
    """
    def __init__(self, interaction_df, colnames=None, verbose=True, keeplargestcomponent=False,
                 allow_self_connected=False, node_types=None, gene2int=None):
        """

        Args:
            interaction_df (pd.DataFrame): Dataframe containing the edge list to be converted into an UndirectInteractionNetwork object
            colnames (list of str, optional): Which columns of interaction_df to proceed with. Defaults to None.
            verbose (bool, optional): Controls verbosity. Defaults to True.
            keeplargestcomponent (bool, optional): Whether to exclude (=True) or include (=False) disconnected components. Defaults to False.
            allow_self_connected (bool, optional): Allow self loops or not. Defaults to False.
            node_types (dict, optional): Dictionary where keys=node types and values=list of nodes that belong to a specific node type. Defaults to None.
            gene2int (dict, optional): A predefined dict that maps each gene ID to an integer for internal usage. Defaults to None.
        """
        super().__init__(interaction_df, colnames, verbose=False, keeplargestcomponent=keeplargestcomponent,
                         allow_self_connected=allow_self_connected, gene2int=gene2int, node_types=node_types)

        self.interactions.values.sort(axis=1)
        self.interactions = self.interactions.drop_duplicates(['Gene_A', 'Gene_B'])

        if verbose:
            print('%d Nodes and %d interactions' % (len(self.nodes), self.interactions.shape[0]))

        self.clf1 = None

    @property
    def isConnected(self):
        G = self.getnxGraph()
        return nx.is_connected(G)

    def interactions_as_set(self, return_names=True):
        if return_names:
            df = self.getInteractionNamed()
        else:
            df = self.interactions

        df = [tuple(sorted(t)) for t in zip(df.Gene_A.values, df.Gene_B.values)]

        return set(df)

    @classmethod
    def createFullyConnectedNetwork(cls, node_names):
        df = pd.DataFrame(np.array([(n1, n2) for i, n1 in enumerate(node_names)
                                    for n2 in node_names[:i]]),
                          columns=['gene_A', 'Gene_B'])

        return cls(df)

    def get_node_type_edge_counts(self):
        if self.node_types is not None:
            df = self.interactions[["Gene_A", 'Gene_B']].applymap(lambda x: self.node_types[x])
            df.values.sort(axis=1)
            return df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'counts'})

        else:
            warnings.warn("The node types are not defined, please provide these first.")
            return None

    def makeSelfConnected(self, inplace=False):
        if not inplace:
            return UndirectedInteractionNetwork(super(UndirectedInteractionNetwork, self).
                                                makeSelfConnected(inplace=False), colnames=('Gene_A', 'Gene_B'),
                                                allow_self_connected=True,
                                                node_types=self.node_type_names,
                                                gene2int=self.gene2int)

    def mergedf(self, interaction_df, node_types=None, colnames=None):
        return self.mergeNetworks(UndirectedInteractionNetwork(interaction_df, colnames=colnames,
                                                               verbose=False, node_types=node_types))

    def subsetNetwork(self, nodes, inplace=False, verbose=True, keeplargestcomponent=True, and_or='and'):
        if inplace:
            self.setEqual(UndirectedInteractionNetwork(super().subsetNetwork(nodes, and_or=and_or),
                                                       node_types=self.node_type_names,
                                                       keeplargestcomponent=keeplargestcomponent))
        else:
            return UndirectedInteractionNetwork(super().subsetNetwork(nodes, and_or=and_or), node_types=self.node_type_names,
                                                verbose=verbose, allow_self_connected=True,
                                                keeplargestcomponent=keeplargestcomponent)

    def mergeNetworks(self, network):
        new_dict = {**self.node_type_names, **network.node_type_names}
        return UndirectedInteractionNetwork(pd.concat([self.getInteractionNamed(), network.getInteractionNamed()],
                                                    axis=0, ignore_index=True),
                                            node_types=new_dict)

    def subsample(self, n=100, weighted=False):
        return super().subsample(n=n,
                                 weighted=weighted)  # UndirectedInteractionNetwork(super().subsample(n=n, weighted=weighted))

    def checkInteractions_df(self, df, colnames=('Gene_A', 'Gene_B')):
        '''
            Checks which interactions from a given df can be found in the interaction network
        '''
        df.values.sort(axis=1)
        named_net_df = self.getInteractionNamed()
        named_net_df.values.sort(axis=1)
        tester_pairs = set(zip(named_net_df.Gene_A, named_net_df.Gene_B))
        df['In Network'] = [pair in tester_pairs for pair in zip(df[colnames[0]], df[colnames[1]])]
        return df

    def removeNodes(self, nodes_tbr, inplace=False):
        if inplace:
            super().removeNodes(nodes_tbr, inplace=inplace)
            self.remove_nodes_from_type_dict(nodes_tbr)
        else:
            return UndirectedInteractionNetwork(super().removeNodes(nodes_tbr, inplace=False),
                                                node_types=self.node_type_names)

    def getAdjMatrix(self, sort='first', as_df=False):
        A, nodes = super().getAdjMatrix(sort=sort)

        if as_df:
            return pd.DataFrame(np.maximum(A, np.transpose(A)), columns=nodes, index=nodes)
        else:
            return np.maximum(A, np.transpose(A)), nodes

    def getComponents(self, return_subgraphs=False, verbose=False):
        '''
        Function that returns the connected components of a graph
        :param return_subgraphs: whether the subgraphs need to be returned as a list of UndirectedInteractionNetwork instances
        or as a pandas DF with for each node a label.
        :param verbose:
        :return: Either a pandas DF with the node labels denoting the different components or a list of
        UndirectedNetwork instances
        '''

        node_names = self.node_names
        if self.isConnected:

            if return_subgraphs:
                return [self.deepcopy()]

            else:
                return pd.DataFrame({'Gene': node_names, 'Component': [0 for _ in node_names]})

        else:
            components = nx.connected_components(self.getnxGraph())

            if not return_subgraphs:
                map_dict = {node: i for i, subgraph_nodes in enumerate(components) for node in subgraph_nodes}
                return pd.DataFrame({'Gene': node_names, 'Component': [map_dict[x] for x in node_names]})

            else:
                return [self.subsetNetwork(subgraph, verbose=verbose) for subgraph in components]

    def diffusePerComponent(self, kernel='LEX', as_df=True, scale=False, alpha=0.01,
                            self_connected=True, symmetric_norm=False, verbose=False):

        if self.isConnected:
            return super(UndirectedInteractionNetwork, self).diffuse(kernel=kernel, as_df=as_df, scale=scale,
                                                                     alpha=alpha,
                                                                     self_connected=self_connected,
                                                                     symmetric_norm=symmetric_norm)

        else:
            components = self.getComponents(return_subgraphs=True, verbose=verbose)
            As, node_names = [], []

            for comp in components:
                A, node_names_ = comp.diffuse(kernel=kernel, as_df=False, scale=scale, alpha=alpha,
                                              self_connected=self_connected, symmetric_norm=symmetric_norm)
                As += [A]
                node_names += list(node_names_)
                # K.loc[node_names, node_names] = A

            K = pd.DataFrame(scipy.linalg.block_diag(*As), columns=node_names, index=node_names)

            if as_df:
                node_names = self.node_names
                return K[node_names].loc[node_names]

            else:
                return K.values, K.columns.values

    def getComponents(self, return_subgraphs=False, verbose=False):
        '''
        Function that returns the connected components of a graph
        :param return_subgraphs: whether the subgraphs need to be returned as a list of UndirectedInteractionNetwork instances
        or as a pandas DF with for each node a label.
        :param verbose:
        :return: Either a pandas DF with the node labels denoting the different components or a list of
        UndirectedNetwork instances
        '''

        node_names = self.node_names
        if self.isConnected:

            if return_subgraphs:
                return [self.deepcopy()]

            else:
                return pd.DataFrame({'Gene': node_names, 'Component': [0 for _ in node_names]})

        else:
            components = nx.connected_components(self.getnxGraph())

            if not return_subgraphs:
                map_dict = {node: i for i, subgraph_nodes in enumerate(components) for node in subgraph_nodes}
                return pd.DataFrame({'Gene': node_names, 'Component': [map_dict[x] for x in node_names]})

            else:
                return [self.subsetNetwork(subgraph, verbose=verbose, keeplargestcomponent=False) for subgraph in components]

    def keepLargestComponent(self, verbose=True, inplace=False):

        if not self.isConnected:
            comps = self.getComponents(return_subgraphs=True)
            largestcomp = max(comps, key=len)

            if verbose:
                print('%i genes from smaller components have been removed.' % (self.N_nodes - largestcomp.N_nodes))

            if inplace:
                self.__init__(largestcomp.getInteractionNamed(), node_types=largestcomp.node_types, gene2int=largestcomp.gene2int)
            else:
                return largestcomp.deepcopy()
        else:
            print('Object is a fully connected graph, returning object copy.')
            if not inplace:
                return self.deepcopy()

    def findcommunities(self, verbose=True, as_df=False):

        '''
        Searches for communities using the PyLouvain algortihm which clusters based on modularity.
        :param verbose:
        :param as_df: whether the communities need to be returned as a pd DF containing the labels for each nodes,
        or a list of UndirectedInteractionNetwork instances.
        :return:
        '''

        comms = self.getComponents(return_subgraphs=True)
        communities = []

        for net in comms:
            df = net.getInteractionNamed()
            nodes = {node: 1 for node in net.node_names}
            edges = [(pair, 1) for pair in zip(df.Gene_A, df.Gene_B)]
            nodes_, edges_, map_dict = in_order(nodes, edges, return_dict=True)
            PyL_object = PyLouvain(nodes_, edges_)
            PyL_object.apply_method()

            map_dict_rev = {v: k for k, v in map_dict.items()}

            communities += [list(map(lambda x: map_dict_rev[x], comm)) for comm in PyL_object.actual_partition]

        communities = [l for comm_genes in communities
                       for l in self.subsetNetwork(comm_genes).getComponents(return_subgraphs=True)]
        # this step is a consequence of a flaw in the PyLouvain algorithm and leads to suboptimal results

        if verbose:
            len_clust = [len(c) for c in communities]

            print('Size of the largest cluster: %i' % max(len_clust))
            print('Size of the smallest cluster: %i' % min(len_clust))
            print('Number of clusters: %i' % len(len_clust))

        if as_df:
            gene2clust = {gene: i for i, cluster in enumerate(communities) for gene in cluster.node_names}
            return pd.Series([gene2clust[g] for g in self.node_names], index=self.node_names, name='Community')

        else:
            return communities

    def getNumberOfCommonNeighbors(self, sources=None, targets=None):

        if sources is None:
            sources = self.node_names

        if targets is None:
            targets = self.node_names

        A = self.getAdjMatrix(as_df=True)
        n_neigbors = np.matmul(np.transpose(A[sources].values), A[targets].values)  # compatible with directed graphs

        return pd.DataFrame(n_neigbors, index=sources, columns=targets)

    def clusterbydiffusion(self, kernel='LEX', alpha=0.01, nclusters=150, linkage='average', verbose=True):
        A, nodes = self.diffuse(kernel=kernel, alpha=alpha, as_df=False)
        ag = AgglomerativeClustering(n_clusters=nclusters, affinity='precomputed', linkage=linkage)
        ag.fit_predict(np.max(A) - A)
        clusters = [list(nodes[ag.labels_ == i]) for i in pd.unique(ag.labels_)]

        if verbose:
            len_clust = [len(c) for c in clusters]

            print('Size of the largest cluster: %i' % max(len_clust))
            print('Size of the smallest cluster: %i' % min(len_clust))
            print('Number of clusters: %i' % len(len_clust))

        return clusters

    def getnxGraph(self, return_names=True):
        '''return a graph instance of the networkx module'''
        if return_names:
            df = self.getInteractionNamed()
        else:
            df = self.interactions

        return nx.from_pandas_edgelist(df, source='Gene_A', target='Gene_B')

    def getAdjDict(self, return_names=True):
        return to_dict_of_lists(self.getnxGraph(return_names=return_names))

    def getGeodesicDistance(self, start_genes, stop_genes, nx_Graph=None):
        '''
        :param: start_genes genes from which to find paths to stop_genes
        :return: a pandas df containing the pathlengths of shape (start_genes, stop_genes)
        '''

        if isinstance(start_genes[0], str):
            node_names = self.node_names
            start_genes_ = np.intersect1d(node_names, start_genes)
            stop_genes_ = np.intersect1d(node_names, stop_genes)
            return_names = True

        else:
            start_genes_ = np.intersect1d(self.nodes, start_genes)
            stop_genes_ = np.intersect1d(self.nodes, stop_genes)
            return_names = False

        if len(start_genes_) == 0:
            raise IOError('The start_genes are not known.')

        if nx_Graph is None:
            nx_Graph = self.getnxGraph(return_names=return_names)

        path_lengths = np.zeros((len(start_genes_), len(stop_genes_)))

        for istop, stop in enumerate(stop_genes_):
            for istart, start in enumerate(start_genes_):

                try:
                    path_lengths[istart, istop] = len(nx.shortest_path(nx_Graph, start, stop)) - 1

                except nx.NetworkXNoPath:  # basically the graph is not fully connected
                    path_lengths[istart, istop] = np.nan

        # paths_lengths = np.array([len(nx.shortest_path(A, start, stop)) - 1 for stop in stop_genes for start in start_genes])

        paths_lengths_df = pd.DataFrame(path_lengths, index=start_genes_, columns=stop_genes_)
        return paths_lengths_df

    def getShortestPathScore(self, genelist, targetlist):
        shortest_dist = self.getGeodesicDistance(genelist, targetlist)
        scores = shortest_dist.mean(axis=0)

        return scores

    def getMinimmumSpanningTree(self, as_edge_list=True, return_names=True):
        edge_list = super().getMinimmumSpanningTree(return_names=return_names)

        if as_edge_list:
            return edge_list

        else:
            edges = np.array(edge_list)
            df = pd.DataFrame(edges, columns=["Gene_A", "Gene_B"])
            return UndirectedInteractionNetwork(df, node_types=self.node_type_names)

    def getSimpleRepresentation(self, dim=2, **kwargs):

        G = self.getnxGraph()
        coords = nx.spring_layout(G, dim=dim, **kwargs)

        return pd.DataFrame(coords).transpose()

    def replaceNodesWithInteractions(self, tbr_nodes):
        df = super().replaceNodesWithInteractions(tbr_nodes)
        return UndirectedInteractionNetwork(df, node_types=self.node_type_names)

    def findMutationHotspots(self, mut_data, N_perm=100, return_K=False, random_state=42, n_jobs=1,
                             precomputed_kernel=None, kernel='LEX', alpha=0.01, scale=False, self_connected=True):
        '''
        Use permutation testing to find which genes are affected more than expexted by changes,
        :param mut_data: a pandas dataframe (samples  x genes or genes x samples)
        :param N_perm: the number of permutations used to calculate the Z-scores (default: 100)
        :param return_K: whether the kernel used for calculating the Z-scores should be returned (default: False)
        :param precomputed_kernel: a precomputed kernel (default None, kernel is calculated on using this graph)
        The kernel should be a pandas dataframe with the same gene ids row and column names.
        :param kwargs: arguments passed to the diffuse function.
        :return: a dataframe containing the Z-scores for each dataframe
        '''

        assert isinstance(mut_data, pd.DataFrame), 'Please provide a dataframe as input.'

        col_names, row_names = mut_data.columns.values, np.array(list(mut_data.index))
        overlap_cols = np.intersect1d(col_names, self.node_names)
        overlap_rows = np.intersect1d(row_names, self.node_names)

        if len(overlap_cols) > 0:
            common_genes = overlap_cols
        elif len(overlap_rows) > 0:
            common_genes = overlap_rows
            mut_data = mut_data.transpose()
        else:
            raise IOError('The dataframe has no genes in common with this network instance.')

        mut_data = mut_data[common_genes]

        # perform network diffusion
        if precomputed_kernel is None:
            K = self.diffuse(as_df=True, kernel=kernel, alpha=alpha, scale=scale, self_connected=self_connected)
        else:
            K = precomputed_kernel

        K = K.loc[common_genes]  # the colums do not need to contain all common genes

        if return_K:
            return calculateZscores(mut_df=mut_data, Kernel_df=K, N=N_perm, random_state=random_state, n_jobs=n_jobs), K
        else:
            return calculateZscores(mut_df=mut_data, Kernel_df=K, N=N_perm, random_state=random_state, n_jobs=n_jobs)

    def getDriverZscore(self, mut_data, N_perm=100, return_K=False, random_state=42, n_jobs=1,
                        precomputed_kernel=None, kernel='LEX', alpha=0.01, scale=False, self_connected=True,
                        z_score_thresh=3):

        assert isinstance(mut_data, pd.DataFrame), 'Please provide a dataframe as input.'

        col_names, row_names = mut_data.columns.values, np.array(list(mut_data.index))
        overlap_cols = np.intersect1d(col_names, self.node_names)
        overlap_rows = np.intersect1d(row_names, self.node_names)

        if len(overlap_cols) > 0:
            common_genes = overlap_cols
        elif len(overlap_rows) > 0:
            common_genes = overlap_rows
            mut_data = mut_data.transpose()
        else:
            raise IOError('The dataframe has no genes in common with this network instance.')

        mut_data = mut_data[common_genes]

        # perform network diffusion
        if precomputed_kernel is None:
            K = self.diffuse(as_df=True, kernel=kernel, alpha=alpha, scale=scale, self_connected=self_connected)
        else:
            K = precomputed_kernel

        z_scores = calculateZscores(mut_df=mut_data, Kernel_df=K.loc[common_genes],
                                    N=N_perm, random_state=random_state, n_jobs=n_jobs)
        driver_scores = getDriverScore(z_scores, mut_data, K.loc[common_genes], z_thresh=z_score_thresh)

        if return_K:
            return driver_scores, K

        else:
            return driver_scores

    def getTrainTestPairs_MStree(self, train_ratio=0.7, train_validation_ratio=None, excluded_sets=None,
                                 neg_pos_ratio=5, check_training_set=False, random_state=42):
        '''
        :param: train_ratio: The fraction of samples used for training
        :param: neg_pos_ratio: The ratio of negative examples to positive examples
        :param: assumption: Whether we work in the open world or  closed world assumption
        :param: excluded_negatives: should be a set of tuples with negatives interactions to exclude
        :return: positive and negative pairs for both train and test set (4 lists in total)
        '''
        np.random.seed(random_state)
        # To get the negatives we first build the adjacency matrix
        df = self.interactions
        df.values.sort(axis=1)

        allpos_pairs = set(
            zip(df.Gene_A, df.Gene_B))  # this is also calculated in def positives_split, give as additional argument

        if excluded_sets is not None:
            excluded_sets = [tuple((self.gene2int[i[0]], self.gene2int[i[1]])) for i in
                             excluded_sets]  # 1321N1_CENTRAL_NERVOUS_SYSTEM, 0
            allpos_pairs = allpos_pairs - set(excluded_sets)

        pos_train, pos_validation, pos_test = positives_split(df, allpos_pairs, train_ratio, train_validation_ratio)

        assert not set(pos_train) & set(pos_test), "getTrainTestPairs_MStree: overlap positives train - test"
        assert not set(pos_validation) & set(pos_test), "getTrainTestPairs_MStree: overlap positives val - test"

        N_neg = int(neg_pos_ratio * len(allpos_pairs))
        margin = int(0.3 * N_neg)

        # row_c = np.random.choice(self.N_nodes, N_neg + margin, replace=True) not flexible with custom gene2int
        # col_c = np.random.choice(self.N_nodes, N_neg + margin, replace=True)
        row_c = np.random.choice(self.nodes, N_neg + margin, replace=True)
        col_c = np.random.choice(self.nodes, N_neg + margin, replace=True)

        all_pairs = set([tuple(sorted((r_, c_))) for r_, c_ in zip(row_c, col_c) if (c_ != r_)])

        if excluded_sets is not None:
            all_neg = np.array(list(all_pairs - allpos_pairs - set(excluded_sets)), dtype=np.uint16)
        else:
            all_neg = np.array(list(all_pairs.difference(allpos_pairs)), dtype=np.uint16)

        if len(all_neg) > N_neg:
            all_neg = all_neg[:N_neg]
        elif len(all_neg) < N_neg:
            print('The ratio of negatives to positives is lower than the asked %f.'
                  '\nReal ratio: %f' % (neg_pos_ratio, len(all_neg) / len(allpos_pairs)))

        train_ids = int(len(all_neg) * train_ratio)
        if train_validation_ratio is not None:
            valid_ids = int(len(all_neg) * train_validation_ratio)
            neg_train_temp, neg_test = all_neg[:train_ids], all_neg[train_ids:]
            neg_train, neg_validation = neg_train_temp[:valid_ids], neg_train_temp[valid_ids:]
        else:
            neg_train, neg_validation, neg_test = all_neg[:train_ids], np.array([]), all_neg[train_ids:]

        neg_train = list(map(tuple, neg_train))
        neg_validation = list(map(tuple, neg_validation))
        neg_test = list(map(tuple, neg_test))

        assert not set(neg_train) & set(neg_test), "getTrainTestPairs_MStree: overlap negatives train - test"
        assert not set(neg_validation) & set(neg_test), "getTrainTestPairs_MStree: overlap negatives val - test"

        if check_training_set:
            degrees = self.getDegreeDF(return_names=False)
            degrees.index = degrees.Gene.values

            genes, counts = np.unique(all_neg.flatten(), return_counts=True)
            df = pd.DataFrame({'Gene': [self.int2gene[g] for g in genes], 'Counts': counts,
                               'Expected': degrees['Count'].loc[genes].values * neg_pos_ratio})
            df['Difference'] = df.Expected - df.Counts
            return list(pos_train), neg_train, list(pos_validation), neg_validation, \
                   list(pos_test), neg_test, df

        else:
            # pp: changed negative array to match the type of the pos: list of tuples, not list of nd array
            return list(pos_train), neg_train, list(pos_validation), neg_validation, \
                   list(pos_test), neg_test

    def getTrainTestPairs_Balanced(self, neg_pos_ratio=5, train_ratio=0.7, check_training_set=False, random_state=42,
                                   include_negatives=None, verbose=True):
        """
        Function to sample train/test data in a balanced way. The positives are still not required to be 80/20 but the
        negatives are. Also, negatives are sampled per gene such that each gene has the same relative number of negatives.
        :param neg_pos_ratio:
        :param train_ratio:
        :param check_training_set:
        :param random_state:
        :param include_negatives: This needs to be a dict with key=gene and value=negative options for that gene
        :return:
        """
        np.random.seed(random_state)
        random.seed(random_state)
        # Store all positive interactions
        pos_train, pos_valid, pos_test = positives_split(self.interactions, train_ratio=train_ratio)
        # TODO: different implementation, not based on MS tree
        degrees = self.getDegreeDF(return_names=False, set_index=True)
        degrees.index = degrees.Gene.values
        adj_dict = self.getAdjDict(return_names=False)

        counts = degrees.sort_values(by='Gene', ascending=True)['Count'].values * neg_pos_ratio # needs to be sorted for easy indexing
        sorted_genes = degrees.sort_values(by='Count', ascending=False)['Gene'].values

        all_negatives, genes, genes_train, genes_test, done, neg_train, neg_test = [], [], [], [], [], [], []

        for gene in sorted_genes:
            if counts[gene] > 0: # If counts is 0 for a gene, that means it has already been seen npr*pos times as a negative
                neighbors = adj_dict[gene]
                zeros = list(np.where(counts <= 0)[0])
                nzeros = len(zeros)

                draws = np.minimum(self.N_nodes,
                                   counts[gene] + len(neighbors) + len(done) + nzeros + 1)  # Two last terms?
                if include_negatives is not None:
                    candidates = include_negatives[gene] if isinstance(include_negatives[gene], list) else list(include_negatives[gene])
                else:
                    candidates = np.random.choice(self.N_nodes, draws, replace=False)

                not_considered = neighbors + [gene] + done + zeros # don't include the zeros here bc these have been seen enough times as a negative
                negatives = np.setdiff1d(np.array(candidates, dtype=int), np.array(not_considered, dtype=int))

                max_id = np.minimum(counts[gene], len(negatives))

                if verbose:
                    if max_id < counts[gene]:
                        print('Not enough negatives available for gene %s' % self.int2gene[gene])

                negatives = random.sample(set(negatives), max_id)
                train_id = int(np.round(train_ratio * len(negatives)))

                neg_train += negatives[:train_id]
                neg_test += negatives[train_id:]

                counts[gene] = counts[gene] - len(negatives)
                counts[negatives] = counts[negatives] - 1

                all_negatives += list(negatives)
                genes_test += [gene for _ in range(len(negatives[train_id:]))]
                genes_train += [gene for _ in range(train_id)]
                genes = genes_train + genes_test
                done += [gene]

        neg_train = list(zip(genes_train, neg_train))
        neg_test = list(zip(genes_test, neg_test))

        if check_training_set:
            genes, counts = np.unique(np.concatenate((all_negatives, genes)), return_counts=True)
            df = pd.DataFrame({'Gene': [self.int2gene[g] for g in genes], 'Counts': counts,
                               'Expected': degrees.loc[genes]['Count'].values * neg_pos_ratio})
            df['Difference'] = df.Expected - df.Counts
            return pos_train, neg_train, pos_test, neg_test, df

        else:
            return pos_train, neg_train, pos_test, neg_test

    def getAllTrainData(self, neg_pos_ratio=5, random_state=42):
        np.random.seed(random_state)
        df = self.interactions
        df.values.sort(axis=1)

        X = set(zip(df.Gene_A, df.Gene_B))
        Y_pos = [1 for _ in range(len(X))]

        N_neg = int(neg_pos_ratio * len(X))
        all_pairs = set([(i, j) for i in np.arange(self.N_nodes, dtype=np.uint16) for j
                         in np.arange(i + 1, dtype=np.uint16)])

        all_neg = list(all_pairs.difference(X))
        ids = np.random.choice(len(all_neg), np.minimum(N_neg, len(all_neg)), replace=False)

        X = np.array(list(X) + [all_neg[i] for i in ids])
        Y = np.array(Y_pos + [0 for _ in range(len(ids))])

        return X, Y

    def diffusionBasedEnrichment(self, gene_list, pathways, N_perm=10000, **kwargs):

        gene_list = np.intersect1d(gene_list, self.node_names)
        pathways = np.intersect1d(pathways, self.node_names)

        if len(gene_list) == 0:
            IOError('The gene names provided do not occur in the network')

        if len(pathways) == 0:
            IOError('The pathway nodes are not present in the networks')

        diffusion_df = self.diffuse(as_df=True, **kwargs)

        path_diffusion = diffusion_df.loc[pathways]
        scores = path_diffusion[gene_list].sum(axis=1)

        # TODO implement pvals
        random_scores = np.zeros((len(scores), N_perm))
        all_genes = np.setdiff1d(diffusion_df.columns.values, pathways)

        for n in range(N_perm):
            random_genes = np.random.choice(all_genes, size=len(gene_list), replace=False)
            random_scores[:, n] = path_diffusion[random_genes].sum(axis=1)

        pvals = np.sum(random_scores < scores.values[..., None], axis=1) / N_perm

        return pd.DataFrame({'Score': scores, 'Pval': pvals}, index=scores.index)

    @staticmethod
    def list_embedding_methods():
        print("\nLaplacian - GF - HOPE - GraRep - DeepWalk - node2vec - LINE - SDNE\n")
        return

    # TODO: Test other methods
    def generateEmbeddings(self, method, train_interactions=None, dimensions=10, save=None, epochs=5,
                           number_walks=32, workers=8, window_size=10, p=1, q=1, order=2,
                           walk_length=64, weight_decay=5e-4, lr=0.01, kstep=4, alpha=0.3, beta=0, nu1=1e-5,
                           nu2=1e-4, bs=200, encoder_list='[1000, 128]'):

        try:
            # from openne import gf, hope, lap, line, node2vec, sdne, grarep, graph
            from openne import lap, node2vec, grarep, graph # Don't require tensorflow

        except ImportError:
            raise ImportError("Please install openNE to use this function.")

        graph.Graph.from_df = from_df

        if train_interactions is None:
            train_interactions = self.getInteractionNamed()

        G_train = graph.Graph()
        G_train.from_df(train_interactions)

        # np.savetxt("tmp_train.csv", train_interactions.astype(int), delimiter=' ', fmt='%i')
        # assert os.path.exists("tmp_train.csv") and os.path.getsize("tmp_train.csv") > 0,"Temporary file could not be written"
        # G_train.read_edgelist("tmp_train.csv", weighted=False, directed=False)
        # os.remove("tmp_train.csv")

        if method == 'Laplacian':
            model = lap.LaplacianEigenmaps(G_train, rep_size=dimensions)

        elif method == 'GF':
            model = gf.GraphFactorization(G_train, rep_size=dimensions,
                                          epoch=epochs, learning_rate=lr, weight_decay=weight_decay)

        elif method == 'HOPE':
            model = hope.HOPE(graph=G_train, d=dimensions)

        elif method == 'GraRep':
            model = grarep.GraRep(graph=G_train, Kstep=kstep, dim=dimensions)

        elif method == 'DeepWalk':
            model = node2vec.Node2vec(graph=G_train, path_length=walk_length,
                                      num_paths=number_walks, dim=dimensions,
                                      workers=workers, window=window_size, dw=True)

        elif method == 'node2vec':
            model = node2vec.Node2vec(graph=G_train, path_length=walk_length,
                                      num_paths=number_walks, dim=dimensions,
                                      workers=workers, p=p, q=q, window=window_size)

        # line already has a random seed each time => cur_seed
        elif method == 'LINE':
            model = line.LINE(G_train, epoch=epochs,
                              rep_size=dimensions, order=order)

        elif method == 'SDNE':
            encoder_layer_list = ast.literal_eval(encoder_list)
            model = sdne.SDNE(G_train, encoder_layer_list=encoder_layer_list,
                              alpha=alpha, beta=beta, nu1=nu1, nu2=nu2,
                              batch_size=bs, epoch=epochs, learning_rate=lr)
        else:
            raise ValueError(f'Invalid method: {method}')

        self.embedding_dict = model.vectors  # TODO: couldn't we make this an instance of the Embedding class and put evaluateEmbeddings in the Embedding class?
        if save is not None:
            print("Saving embeddings...")
            model.save_embeddings(save)
            # tf.compat.v1.reset_default_graph() # Only necessary for methods that require TF
        else:
            # tf.compat.v1.reset_default_graph()

            return model.vectors  # TODO: this is dangerous, if the user changes the embedding they will also change the class attribute => is this better?

    def evaluateEmbeddings(self, X_train, y_train, X_test, y_test, shuffle=False, seed=42, embedding_lookup=None,
                           fitter='logistic_regression', verbose=True, return_model=False):

        if embedding_lookup is None:
            embedding_lookup = self.embedding_dict

        # TODO: other edge embedding methods
        if type(list(embedding_lookup.keys())[0]) == str:
            X_train_embs = np.array([np.append(embedding_lookup[str(i[0])],
                                               embedding_lookup[str(i[1])]) for i in X_train])
            X_test_embs = np.array([np.append(embedding_lookup[str(i[0])],
                                              embedding_lookup[str(i[1])]) for i in X_test])
        else:
            X_train_embs = np.array([np.append(embedding_lookup[i[0]],
                                               embedding_lookup[i[1]]) for i in X_train])
            X_test_embs = np.array([np.append(embedding_lookup[i[0]],
                                              embedding_lookup[i[1]]) for i in X_test])

        if shuffle:
            c = list(zip(X_train_embs, y_train))
            random.shuffle(c)
            X_train_embs, y_train = zip(*c)

            c = list(zip(X_test_embs, y_test))
            random.shuffle(c)
            X_test_embs, y_test = zip(*c)

        self.clf1 = FITTERMAPPER[fitter]
        self.clf1.fit(X_train_embs, y_train)
        model = self.clf1
        y_pred_proba = self.clf1.predict_proba(X_test_embs)[:, 1]
        y_pred = self.clf1.predict(X_test_embs)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        if verbose:
            print('\n' + '#' * 9 + ' Link Prediction Performance ' + '#' * 9)
            print(f'AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}')
            print('#' * 50)
        if return_model:
            return auc_roc, auc_pr, accuracy, f1, cm, model
        else:
            return auc_roc, auc_pr, accuracy, f1, cm

    def filter_graph_by_LP(self, prob_matrix_df, n_swaps, max_attempts, strict=False, inplace=False):
        pred_genes = set(prob_matrix_df.columns.to_list())

        assert pred_genes == set(prob_matrix_df.index.to_list()), \
            'Please provide prob_matrix with identical rows and columns'
        subset = False

        if pred_genes != set(self.node_names):
            common_genes = list(pred_genes.intersection(set(self.node_names)))
            net = self.subsetNetwork(common_genes)
            correct_order = np.array([net.int2gene[i] for i in range(net.N_nodes)])

            prob_matrix_df = prob_matrix_df.loc[correct_order][correct_order]

            current_interactions = np.transpose(net.interactions.values)
            current_interactions = (current_interactions[0, :], current_interactions[1, :])
            subset = True

        else:
            correct_order = np.array([self.int2gene[i] for i in range(self.N_nodes)])
            prob_matrix_df = prob_matrix_df.loc[correct_order][correct_order]

            current_interactions = np.transpose(self.interactions.values)
            current_interactions = (current_interactions[0, :], current_interactions[1, :])

        current_interactions = (np.hstack((current_interactions[0], current_interactions[1])),
                                np.hstack((current_interactions[1], current_interactions[0])))

        new_interactions = filter_graph_by_LP_swaps(prob_matrix=prob_matrix_df.values,
                                                    current_interactions=current_interactions,
                                                    n_swaps=n_swaps,
                                                    max_attempts=max_attempts,
                                                    strict=strict)

        if subset:
            int2gene = net.int2gene
            current_interactions = set(zip([int2gene[i] for i in current_interactions[0]],
                                           [int2gene[i] for i in current_interactions[1]]))
            new_interactions = set(zip([int2gene[i] for i in new_interactions[0]],
                                       [int2gene[i] for i in new_interactions[1]]))

            new_interactions = self.interactions_as_set(return_names=True).difference(current_interactions) \
                .union(new_interactions)

            new_interactions = tuple(zip(*list(new_interactions)))
            new_interactions = (np.array(new_interactions[0]), np.array(new_interactions[1]))
            new_interactions = np.transpose(new_interactions)
            new_interactions = pd.DataFrame(new_interactions, columns=['Gene_A', 'Gene_B'])

        else:
            new_interactions = np.transpose(new_interactions)
            new_interactions = pd.DataFrame(new_interactions, columns=['Gene_A', 'Gene_B'])
            new_interactions = new_interactions.applymap(lambda x: self.int2gene[x])

        if inplace:
            self.interactions = new_interactions

        else:
            return UndirectedInteractionNetwork(new_interactions, node_types=self.node_type_names)

    def filter_graph_by_LP_fastest(self, prob_matrix_df, n_swaps, max_attempts, strict=False, inplace=False):
        pred_genes = set(prob_matrix_df.columns.to_list())

        assert pred_genes == set(prob_matrix_df.index.to_list()), \
            'Please provide prob_matrix with identical rows and columns'
        subset = False

        A = self.makeSelfConnected().getAdjMatrix(as_df=True)

        if pred_genes != set(self.node_names):
            common_genes = list(pred_genes.intersection(set(self.node_names)))
            net = self.subsetNetwork(common_genes)
            correct_order = np.array([net.int2gene[i] for i in range(net.N_nodes)])

            prob_matrix_df = prob_matrix_df.loc[correct_order][correct_order]
            A = A.loc[correct_order][correct_order]

            current_interactions = (net.interactions.Gene_A.values,
                                    net.interactions.Gene_B.values)
            subset = True

        else:
            correct_order = np.array([self.int2gene[i] for i in range(self.N_nodes)])
            prob_matrix_df = prob_matrix_df.loc[correct_order][correct_order]
            A = A.loc[correct_order][correct_order]

            current_interactions = (self.interactions.Gene_A.values,
                                    self.interactions.Gene_B.values)

        new_interactions = filter_graph_by_LP_swaps_fastest(prob_matrix=prob_matrix_df.values,
                                                            A=A.values,
                                                            current_interactions=current_interactions,
                                                            n_swaps=n_swaps,
                                                            max_attempts=max_attempts,
                                                            strict=strict)

        if subset:
            int2gene = net.int2gene
            current_interactions = set(zip([int2gene[i] for i in current_interactions[0]],
                                           [int2gene[i] for i in current_interactions[1]]))
            new_interactions = set(zip([int2gene[i] for i in new_interactions[0]],
                                       [int2gene[i] for i in new_interactions[1]]))

            new_interactions = self.interactions_as_set(return_names=True).difference(current_interactions) \
                .union(new_interactions)

            new_interactions = tuple(zip(*list(new_interactions)))
            new_interactions = (np.array(new_interactions[0]), np.array(new_interactions[1]))
            new_interactions = np.transpose(new_interactions)
            new_interactions = pd.DataFrame(new_interactions, columns=['Gene_A', 'Gene_B'])

        else:
            new_interactions = np.transpose(new_interactions)
            new_interactions = pd.DataFrame(new_interactions, columns=['Gene_A', 'Gene_B'])
            new_interactions = new_interactions.applymap(lambda x: self.int2gene[x])

        if inplace:
            self.interactions = new_interactions

        else:
            return UndirectedInteractionNetwork(new_interactions, node_types=self.node_type_names)

    def predict_full_matrix(self, sources=None, targets=None, embedding_lookup=None, evaluate=False, Y_true=None,
                            verbose=True, embed_dim=10):
        """
        Uses the fitter from the function 'evaluateEMbeddings'
        :param sources:
        :param targets:
        :param embedding_lookup:
        :param evaluate:
        :param Y_true:
        :param verbose:
        :param embed_dim:
        :return:
        """
        if embedding_lookup is None:
            embedding_lookup = self.embedding_dict

        interactions_to_predict_tuples = product(sources, targets)

        interactions_to_predict = np.array([np.append(embedding_lookup[str(self.gene2int[s])],
                                                      embedding_lookup[str(self.gene2int[t])])
                                            for s, t in interactions_to_predict_tuples])

        assert interactions_to_predict.shape == tuple((len(sources) * len(targets), embed_dim * 2)), 'ERROR sumtin wong'

        if self.clf1 is not None:
            y_pred_proba = self.clf1.predict_proba(interactions_to_predict)[:, 1]
            y_pred = self.clf1.predict(interactions_to_predict)
            prob_mat = pd.DataFrame(np.reshape(y_pred_proba, (len(sources), len(targets)), order='F'),
                                    columns=targets,
                                    index=sources)

            if evaluate:
                auc_roc = roc_auc_score(Y_true, y_pred_proba)
                auc_pr = average_precision_score(Y_true, y_pred_proba)
                accuracy = accuracy_score(Y_true, y_pred)
                f1 = f1_score(Y_true, y_pred)
                cm = confusion_matrix(Y_true, y_pred)
                if verbose:
                    print('\n' + '#' * 9 + ' Link Prediction Performance ' + '#' * 9)
                    print(f'AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}')
                    print('#' * 50)
                return prob_mat, tuple((auc_roc, auc_pr, accuracy, f1, cm))
            else:
                return prob_mat
        else:
            raise TypeError("No Classifier model definied, run function evaluateEmbeddings first")

    # def get_KNN_network_(self, kn=11, verbose=True, kernel='VANDIN', alpha=0.01, self_connected=False, **kernel_args):
    #     '''original, but deprecated version to get the knn_graph'''

    #     L_inv = self.diffuse(kernel=kernel, alpha=alpha, as_df=True, self_connected=self_connected, **kernel_args)
    #     np.fill_diagonal(L_inv.values, 0)

    #     edges = []
    #     # TODO:make this faster
    #     for gene in L_inv.index:
    #         gene_knn = L_inv[gene].sort_values(ascending=False)[:kn].index

    #         for neighbor in gene_knn:
    #             if L_inv[gene].loc[neighbor] > 0:
    #                 edges.append((gene, neighbor))

    #     df = pd.DataFrame(np.array(edges), columns=['Gene_A', 'Gene_B'])

    #     return DirectedInteractionNetwork(df, allow_self_connected=self_connected, node_types=self.node_type_names)

    # def get_KNN_network(self, kn=11, verbose=True, kernel='VANDIN', kn_criterion='connectivity',
    #                     alpha=0.01, self_connected=False, **kernel_args):
    #     '''
    #     Create a knn graph
    #     :param kn: number of neighbors, if None kn is set according to kn_criterion
    #     :param verbose:
    #     :param kernel: what kernel to use for diffusion
    #     :param kn_criterion: what criterion to use if kn is not specified
    #     :param alpha: the strength of diffusion, its interpretation depends on the kernel chosen
    #     :param self_connected: whether the Graph is self connected for the diffusion step
    #     :param kernel_args: other args passed to the diffuse() function
    #     :return:
    #     '''

    #     assert self.isConnected, "Graph is not connected, please use get_KNN_network_ function"
    #     L_inv, nodes = self.diffuse(kernel=kernel, alpha=alpha, as_df=False,
    #                                 self_connected=self_connected, **kernel_args)
    #     np.fill_diagonal(L_inv, 0)
    #     sorted = np.argsort(-L_inv, axis=1)  # sort row_wise for directed graphs

    #     if kn is None:

    #         # if kn_criterion.lower() == 'connectivity':
    #         kn = 0
    #         net_connected = False

    #         while not net_connected:
    #             edges = []
    #             kn += 1
    #             for i, gene in enumerate(nodes):
    #                 edges += list(zip(kn * [nodes[i]], list(nodes[sorted[i, :kn]])))

    #             net = DirectedInteractionNetwork(pd.DataFrame(np.array(edges),
    #                                                           columns=['Gene_A', 'Gene_B']),
    #                                              allow_self_connected=False,
    #                                              node_types=self.node_type_names)
    #             net_connected = net.isConnected

    #         # else:  # second criterion is based on assuring that each gene is uniquely defined by its neighborhood
    #         #    not_unique = True
    #         #    kn = 1
    #         #    envs = np.vstack((np.arange(self.N_nodes), sorted))

    #         # while not_unique:
    #         #    envs_k = set(np.unique(envs[i, :(kn + 1)]) for i in range(self.N_nodes))
    #         #    not_unique = len(envs_k) != self.N_nodes

    #         # print('Used value of kn: %i' %kn)

    #         return net

    #     else:
    #         edges = []
    #         for i, gene in enumerate(nodes):
    #             edges += list(zip(kn * [gene], list(nodes[sorted[i, :kn]])))

    #         df = pd.DataFrame(np.array(edges), columns=['Gene_A', 'Gene_B'])

    #         return DirectedInteractionNetwork(df, allow_self_connected=self_connected, node_types=self.node_type_names)

    def getKNNglap(self, kn=11, verbose=True, kernel='VANDIN', alpha=0.01, self_connected=False, **kernel_args):

        glap_inv_starttime = time.time()

        # Construct network laplacian matrix
        L_inv = self.diffuse(kernel=kernel, alpha=alpha, as_df=True, self_connected=self_connected, **kernel_args)

        if verbose:
            print('Graph influence matrix calculated:', time.time() - glap_inv_starttime, 'seconds')

        KNN_graph = nx.Graph()

        for gene in L_inv.index:
            gene_knn = L_inv[gene].sort_values(ascending=False)[:kn].index

            for neighbor in gene_knn:
                if L_inv[gene].loc[neighbor] > 0:
                    KNN_graph.add_edge(gene, neighbor)

        KNN_nodes = list(KNN_graph.nodes)

        knnGlap_sparse = nx.laplacian_matrix(KNN_graph)
        knnGlap = pd.DataFrame(knnGlap_sparse.todense(), index=KNN_nodes, columns=KNN_nodes)

        return knnGlap, KNN_graph

    def mixed_netNMF(self, data, k=3, l=200, maxiter=250, eps=1e-15, err_tol=1e-4, err_delta_tol=1e-8,
                     verbose=False, **kernelargs):

        KNN_glap = self.getKNNglap(**kernelargs)
        # Initialize H and W Matrices from data array if not given
        r, c = data.shape[0], data.shape[1]
        # Initialize H
        H_init = np.random.rand(k, c)
        H = np.maximum(H_init, eps)
        # Initialize W
        W_init = np.linalg.lstsq(H.T, data.T)[0].T
        W_init = np.dot(W_init, np.diag(1 / sum(W_init)))
        W = np.maximum(W_init, eps)

        if verbose:
            print('W and H matrices initialized')

        # Get graph matrices from laplacian array
        D = np.diag(np.diag(KNN_glap)).astype(float)
        A = (D - KNN_glap).astype(float)
        if verbose:
            print('D and A matrices calculated')
        # Set mixed netNMF reconstruction error convergence factor
        XfitPrevious = np.inf

        # Updating W and H
        for i in range(maxiter):
            XfitThis = np.dot(W, H)
            WHres = np.linalg.norm(data - XfitThis)  # Reconstruction error

            # Change in reconstruction error
            if i == 0:
                fitRes = np.linalg.norm(XfitPrevious)
            else:
                fitRes = np.linalg.norm(XfitPrevious - XfitThis)
            XfitPrevious = XfitThis

            # Reporting netNMF update status
            if (verbose) & (i % 10 == 0):
                print('Iteration >>', i, 'Mat-res:', WHres, 'Lambda:', l, 'Wfrob:', np.linalg.norm(W))
            if (err_delta_tol > fitRes) | (err_tol > WHres) | (i + 1 == maxiter):
                if verbose:
                    print('NMF completed!')
                    print('Total iterations:', i + 1)
                    print('Final Reconstruction Error:', WHres)
                    print('Final Reconstruction Error Delta:', fitRes)

                numIter = i + 1
                finalResidual = WHres
                break

            # Terms to be scaled by regularization constant: l
            KWmat_D = np.dot(D, W)
            KWmat_W = np.dot(A, W)

            # Update W with network constraint
            W = W * ((np.dot(data, H.T) + l * KWmat_W + eps) / (np.dot(W, np.dot(H, H.T)) + l * KWmat_D + eps))
            W = np.maximum(W, eps)
            # Normalize W across each gene (row-wise)
            W = W / matlib.repmat(np.maximum(sum(W), eps), len(W), 1);

            # Update H
            H = np.array([nnls(W, data[:, j])[0] for j in range(c)]).T
            # ^ Hofree uses a custom fast non-negative least squares solver here, we will use scipy's implementation here
            H = np.maximum(H, eps)

        return W, H, numIter, finalResidual

    def degreePreservingPermutation(self, N_swaps=1000):
        df = super().degreePreservingPermutation(N_swaps=N_swaps)
        return UndirectedInteractionNetwork(df, node_types=self.node_type_names)

    def plot_degree_distribution(self, degreeDf1=None, degreeDf2=None, title=None, legend=False, save_name=None,
                                 s=5, labels=None):

        if degreeDf1 is None:
            degreeDf1 = self.getDegreeDF()

        fig, ax = plt.subplots()
        ax.scatter(np.arange(1, degreeDf1.shape[0] + 1), degreeDf1['Count'], s=s, label=labels[0])
        if degreeDf2 is not None:
            ax.scatter(np.arange(1, degreeDf2.shape[0] + 1), degreeDf2['Count'], s=s, label=labels[1])
        if title is not None:
            plt.title(title)
        else:
            plt.title(f'nodes degreeDf1 {degreeDf1.shape[0]}  |  nodes degreeDf2 {degreeDf2.shape[0]}')
        if legend:
            plt.legend()
        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name)
            plt.close()

