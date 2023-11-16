from sklearn.linear_model import LogisticRegression, SGDClassifier
from networkx.exception import NetworkXError, AmbiguousSolution
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from collections import defaultdict
from scipy.linalg import expm
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import warnings
import copy
import time
import types

# complete the class specific functions in Directed networks


class Graph:
    '''
        Builds a graph from _path.
        _path: a path to a file containing "node_from node_to" edges (one per line)
    '''

    @classmethod
    def from_file(cls, path, colnames=('Gene1', 'Gene2'), sep=',',
                  header=0, column_index=None, keeplargestcomponent=False,
                  network_type='kegg', gene_id_type='symbol'):

        if network_type is None:
            network_df = pd.read_csv(path, sep=sep, header=header, low_memory=False, index_col=column_index)
            network_df = network_df[list(colnames)]
        elif network_type.lower() == 'kegg':
            network_df = pd.read_csv(path, sep='\t', header=0, dtype=str)[['from', 'to']]

        elif network_type.lower() == 'string':
            network_df = pd.read_csv(path, sep='\t', header=0)[['Gene1', 'Gene2']]

        elif network_type.lower() == 'biogrid':
            network_df = pd.read_csv(path, sep='\t', header=0)

            if gene_id_type.lower() == 'entrez':
                network_df = network_df[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']]

            elif gene_id_type.lower() == 'symbol':
                network_df = network_df[['Official Symbol Interactor A', 'Official Symbol Interactor B']]

            else:
                raise IOError('gene_id_type not understood.'
                              'For Biogrid please specify entrez or symbol.')

        else:
            raise IOError('Network type not understood.'
                          'Please specify kegg, biogrid or reactome, or enter None for custom network type.')

        return cls(network_df, keeplargestcomponent=keeplargestcomponent)

    def __init__(self, interaction_df, colnames=None, verbose=True, keeplargestcomponent=False,
                 allow_self_connected=False, node_types=None, drop_duplicates=True,
                 gene2int=None):
        '''
        :param: interaction_df a pandas edgelist consisting of (at least two) columns,
        indicating the two nodes for each edge
        :param: colnames, the names of the columns that contain the nodes and optionally some edge attributes.
        :param: node_types, dictionary with node types = keys, node names = values
        The first two columns must indicate the nodes from the edgelsist
        '''

        def isinteger(x):
            try:
                return np.all(np.equal(np.mod(x, 1), 0))

            except:
                return False

        self.attr_names = None

        if colnames is not None:
            interaction_df = interaction_df[list(colnames)]
            if len(colnames) > 2:
                self.attr_names = colnames[2:]  # TODO this needs to be done better

        elif interaction_df.shape[1] == 2:
            interaction_df = interaction_df

        else:
            print('Continuing with %s and %s as columns for the nodes' % (interaction_df.columns.values[0],
                                                                          interaction_df.columns.values[1]))
            interaction_df = interaction_df.iloc[:, :2]

        if drop_duplicates:
            interaction_df = interaction_df.drop_duplicates()

        self.interactions = interaction_df
        old_col_names = list(self.interactions.columns)
        self.interactions.rename(columns={old_col_names[0]: 'Gene_A',
                                          old_col_names[1]: 'Gene_B'},
                                 inplace=True)

        if not allow_self_connected:
            self.interactions = self.interactions.loc[self.interactions.Gene_A != self.interactions.Gene_B]

        if isinteger(self.interactions.Gene_A.values):  # for integer nodes do numerical ordering of the node_names
            node_names = np.unique(self.interactions[['Gene_A', 'Gene_B']].values)
            self.interactions = self.interactions.astype(str)
            node_names = node_names.astype(str)

        else:
            self.interactions = self.interactions.astype(str)
            node_names = np.unique(self.interactions[['Gene_A', 'Gene_B']].values)

        if gene2int is not None:
            assert isinstance(gene2int, dict), "If provided, gene2int must be a dict mapping nodenames onto ints."
            gene2int = check_node_dict(node_names, gene2int, type_dict="gene2int")
            assert len(gene2int.values()) == len(set(gene2int.values())), "if provided, " \
                                                                          "gene2int must map each nodename onto a unique int."
            self.int2gene = {i: g for g, i in gene2int.items()}

        else:
            self.int2gene = {i: name for i, name in enumerate(node_names)}
            gene2int = self.gene2int

        self.interactions = self.interactions.applymap(lambda x: gene2int[x])
        self.nodes = np.array([gene2int[s] for s in node_names]).astype(np.int_)

        if node_types is None:
            self.node_types = {i: "node" for i in self.nodes}

        elif isinstance(node_types, dict):
            if isinstance(list(node_types.keys())[0], int):
                self.node_types = check_node_dict(self.nodes, node_types, type_dict="node_types")
            else:
                node_type_names = check_node_dict(self.node_names, node_types, type_dict="node_types")
                self.node_types = {self.gene2int[k]: v for k, v in node_type_names.items()}

        else:
            raise IOError("The node_types are not understood, "
                          "please provide a dict mapping each node on their node type.")

        self.embedding_dict = None
        if keeplargestcomponent:
            self.keepLargestComponent(verbose=verbose, inplace=True)

        if verbose:
            print('%d Nodes and %d interactions' % (len(self.nodes),
                                                    self.interactions.shape[0]))
        self.directed = False

    def deepcopy(self):
        return copy.deepcopy(self)

    @property
    def node_type_names(self):
        return {self.int2gene[g]: t for g, t in self.node_types.items()}

    @property
    def gene2int(self):
        return {v: k for k, v in self.int2gene.items()}

    @property
    def node_names(self):
        return np.array([self.int2gene[i] for i in self.nodes])

    @property
    def N_nodes(self):
        return len(self.nodes)

    @property
    def N_interactions(self):
        return self.interactions.shape[0]

    @property
    def type2nodes(self):
        if self.node_types is not None:
            odict = defaultdict(list)
            for k, v in self.node_types.items():
                odict[v].append(k)

        else:
            odict = None

        return odict

    @property
    def _get_edge_ids(self):
        edge_ids = self.interactions["Gene_A"].values * self.N_nodes + self.interactions["Gene_B"].values
        return edge_ids

    @property
    def is_bipartite(self):
        G = self.getnxGraph(return_names=True)
        bipartite = True

        try:
            l, r = nx.bipartite.sets(G.to_undirected())
            return bipartite, l, r

        except (NetworkXError, AmbiguousSolution) as e:
            bipartite = False
            return bipartite, None, None

    def __contains__(self, gene):
        return gene in self.node_names

    def __repr__(self):
        return self.getInteractionNamed().__repr__()

    def __str__(self):
        return self.getInteractionNamed().__str__()

    def __len__(self):
        return self.N_nodes

    def __eq__(self, other):
        if isinstance(other, Graph):
            return self.interactions_as_set() == other.interactions_as_set()
        return NotImplemented

    def edge_list(self, return_names=True):
        if return_names:
            df = self.getInteractionNamed()

        else:
            df = self.interactions

        return list(zip(df.Gene_A.values, df.Gene_B.values))

    def set_node_types(self, node_types):
        if isinstance(node_types, dict):
            node_type_names = check_node_dict(self.node_names, node_types, type_dict="node_types")
            self.node_types = {self.gene2int[k]: v for k, v in node_type_names.items()}

        else:
            raise IOError("The node_types are not understood, "
                          "please provide a dict mapping each node on their node type.")

    def get_node_type_subnet(self, type, inplace=False):
        """
        returns the subnetwork containing all nodes of a particular type
        :param type:
        :param inplace:
        :return:
        """
        try:
            genes = self.type2nodes[type]

        except KeyError:
            raise IOError("The type is not known, please check that the type is present in node_types.")

        if inplace:
            self.removeNodes(genes, inplace=inplace)

        else:
            return self.removeNodes(genes, inplace=inplace)

    def get_interactions_per_node_type(self):
        if self.node_types is None:
            return None

        node_type_interactions = self.interactions[["Gene_A", 'Gene_B']].applymap(lambda x: self.node_types[x])
        uniq = set(node_type_interactions.itertuples(index=False, name=None))

        return uniq

    def get_node_type_edge_counts(self):
        if self.node_types is not None:
            df = self.interactions[["Gene_A", 'Gene_B']].applymap(lambda x: self.node_types[x])
            return df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'counts'})

        else:
            warnings.warn("The node types are not defined, please provide these first.")
            return None

    def get_edges_by_node_types(self, node_type1, node_type2, return_names=True, directed=False):
        df = self.interactions[["Gene_A", 'Gene_B']].applymap(lambda x: self.node_types[x]).values

        if directed:
            mask = (df[:, 0] == node_type1) & (df[:, 1] == node_type2)

        else:
            mask = ((df[:, 0] == node_type1) & (df[:, 1] == node_type2)) | \
                   ((df[:, 0] == node_type2) & (df[:, 1] == node_type1))

        if return_names:
            return self.getInteractionNamed()[mask]
        else:
            return self.interactions[mask]

    def sample_positives_and_negatives(self, neg_pos_ratio=5, excluded_sets=None, return_names=False):
        # TODO: make excluded sets name-based instead of number-based, without breaking getTrainTestPairs_MStree_ML
        directed = self.directed
        edge_counts = self.get_node_type_edge_counts()
        all_neg, all_pos = [], []
        type2nodes = self.type2nodes

        if excluded_sets is None:
            excluded_sets = set()

        elif not self.directed:
            excluded_sets = set([tuple(sorted((r_, c_))) for r_, c_ in excluded_sets])

        for row in edge_counts.itertuples(index=False, name=None):
            # select all positives from the training set
            positives = self.get_edges_by_node_types(node_type1=row[0], node_type2=row[1], return_names=False,
                                                     directed=directed)

            sets = set(zip(positives.Gene_A, positives.Gene_B))

            positives = set(positives.itertuples(index=False, name=None)).difference(excluded_sets)

            # get negatives based on the count
            N_negatives = np.round(len(positives) * neg_pos_ratio).astype(int)

            nodes1, nodes2 = type2nodes[row[0]], type2nodes[row[1]]

            margin = np.int_(0.3 * N_negatives)


            row_c = np.random.choice(nodes1, N_negatives + margin, replace=True)
            col_c = np.random.choice(nodes2, N_negatives + margin, replace=True)

            if not self.directed:
                assert len(sets) == len(set([tuple(sorted(t)) for t in sets])), "Duplicate edges pos"

                all_pairs = set([tuple(sorted((r_, c_))) for r_, c_ in zip(row_c, col_c) if
                                 (c_ != r_)])  # should be sorted for Undirected

            else:
                all_pairs = set([(r_, c_) for r_, c_ in zip(row_c, col_c) if (c_ != r_)])

            if excluded_sets is not None:
                negatives = list(all_pairs - positives - set(excluded_sets))

            else:
                negatives = list(all_pairs.difference(positives))

            if len(negatives) > N_negatives:
                negatives = negatives[:N_negatives]

            if len(negatives) < N_negatives:
                print('The ratio of negatives to positives is lower than the asked %f.'
                      '\nReal ratio: %f' % (neg_pos_ratio, len(negatives) / len(positives)))

            excluded_sets = excluded_sets.union(positives).union(set(negatives))

            all_neg += negatives
            all_pos += positives

        if return_names:
            func_ = lambda x: self.int2gene[x]
            # func_reverse_ = lambda x: net.gene2int[x] if x in net.gene2int else np.nan # genes in one might not be in other
            vf = np.vectorize(func_)

            return vf(np.asarray(all_pos)), vf(np.asarray(all_neg))

        else:
            return np.asarray(all_pos), np.asarray(all_neg)

    def update_node_type(self, new_dict):
        for k, v in new_dict:
            self.node_types[k] = v

    def get_type_Series(self):
        return pd.Series(self.node_types)

    def remove_nodes_from_type_dict(self, genes):
        for gene in genes:
            self.node_types.pop(gene, None)

    def check_node_dict(self, node_dict):
        """
        expects a dict where keys = nodes in the network, values are the corresponding types
        """
        net_node_names = self.node_names
        nodes = list(node_dict.keys())

        nodes_missing_in_dict = set(net_node_names) - set(nodes)

        if len(nodes_missing_in_dict) > 0:
            print("The following genes have no annotation:")
            print(nodes_missing_in_dict)
            return None

        nodes_missing_in_network = set(nodes) - set(net_node_names)

        if len(nodes_missing_in_network) > 0:
            print("The following genes are missing from the network and will be removed:")
            print(nodes_missing_in_network)
            return {k: v for k, v in node_dict.items() if k not in nodes_missing_in_network}
        else:
            return node_dict

    def getInteractionNamed(self, return_both_directions=False):
        if return_both_directions:
            df = self.interactions.applymap(lambda x: self.int2gene[x])
            df2 = df.copy(deep=True).rename(columns={'Gene_B': 'Gene_A', 'Gene_A': 'Gene_B'})
            return pd.concat([df, df2], axis=0, ignore_index=True)
        else:
            return self.interactions.applymap(lambda x: self.int2gene[x])

    def interactions_as_set(self, return_names=True):
        if return_names:
            df = self.getInteractionNamed()
        else:
            df = self.interactions
        return set(zip(df.Gene_A.values, df.Gene_B.values))

    def getInteractionInts_as_tuple(self, both_directions=False):
        tup = (self.interactions.Gene_A.values, self.interactions.Gene_B.values)

        if both_directions:
            tup = (np.hstack((tup[0], tup[1])), np.hstack((tup[1], tup[0])))

        return tup

    def setEqual(self, network):
        '''
        Convenience function for setting the attributes of a network equal to another network
        '''
        self.interactions = copy.deepcopy(network.interactions)
        self.nodes = copy.deepcopy(network.nodes)

        self.int2gene = copy.deepcopy(network.int2gene)
        self.attr_names = copy.deepcopy(network.attr_names)

    def mapNodeNames(self, map_dict):
        if len(set(self.node_names).difference(set(list(map_dict.keys())))) > 0:
            warnings.warn('The provided mapping does not convert all ids, for these nodes, old IDs will be kept.')

        int2gene = self.int2gene
        self.int2gene = {i: map_dict[name] if (name in map_dict.keys()) else name for i, name in int2gene.items()}

    def subsetNetwork(self, nodes, inplace=True, and_or='and'):
        nodes = set(nodes)
        df = self.getInteractionNamed()
        if and_or == 'and':
            df = df.loc[df.Gene_A.isin(nodes) &
                        df.Gene_B.isin(nodes)]
        else:
            df = df.loc[df.Gene_A.isin(nodes) |
                        df.Gene_B.isin(nodes)]

        return df

    def makeSelfConnected(self, inplace=False):
        self_df = pd.DataFrame({'Gene_A': self.node_names, 'Gene_B': self.node_names})

        if inplace:
            self_df = self_df.applymap(lambda x: self.gene2int[x])
            self.interactions = pd.concat([self.interactions, self_df], ignore_index=True)

        else:
            new_df = pd.concat([self.getInteractionNamed(), self_df], ignore_index=True)
            return new_df

    def mergeNetworks(self, network):
        pass

    def mergedf(self, interaction_df, colnames=None):
        pass

    '''
        get adjacency matrix
    '''

    def getAdjMatrix(self, sort='first', as_df=False):

        row_ids = list(self.interactions['Gene_A'])
        col_ids = list(self.interactions['Gene_B'])

        A = np.zeros((self.N_nodes, self.N_nodes), dtype=np.uint8)
        A[(row_ids, col_ids)] = 1

        if as_df:
            return pd.DataFrame(A, index=self.node_names, columns=self.node_names)
        else:
            return A, np.array(self.node_names)

    def normalizeAdjecencyMatrix(self, symmetric_norm=False):
        adj_array, node_names = self.getAdjMatrix()

        if symmetric_norm:
            D = np.diag(1. / np.sqrt(np.sum(adj_array, axis=0)))
            adj_array_norm = np.dot(np.dot(D, adj_array), D)
        else:
            degree = np.sum(adj_array, axis=0)
            adj_array_norm = (adj_array * 1.0 / degree).T

        return pd.DataFrame(adj_array_norm, index=node_names, columns=node_names)

    '''
        perform kernel diffusion
    '''

    def diffuse(self, kernel='LEX', alpha=0.01, as_df=True, scale=False, self_connected=True,
                symmetric_norm=False):

        A, nodes = self.getAdjMatrix()

        if self_connected:
            np.fill_diagonal(A, np.uint8(1))

        A = A.astype(float)
        starttime = time.time()
        if kernel.upper() == 'LEX':  # TODO insert other diffusion techniques
            A = np.diag(np.sum(A, axis=0)) - A
            # for undirected graphs the axis does not matter, for directed graphs use the in-degree
            A = expm(-alpha * A)

        elif kernel.upper() == 'RWR':
            term1 = (1 - alpha) * A
            term2 = np.identity(A.shape[1]) - alpha * self.normalizeAdjecencyMatrix(
                symmetric_norm=symmetric_norm).values
            term2_inv = np.linalg.inv(term2)
            A = np.dot(term1, term2_inv)

        elif kernel.upper() == 'VANDIN':
            A = np.diag(np.sum(A, axis=0)) - A
            # Adjust diagonal of laplacian matrix by small gamma as seen in Vandin 2011
            A = np.linalg.inv(A + alpha * np.identity(self.N_nodes))
            # status: block tested with the original code

        if scale:
            A = A / np.outer(np.sqrt(np.diag(A)), np.sqrt(np.diag(A)))
        print('Network Propagation Complete: %i seconds' % (time.time() - starttime))
        if as_df:
            df = pd.DataFrame(A, index=nodes, columns=nodes)
            return df
        else:
            return A, nodes

    def propagateMutations(self, mut_df, scale_mutations=False, precomputed_kernel=None, **kernelargs):

        if precomputed_kernel is None:
            K = self.diffuse(as_df=True, **kernelargs)
        else:
            assert isinstance(precomputed_kernel,
                              pd.DataFrame), "Please provide the mutation data as a pandas DataFrame."
            K = precomputed_kernel

        assert isinstance(mut_df, pd.DataFrame), "Please provide the mutation data as a pandas DataFrame."

        mut_genes = mut_df.columns.values
        K_genes = K.columns.values

        common_genes = np.intersect1d(K_genes, mut_genes)

        if len(common_genes) == 0:
            mut_df = mut_df.transpose()
            mut_genes = mut_df.columns.values

            common_genes = np.intersect1d(K_genes, mut_genes)

            if len(common_genes) == 0:
                raise IOError('There are not genes in common between the mutation dataframe and the network kernel.')

        print('There are %i genes in common between the mutation dataset and the network.' % len(common_genes))
        mut_df = mut_df[common_genes]

        if scale_mutations:
            mut_df = mut_df / np.sum(mut_df.values, keepdims=True, axis=1)

        diff_scores = np.matmul(mut_df.values, K.loc[common_genes].values)

        return pd.DataFrame(diff_scores, index=mut_df.index, columns=K.columns.values)

    '''
        check interactions given a list
        find all interactions between the genes in a list
    '''

    def checkInteraction_list(self, gene_list, attribute_separator=None):

        if attribute_separator is not None:
            gene_list = [s.split(attribute_separator)[0] for s in gene_list]

        df = self.getInteractionNamed()
        interactions_df = df.loc[df.Gene_A.isin(gene_list) &
                                 df.Gene_B.isin(gene_list)]
        return Graph(interactions_df)

    '''
        Get shortest path distance
    '''

    def getGeodesicDistance(self, start_genes, stop_genes, nx_Graph=None):
        pass

    def getAdjDict(self, return_names=True):
        pass

    def getEdgeArray(self):
        '''
        Return an edge array, a data structure that allows for
        a very fast retrieval of neighbors.
        :param return_names: Whether the array contains the node names (string) or ids (int)
        :return: edge array. a np array of arrays np.array([arr1, arr2, ..., arrN]) with arr1,
        containing the neighbors of the node with id 1 etc.
        '''
        adj_dict = self.getAdjDict(return_names=False)
        return np.array([np.array(adj_dict[i]) for i in range(self.N_nodes)])

    def getNOrderNeighbors(self, order=2, include_lower_order=True, gene_list=None, return_names=False):

        adj_dict = copy.deepcopy(self.getAdjDict(return_names=return_names))
        orig_dict = self.getAdjDict(return_names=return_names)

        if gene_list is not None:
            adj_dict = {k: v for k, v in adj_dict.items() if k in gene_list}

        for _ in range(order - 1):
            adj_dict = getSecondOrderNeighbors(adj_dict, adj_dict0=orig_dict,
                                               incl_first_order=include_lower_order)
        return adj_dict

    def getDegreeDF(self, return_names=True, set_index=False):
        v, c = np.unique(self.interactions.values.flatten(), return_counts=True)
        if return_names:
            if set_index:
                return pd.DataFrame({'Gene': [self.int2gene[i] for i in v],
                                     'Count': c}, index=[self.int2gene[i] for i in v]).sort_values(by='Count',
                                                                                                   ascending=False,
                                                                                                   inplace=False)
            else:
                return pd.DataFrame({'Gene': [self.int2gene[i] for i in v],
                                     'Count': c}).sort_values(by='Count', ascending=False, inplace=False)
        else:
            if set_index:
                return pd.DataFrame({'Gene': v,
                                     'Count': c}, index=v).sort_values(by='Count', ascending=False, inplace=False)
            else:
                return pd.DataFrame({'Gene': v,
                                     'Count': c}).sort_values(by='Count', ascending=False, inplace=False)

    def removeNodes(self, nodes_tbr, inplace=False):
        nodes_tbr = [self.gene2int[s] for s in nodes_tbr if s in self.gene2int.keys()]
        nodes_tbr = set(nodes_tbr)

        if inplace:
            self.interactions = self.interactions.loc[~(self.interactions.Gene_A.isin(nodes_tbr) |
                                                        self.interactions.Gene_B.isin(nodes_tbr))]
            self.remove_nodes_from_type_dict(nodes_tbr)

        else:
            new_df = self.interactions.loc[~(self.interactions.Gene_A.isin(nodes_tbr) |
                                             self.interactions.Gene_B.isin(nodes_tbr))]
            new_df = new_df.applymap(lambda x: self.int2gene[x])

            return new_df

    def replaceNodesWithInteractions(self, nodes_tbr):
        '''
        Replaces a list of nodes, while connecting all neighbors to each other.
        To be used in pathfinding.
        :param nodes_tbr:
        :return:
        '''

        def get_all_nb_combos(neighbors):

            neighbors = [(nb1, nb2) for i, nb1 in enumerate(neighbors) for nb2 in neighbors[:i]]

            if len(neighbors) > 0:
                neighbors = np.array(neighbors)
                return pd.DataFrame(neighbors, columns=['Gene_A', 'Gene_B'])

            else:
                return None

        df_filtered = self.getInteractionNamed()
        adj_dict = self.getAdjDict()
        adj_dict = {k: np.array(v) for k, v in adj_dict.items()}
        nodes_tbr = set(nodes_tbr)
        new_interactions = []
        for node in nodes_tbr:

            new_interactions_ = get_all_nb_combos(adj_dict[node])
            adj_dict = {k: v[v != node] for k, v in adj_dict.items() if k != node}

            if new_interactions_ is not None:
                new_interactions.append(new_interactions_)

        df_filtered = df_filtered.loc[~(df_filtered.Gene_A.isin(nodes_tbr) |
                                        df_filtered.Gene_B.isin(nodes_tbr))]

        if len(new_interactions) > 0:
            new_interactions = pd.concat(new_interactions, axis=0)
            return pd.concat([df_filtered, new_interactions], axis=0)

        else:
            return df_filtered

    def filterDatasetGenes(self, omicsdatasets, remove_leaves=False, inplace=True):
        '''
        :param: omicsdatasets: datasets that are to be filtered
        :params: should the leaves of the network also be removed?
        :return: the filtered datasets whose genes are all on the network
        '''

        try:
            _ = len(omicsdatasets)

        except TypeError:  # convert to iterable
            omicsdatasets = [omicsdatasets]
            print('converted to iterable')

        if inplace:
            network_genes = set(self.node_names)
            nodes_in_datasets = set()

            for dataset in omicsdatasets:
                intersecting_genes = network_genes.intersection(dataset.genes(as_set=True))

                print('%s: %i genes found on the network.' % (dataset.type, len(intersecting_genes)))
                dataset.subsetGenes(list(intersecting_genes), inplace=True)

                network_genes = nodes_in_datasets.union(network_genes)
                nodes_in_datasets = nodes_in_datasets.union(dataset.genes(as_set=True))

            if remove_leaves:
                self.pruneNetwork(exception_list=nodes_in_datasets, inplace=True)

        else:
            network_genes = set(self.node_names)
            nodes_in_datasets = set()

            datasets_new = []

            for dataset in omicsdatasets:
                intersecting_genes = network_genes.intersection(dataset.genes(as_set=True))

                print('%s: %i genes found on the network.' % (dataset.type, len(intersecting_genes)))
                datasets_new += [dataset.subsetGenes(list(intersecting_genes), inplace=False)]

                network_genes = nodes_in_datasets.union(network_genes)
                nodes_in_datasets = nodes_in_datasets.union(dataset.genes(as_set=True))

            if remove_leaves:
                network = self.pruneNetwork(exception_list=nodes_in_datasets, inplace=False)

            return network, datasets_new

    def filterDataset(self, dataset, remove_leaves=False, inplace=False):

        keeps = list(set(self.node_names).intersection(dataset.genes(as_set=True)))

        if inplace:
            dataset.subsetGenes(keeps, inplace=True)

            if remove_leaves:
                self.pruneNetwork(exception_list=keeps, inplace=True)

        else:
            dataset = dataset.subsetGenes(keeps, inplace=False)
            net = self.deepcopy()
            if remove_leaves:
                net = net.pruneNetwork(exception_list=keeps, inplace=False)

            return net, dataset

    def pruneNetwork(self, exception_list=[], inplace=False):
        '''
        Iteratively prunes the network such that leaves of the network are removed
        '''
        if inplace:
            net = self
        else:
            net = self.deepcopy()

        degreedf = net.getDegreeDF()
        leaves = set(degreedf.Gene[degreedf.Count < 2])
        tbr = leaves.difference(set(exception_list))

        while len(tbr) > 0:
            net.removeNodes(tbr, inplace=True)
            degreedf = net.getDegreeDF()

            leaves = set(degreedf.Gene[degreedf.Count < 2])
            tbr = leaves.difference(exception_list)

        if not inplace:
            return net

    def getEdgeSet(self):
        df = self.getInteractionNamed()
        return set(zip(df['Gene_A'].values, df['Gene_B'].values))

    def getOverlap(self, other_net):
        set1 = self.getEdgeSet()
        set2 = other_net.getEdgeSet()

        min_len = np.minimum(len(set1), len(set2))

        return len(set1.intersection(set2)) / min_len

    def keepLargestComponent(self, verbose=True, inplace=False):

        if self.isConnected:
            print('Graph is connected, returning a copy.')
            return self.deepcopy()

        else:
            components = self.getComponents(return_subgraphs=True)
            largest_subnet = max(components, key=len)

            if verbose:
                print('%i genes from smaller components have been removed.' % (self.N_nodes - largest_subnet.N_nodes))

            if inplace:
                self.setEqual(largest_subnet)
            else:
                return max(components, key=len)

    def getnxGraph(self, return_names=True):
        '''return a graph instance of the networkx module'''
        if return_names:
            df = self.getInteractionNamed()
        else:
            df = self.interactions

        return nx.from_pandas_edgelist(df, source='Gene_A', target='Gene_B')

    def getMinimmumSpanningTree(self, return_names=True):
        A = self.getnxGraph(return_names=return_names)
        T = nx.minimum_spanning_tree(A.to_undirected())

        E = T.edges()
        self_edges = set(self.edge_list(return_names=return_names))
        # assure that the edges are in the same order as the original Graph
        return [e if e in self_edges else tuple(reversed(e)) for e in E]

    def subsample(self, n=100, weighted=False):
        if weighted:
            v, c = np.unique(self.getInteractionNamed().values, return_counts=True)
            genes = np.random.choice(v, size=n, replace=False, p=c / np.sum(c))

        else:
            v = np.unique(self.getInteractionNamed())
            genes = np.random.choice(v, size=n, replace=False)

        subset_df = self.subsetNetwork(genes, inplace=False)

        return subset_df

    def getSimpleRepresentation(self, **kwargs):
        pass

    def get_degree_binning(self, bin_size=100, degree_to_nodes=None, return_names=True):
        '''
        code taken from Network-based in silico drug efficacy screening
        (https://github.com/emreg00/toolbox/blob/master/network_utilities.py)
        :param bin_size:
        :param degree_to_nodes: (optional) a precomputed dict with degrees as keys and node lists as value
        :return:
        '''
        if degree_to_nodes is None:
            degrees = self.getDegreeDF(return_names=return_names)
            unique_degrees = np.unique(degrees.Count.values)
            genes = degrees.index.values
            counts = degrees.Count.values
            degree_to_nodes = {i: list(genes[counts == i]) for i in unique_degrees}

        values = list(degree_to_nodes.keys())
        values.sort()
        bins = []
        i = 0
        while i < len(values):
            low = values[i]
            val = degree_to_nodes[values[i]]
            while len(val) < bin_size:
                i += 1
                if i == len(values):
                    break
                val.extend(degree_to_nodes[values[i]])
            if i == len(values):
                i -= 1
            high = values[i]
            i += 1
            # print i, low, high, len(val)
            if len(val) < bin_size:
                low_, high_, val_ = bins[-1]
                bins[-1] = (low_, high, val_ + val)
            else:
                bins.append((low, high, val))
        return bins

    def calculate_proximity_significance(self, nodes_from, nodes_to, shuffle_strategy='nodes_to',
                                         n_random=1000, min_bin_size=100, seed=452456, measure='d_c'):
        """
        Calculate proximity from nodes_from to nodes_to
        If degree binning or random nodes are not given, they are generated
        lengths: precalculated shortest path length dictionary
        """

        np.random.seed(seed)
        nodes_network = set(self.node_names)

        nodes_from = set(nodes_from) & nodes_network
        nodes_to = set(nodes_to) & nodes_network
        gene2int = self.gene2int

        nodes_from = [gene2int[g] for g in nodes_from]
        nodes_to = [gene2int[g] for g in nodes_to]

        if len(nodes_from) == 0 or len(nodes_to) == 0:
            return None  # At least one of the node group not in network

        nx_Graph = self.getnxGraph(return_names=False)

        d = self.calculate_dist_from_group(nodes_from, nodes_to, measure=measure, nx_Graph=nx_Graph)

        if shuffle_strategy.lower() == 'nodes_to':
            node_to_equivalents = self.get_degree_equivalents(seeds=nodes_to, bin_size=min_bin_size,
                                                              return_names=False)
            nodes_to = select_random_sets(node_to_equivalents, nodes_to, n_random)
            random_dists = [self.calculate_dist_from_group(nodes_from, nodes_to[i, :], nx_Graph=nx_Graph) for i in
                            range(n_random)]

        elif shuffle_strategy.lower() == 'nodes_from':
            node_to_equivalents = self.get_degree_equivalents(seeds=nodes_from, bin_size=min_bin_size,
                                                              return_names=False)
            nodes_from = select_random_sets(node_to_equivalents, nodes_from, n_random)
            random_dists = [self.calculate_dist_from_group(nodes_from[i, :], nodes_to, nx_Graph=nx_Graph) for i in
                            range(n_random)]

        elif shuffle_strategy.lower() == 'both':
            node_to_equivalents = self.get_degree_equivalents(seeds=np.union1d(nodes_to, nodes_from),
                                                              bin_size=min_bin_size,
                                                              return_names=False)
            nodes_to = select_random_sets(node_to_equivalents, nodes_to, n_random)
            nodes_from = select_random_sets(node_to_equivalents, nodes_from, n_random)
            random_dists = [self.calculate_dist_from_group(nodes_from[i, :], nodes_to[i, :], nx_Graph=nx_Graph) for i in
                            range(n_random)]

        else:
            raise IOError('shuffle_strategy not understood.')

        pval = float(sum(random_dists <= d)) / len(random_dists)  # needs high number of n_random
        m, s = np.mean(random_dists), np.std(random_dists)
        if s == 0:
            z = 0.0
        else:
            z = (d - m) / s

        return d, z, (m, s), pval

    def get_degree_equivalents(self, seeds, bin_size=100, return_names=True):

        if return_names:
            seeds = np.intersect1d(seeds, self.node_names)
        else:
            seeds = np.intersect1d(seeds, self.nodes)

        if len(seeds) == 0:
            raise IOError('The seeds do not match the names of the graph nodes.')

        degrees = self.getDegreeDF(return_names=return_names, set_index=True)
        unique_degrees = np.unique(degrees.Count.values)
        genes = degrees.index.values
        counts = degrees.Count.values
        degree_to_nodes = {i: list(genes[counts == i]) for i in unique_degrees}

        bins = self.get_degree_binning(bin_size=bin_size, degree_to_nodes=degree_to_nodes,
                                       return_names=return_names)
        seed_to_nodes = {}

        for seed in seeds:
            d = counts[genes == seed]

            for l, h, nodes in bins:
                if (l <= d) and (h >= d):
                    mod_nodes = list(nodes)
                    mod_nodes.remove(seed)
                    seed_to_nodes[seed] = mod_nodes
                    break

        return seed_to_nodes

    def calculate_dist_from_group(self, nodes_from, nodes_to, measure='d_c', nx_Graph=None):
        dist_mat = self.getGeodesicDistance(nodes_from, nodes_to, nx_Graph=nx_Graph)

        if measure == 'd_c':
            min_dists = np.min(dist_mat.values, axis=1)
            return np.mean(min_dists)

        elif measure == 'd_s':
            mean_dists = np.mean(dist_mat.values, axis=1)
            return np.mean(mean_dists)

    def visualize(self, return_large=False, gene_list=None, edge_df=None, show_labels=False,
                  node_colors=None, cmap='spectral', title=None,
                  color_scheme_nodes=('lightskyblue', 'tab:orange'),
                  color_scheme_edges=('gray', 'tab:green'), labels_dict=None,
                  filename=None, save_path=None):

        """ Visualize the graph
         gene_list = MUST be a list of lists
         labels_dict: a dictionary of dictionaries, containing the labels, fontsizes etc for each group of labels.

         example: {'group1': {'labels': ['G1', 'G2'],
                         font_size:12,
                         font_color:'k',
                         font_family:'sans-serif',
                         font_weight:'normal',
                         alpha:None,
                         bbox:None,
                         horizontalalignment:'center',
                         verticalalignment:'center'}}

        note that the name of the keys is not used.
         """

        if gene_list is not None:
            assert len(gene_list) == len(color_scheme_nodes) - 1, \
                "ERROR number of gene lists provided must match the color scheme for nodes"

        if (not return_large) and (len(self.nodes) > 500):
            raise IOError('The graph contains more than 500 nodes, if you want to plot this specify return_large=True.')

        G = self.getnxGraph()
        if (gene_list is None) and (node_colors is None):
            node_colors = color_scheme_nodes[0]
        elif node_colors is None:
            additional_gl = set.intersection(*[set(i) for i in gene_list])
            if additional_gl:
                gene_list = [set(gl) - additional_gl for gl in gene_list]
                gene_list.append(additional_gl)
                color_scheme_nodes += ("tab:purple",)
            node_colors = []
            for i, gl in enumerate(gene_list):
                node_colors.append([color_scheme_nodes[i + 1] if node in gl else "" for node in G.nodes])
            node_colors = list(map(''.join, zip(*node_colors)))
            node_colors = [i if i else color_scheme_nodes[0] for i in node_colors]
            # node_colors = [color_scheme_nodes[1] if node in gene_list else color_scheme_nodes[0] for node in G.nodes]
        elif isinstance(node_colors, dict):
            node_colors = [node_colors[i] for i in G.nodes]

        # assert len(G.nodes) == len(node_colors), "ERROR number of node colors does not match size of graph"

        if all(isinstance(c, (int, float)) for c in node_colors):  # perform rescaling in case of floats for cmap
            node_colors = np.array(node_colors)
            node_colors = (node_colors - np.min(node_colors)) / (np.max(node_colors) - np.min(node_colors))

        if edge_df is not None:
            edges = list(G.edges())
            edge_list = [tuple(pair) for pair in edge_df.values]

            edge_color = [color_scheme_edges[1] if edge in edge_list else color_scheme_edges[0] for edge in edges]
            edge_thickness = [2 if edge in edge_list else 1 for edge in edges]

        else:
            edge_color = color_scheme_edges[0]
            edge_thickness = 1.

        plt.figure()
        # TODO: make this prettier
        if title is not None:
            plt.title(title)

        if labels_dict is None:
            nx.draw(G, with_labels=show_labels,
                    node_size=2e2, node_color=node_colors, edge_color=edge_color, width=edge_thickness, cmap=cmap)

        else:
            pos = nx.drawing.spring_layout(G)  # default to spring layout

            nx.draw(G, pos=pos, with_labels=False,
                    node_size=2e2, node_color=node_colors,
                    edge_color=edge_color, width=edge_thickness, cmap=cmap)

            for label_kwds in labels_dict.values():
                nx.draw_networkx_labels(G, pos, **label_kwds)

        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = [20, 15]

        if filename:
            plt.savefig(save_path + filename + '.png')
            plt.close()
        else:
            plt.show()

    def plotRepresentation(self, gene_list=None, node_colors=None, cmap='spectral', precomputed_representations=None,
                           change_size=True, **kwargs):
        '''
        Function to plot the network without the edges, resulting in a much faster visualization, making it possible to visualize larger graphs
        :param gene_list: list of genes to be plotted
        :param node_colors: color of the nodes can be strings with color or floats
        :param node_size:
        :param: cmap:
        :param: precomputed_representations:
        :return:
        '''
        size = None

        if node_colors is None:
            node_colors = 'lightskyblue'

        if all(isinstance(c, (int, float)) for c in node_colors):  # perform rescaling in case of floats for cmap
            node_colors = np.array(node_colors)
            node_colors = (node_colors - np.min(node_colors)) / (np.max(node_colors) - np.min(node_colors))

            if change_size:
                size = 50 * node_colors

        if precomputed_representations is not None:
            coords = precomputed_representations
        else:
            coords = self.getSimpleRepresentation(dim=2)

        if gene_list is not None:
            coords = coords.loc[gene_list]

        fig, ax = plt.subplots()
        ax.scatter(coords.iloc[:, 0], coords.iloc[:, 1], c=node_colors, s=size, **kwargs)
        ax.set_visible('off')
        plt.show()

    def degreePreservingPermutation(self, N_swaps=1000, random_state=42):
        np.random.seed(42)
        adj_dict = self.getAdjDict(return_names=False)
        adj_dict = {k: np.array(v) for k, v in adj_dict.items()}

        weights = self.getDegreeDF(return_names=False).set_index('Gene')['Count']
        weights = 1. * weights.loc[np.arange(self.N_nodes)].values / np.sum(weights.values)
        n_swaps = 0
        visited_pairs = []

        while n_swaps < N_swaps:
            gene1, gene2 = np.random.choice(self.N_nodes, 2, replace=False, p=weights)

            if (gene1, gene2) not in visited_pairs:
                nb_gene1, nb_gene2 = copy.deepcopy(adj_dict[gene1]), copy.deepcopy(adj_dict[gene2])

                connected = (gene1 in nb_gene2) | (gene2 in nb_gene1)  # TODO: check if this works for Directed Graphs

                if connected:
                    nb_gene1 = np.append(nb_gene1, gene1)
                    nb_gene2 = np.append(nb_gene2, gene2)

                overlap = np.intersect1d(nb_gene1, nb_gene2)

                if (len(nb_gene2) > len(overlap)) & (len(nb_gene1) > len(overlap)):

                    if len(nb_gene1) < len(nb_gene2):  # make sure nb1 has the most neighbors
                        t_ = nb_gene2
                        nb_gene2 = nb_gene1
                        nb_gene1 = t_

                        t_ = gene2
                        gene2 = gene1
                        gene1 = t_

                    diff = np.setdiff1d(nb_gene1, nb_gene2)  # append gene1 in case gene1 -- gene2

                    n_swapped_genes = len(nb_gene2) - len(overlap)

                    random_ids = np.random.choice(len(diff), n_swapped_genes, replace=False)
                    one_to_two = diff[random_ids]
                    two_to_one = np.setdiff1d(nb_gene2, nb_gene1)

                    arr2 = np.union1d(overlap, one_to_two)
                    arr1 = np.union1d(nb_gene2, np.delete(diff, random_ids))

                    if connected:
                        arr2 = arr2[arr2 != gene2]
                        arr1 = arr1[arr1 != gene1]

                    adj_dict[gene2] = copy.deepcopy(arr2)
                    adj_dict[gene1] = copy.deepcopy(arr1)

                    for a in one_to_two:
                        adj_dict[a][adj_dict[a] == gene1] = gene2

                    for a in two_to_one:
                        adj_dict[a][adj_dict[a] == gene2] = gene1

                    visited_pairs.append((gene1, gene2))
                    visited_pairs.append((gene2, gene1))
                    n_swaps += 1

        df = adj_dict_to_df(adj_dict)
        df = df.applymap(lambda x: self.int2gene[x])

        return df

    # def getTrainTestPairs_MStree(self, train_ratio=0.7, train_validation_ratio=0.7,
    #                              excluded_sets=None, neg_pos_ratio=5, check_training_set=True, random_state=42):
    #     pass

    def getTrainTestData(self, train_ratio=0.7, neg_pos_ratio=5, train_validation_ratio=None, excluded_sets=None,
                         return_summary=True, random_state=42, balanced=False, include_negatives=None, verbose=True):
        '''
        :param: train_ratio: The fraction of samples used for training
        :param: neg_pos_ratio: The ratio of negative examples to positive examples
        :param: excluded_negatives: should be a set of tuples with negatives interactions to exclude
        :param: method: The sampling method used for generating the pairs:
                - ms_tree: uses a minimum spanning tree to find at least one positive pair for each node
                - balanced: draws approximately (neg_pos_ratio * n_positives) negatives for each gene
        :return: positive and negative pairs for both train and test set (4 lists in total)
        '''

        if balanced:
            # TODO: balanced with validation split
            print("Using balanced sampling")
            pos_train, neg_train, \
            pos_test, neg_test, summary_df = self.getTrainTestPairs_Balanced(neg_pos_ratio=neg_pos_ratio,
                                                                             train_ratio=train_ratio,
                                                                             check_training_set=True,
                                                                             random_state=random_state,
                                                                             include_negatives=include_negatives,
                                                                             verbose=verbose)
            pos_val, neg_val = [], []
        else:
            print("Using unbalanced sampling")
            pos_train, neg_train, pos_val, neg_val, \
            pos_test, neg_test, summary_df = self.getTrainTestPairs_MStree(train_ratio=train_ratio,
                                                                           train_validation_ratio=train_validation_ratio,
                                                                           excluded_sets=excluded_sets,
                                                                           neg_pos_ratio=neg_pos_ratio,
                                                                           check_training_set=True,
                                                                           random_state=random_state)

        assert len(pos_train) == len(set(pos_train)), "getTrainTestPairs_MStree: Duplicate pos train"
        assert len(pos_test) == len(set(pos_test)), "getTrainTestPairs_MStree: Duplicate pos test"

        assert len(neg_train) == len(set(neg_train)), "getTrainTestPairs_MStree: Duplicate neg train"
        assert len(neg_test) == len(set(neg_test)), "getTrainTestPairs_MStree: Duplicate neg test"

        assert not set(pos_train) & set(pos_test), "getTrainTestPairs_MStree: Overlap pos train - test"
        assert not set(neg_train) & set(neg_test), "getTrainTestPairs_MStree: Overlap neg train - test"

        if len(pos_val) > 1:
            assert not set(pos_val) & set(pos_test), "getTrainTestPairs_MStree: Overlap pos val - test"
            assert not set(neg_val) & set(neg_test), "getTrainTestPairs_MStree: Overlap neg val - test"

        X_train = np.array(pos_train + neg_train)
        X_val = np.array(pos_val + neg_val)
        X_test = np.array(pos_test + neg_test)

        Y_train = np.array([1 for _ in range(len(pos_train))] + [0 for _ in range(len(neg_train))])
        Y_val = np.array([1 for _ in range(len(pos_val))] + [0 for _ in range(len(neg_val))])
        Y_test = np.array([1 for _ in range(len(pos_test))] + [0 for _ in range(len(neg_test))])

        if return_summary:
            if train_validation_ratio is None:
                return X_train, X_test, Y_train, Y_test, summary_df
            else:
                return X_train, X_val, X_test, Y_train, Y_val, Y_test, summary_df
        else:
            if train_validation_ratio is None:
                return X_train, X_test, Y_train, Y_test
            else:
                return X_train, X_val, X_test, Y_train, Y_val, Y_test


FITTERMAPPER = {'logistic_classifier': LogisticRegression(solver='lbfgs'),
                'random_forest': RandomForestClassifier(),
                'sgd_classifier': SGDClassifier(loss='log')}


def permuteVector(v, N=100, gene_lengths=None):
    if gene_lengths is None:
        return np.array([np.random.permutation(np.array(v)) for _ in range(N)])

    else:
        probs = gene_lengths / np.sum(gene_lengths)
        random_vals = []

        for _ in range(N):
            zeros = np.zeros(v.shape)
            rand_ids = np.random.choice(np.arange(len(v)), replace=False, p=probs)
            zeros[rand_ids] = 1
            random_vals.append(zeros)

        return np.array(zeros)


def ZscorePerSample_(sample_vec, K_df, eps=1e-15, N=100):
    random_mat = permuteVector(sample_vec, N=N)
    random_vals = np.matmul(random_mat, K_df.values)
    true_val = np.matmul(sample_vec, K_df.values)

    return (true_val - np.mean(random_vals, axis=0)) / (np.std(random_vals, axis=0) + eps)


def calculateZscores(mut_df, Kernel_df, N=100, eps=1e-15, n_jobs=1, random_state=42):
    '''
    Calculate the Z-score of the diffused values, by randomly shuffling the mutations over the network
    :param mut_df: a dataframe containing mutations (VAFs, scores, ...) samples x genes
    :param Kernel_df: A kernel giving the distances between genes
    :param N: the number of permutations used to calculate the Z-score
    :param eps: a small number to prevent infs or NaNs when dividing by the standard deviation
    :return: a Dataframe (samples x genes) containing the Z-scores.
    '''
    np.random.seed(random_state)

    samples = list(mut_df.index)

    common_genes = np.intersect1d(mut_df.columns.values, Kernel_df.columns.values)
    assert len(common_genes) > 0, IOError('The kernel and the data have no genes in common.')

    mut_df = mut_df[common_genes]
    Kernel_df = Kernel_df.loc[common_genes]

    if n_jobs == 1:
        z_scores = [ZscorePerSample_(mut_df.loc[sample].values, Kernel_df, eps=eps, N=N) for sample in samples]

    else:
        z_scores = Parallel(n_jobs=n_jobs)(delayed(ZscorePerSample_)(mut_df.loc[sample].values, Kernel_df, eps=eps, N=N)
                                           for sample in samples)
    return pd.DataFrame(np.array(z_scores), index=samples, columns=Kernel_df.columns)


def getDriverScore(z_score_df, mut_df, M, z_thresh=3):
    # calculates which mutations are drivers in a sample based
    # on the contribution of each variant to high Z-score regions

    z_genes_dict = z_score_df.apply(lambda x: list(x[x > z_thresh].index), axis=1).to_dict()
    mut_genes = mut_df.apply(lambda x: list(x[x > 0].index), axis=1).to_dict()
    driver_scores = pd.DataFrame(np.zeros(mut_df.shape), index=mut_df.index, columns=mut_df.columns)

    for sample, z_genes in z_genes_dict.items():
        driver_scores.loc[sample, mut_genes[sample]] = M.loc[mut_genes[sample], z_genes].sum(axis=1).values

    return driver_scores


def ind2sub(X, nrows):
    X = np.array(X, dtype=np.uint64)
    col_ids, row_ids = np.divmod(X, nrows)

    return row_ids.astype(np.uint16), col_ids.astype(np.uint16)


def sub2ind(X, nrows):
    X = np.array(X)
    ind = X[:, 0] + X[:, 1] * nrows

    return ind


def positives_split(interaction_df, all_pos_pairs=None, train_ratio=0.7, train_validation_ratio=None):
    # First start with the positives
    if all_pos_pairs is None:
        all_pos_pairs = set(zip(interaction_df.Gene_A, interaction_df.Gene_B))
    N_edges = len(all_pos_pairs)
    min_tree = nx.minimum_spanning_tree(nx.from_pandas_edgelist(interaction_df, source='Gene_A', target='Gene_B')).edges

    pos_samples_train = set([tuple(sorted(tup)) for tup in list(min_tree)])
    all_pos_pairs = list(all_pos_pairs.difference(pos_samples_train))

    # determine how much samples need to drawn from the remaining positive pairs to achieve the train_test ratio
    still_needed_samples = np.maximum(0, np.round(train_ratio * N_edges - len(pos_samples_train)).astype(np.int_))


    if still_needed_samples == 0:
        print('The train ratio has been increased to include every node in the training set.')

    ids_train = np.random.choice(len(all_pos_pairs), still_needed_samples, replace=False)
    ids_test = np.setdiff1d(np.arange(len(all_pos_pairs)), ids_train)

    pos_samples_train = list(pos_samples_train) + [all_pos_pairs[i] for i in ids_train]
    pos_samples_test = [all_pos_pairs[i] for i in ids_test]

    if train_validation_ratio is not None:
        # pdb.set_trace()
        min_tree_val = nx.minimum_spanning_tree(
            nx.from_pandas_edgelist(pd.DataFrame(pos_samples_train, columns=['Gene_A', 'Gene_B']), source='Gene_A',
                                    target='Gene_B')).edges
        pos_samples_train_valid = set([tuple(sorted(tup)) for tup in list(min_tree_val)])
        all_pos_pairs_valid = list(set(pos_samples_train).difference(pos_samples_train_valid))
        N_edges_valid = len(all_pos_pairs_valid)
        still_needed_samples_valid = np.maximum(0, np.round(train_validation_ratio * N_edges_valid - len(pos_samples_train_valid)).astype(np.int_))
        ids_train_valid = np.random.choice(len(all_pos_pairs_valid), still_needed_samples_valid, replace=False)
        ids_test_valid = np.setdiff1d(np.arange(len(all_pos_pairs_valid)), ids_train_valid)

        pos_samples_train = list(pos_samples_train_valid) + [all_pos_pairs_valid[i] for i in ids_train_valid]
        pos_samples_valid = [all_pos_pairs_valid[i] for i in ids_test_valid]

    else:
        pos_samples_valid = []

    return pos_samples_train, pos_samples_valid, pos_samples_test


def checkTrainingSetsPairs(X_train, Y_train, X_test, Y_test):
    '''
    :param X_train: a (train_samples, 2) np array
    :param Y_train: a (train_samples, ) np array
    :param X_test: a (test_samples, 2) np array
    :param Y_test: a (test_samples, ) np array
    :return: some statistics on the test and train set
    '''

    # First we check whether every pair in the training and the testing set is unique
    X_train_pairs = set(zip(X_train[:, 0], X_train[:, 1]))
    assert X_train.shape[0] == len(X_train_pairs), 'The training set contains non-unique entries.'
    X_test_pairs = set(zip(X_test[:, 0], X_test[:, 1]))
    assert X_test.shape[0] == len(X_test_pairs), 'The test set contains non-unique entries.'

    # Then we check for data leakage
    assert len(X_train_pairs.intersection(X_test_pairs)) == 0, 'Some gene pairs occur in both training and testing set.'

    # We also check if the ratio of the labels is comparable
    print('Positive-Negtative ratio for the training set: %f' % (sum(Y_train) / len(Y_train)))
    print('Positive-Negtative ratio for the test set: %f' % (sum(Y_test) / len(Y_test)))


def filter_graph_by_LP_swaps(prob_matrix, current_interactions, adj_dict=None, n_swaps=1000, strict=False,
                             max_attempts=100000):
    """
    Filters a graph by performing LP swaps on the graph
    This assures that the degree distribution of the graph remains unchanged.
    The algorithm starts by finding the lowest probability edge and performs a swap,
    i.e. both interaction node get another interaction with a different pair of nodes that had
    an interaction as well. This is also removed such that the degree of each nodes remains the same:

    O-----O          O     O
                ->   |     |
    O-----O          O     O

    The swaps are optimal, in the sense that this function will only allow a swap such that

    :param prob_matrix: a n_node x n_node np array that contains the probability for each possible edge
    :param current_interactions: a tuple containing the row and col indices of the current interaction in the prob_matrix
    :param n_attempts: the number of swaps that are attempted
    :return:
    """

    # if adj_dict is None:
    #    df = pd.DataFrame(np.transpose(np.array(current_interactions)), columns=['Gene_A', 'Gene_B'])
    #    adj_dict = to_dict_of_lists(nx.from_pandas_edgelist(df, source='Gene_A', target='Gene_B'))

    n_successes = 0
    start = time.time()

    probs_original_edges = prob_matrix[current_interactions]
    sorted_id = np.argsort(probs_original_edges)

    r_is, c_is = current_interactions[0][sorted_id], current_interactions[1][sorted_id]
    # probs_current_edges = probs_current_edges[sorted_id]

    id, attempt = 0, 0
    n_interactions = len(r_is)
    n_swaps = np.minimum(n_swaps, n_interactions)

    while (n_successes < n_swaps) & (attempt < max_attempts):
        attempt += 1
        r_i, c_i = r_is[id], c_is[id]
        # TODO: make this faster by just checking the probability of the new edges
        swapped, current_interactions, new = perform_single_LP_swap(prob_matrix,
                                                                    current_interactions[0],
                                                                    current_interactions[1],
                                                                    r_i=r_i,
                                                                    c_i=c_i,
                                                                    strict=strict)

        if swapped:
            # TODO think of strategy to drop back to lower id
            # id = 0 this seems to work less well
            id += 1
            probs_current_edges = prob_matrix[current_interactions]
            sorted_id = np.argsort(probs_current_edges)

            r_is, c_is = current_interactions[0][sorted_id], current_interactions[1][sorted_id]

            # probs_current_edges = probs_current_edges[sorted_id]
            # probs_current_edges = np.delete(probs_current_edges, id, 0)
            # new_probs = prob_matrix[new]

            # ii = np.searchsorted(probs_current_edges, new_probs)
            # probs_current_edges = np.insert(probs_current_edges, ii, new_probs)

        else:
            id += 1

        if id == (n_interactions - 1):
            id = 0  # perform a complete restart if all edges have been investigated once

        n_successes += swapped

    stop = time.time()

    print('Finished swaps!')
    print('Elapsed time: %f' % (stop - start))
    print('Number of attempted swaps: %i' % attempt)
    print('Number of succesful swaps: %i' % n_successes)
    print('Total probability increase: %f' % np.sum(prob_matrix[(current_interactions[0], current_interactions[1])]
                                                    - probs_original_edges))

    return current_interactions


def perform_single_LP_swap(prob_matrix, current_rows, current_cols, r_i, c_i, strict=False):
    '''
    Optimal only for undirected graphs
    :param prob_matrix:
    :param current_rows: the row indices of the current interactions in prob_matrix
    :param current_cols: the row indices of the current interactions in prob_matrix
    :param r_i: the row index of the edge for which a swap is attempted
    :param c_i: the col index of the edge for which a swap is attempted
    :param i: the index r_i and c_i have in current_rows and current_cols
    :return: the indices of the node that should be swapped with
    '''

    swapped = False
    edge_index_mask = (current_rows == r_i) & (current_cols == c_i)
    # these can be precomputed in an adj dict, but then the filter function needs to be modified
    nb_ri = np.hstack(
        (current_cols[current_rows == r_i], current_rows[current_cols == r_i], [r_i]))  # for undirected graphs
    nb_ci = np.hstack(
        (current_cols[current_rows == c_i], current_rows[current_cols == c_i], [c_i]))  # for undirected graphs

    assert np.sum(edge_index_mask) == 1, 'Please make sure that c_i and r_i correspond to existing nodes in the graph!'

    lowest_prob = prob_matrix[r_i, c_i]

    pot_c = np.arange(prob_matrix.shape[1])[prob_matrix[r_i, :] > lowest_prob]
    pot_c = np.setdiff1d(pot_c, nb_ri)
    mask = np.isin(current_cols, pot_c)

    r_pot, c_pot = current_rows[mask], current_cols[mask]

    mask = ~ np.isin(r_pot, nb_ci)  # this might give problems for directed graphs
    r_pot, c_pot = r_pot[mask], c_pot[mask]

    # optionally impose the additional constraint that the other edge should also benefit from the swap
    if strict:
        mask = prob_matrix[r_pot, c_i] > prob_matrix[(r_pot, c_pot)]
        r_pot, c_pot = r_pot[mask], c_pot[mask]

    if len(r_pot) > 0:
        new_probs = prob_matrix[r_i, c_pot] + prob_matrix[r_pot, c_i]
        old_probs = prob_matrix[r_i, c_i] + prob_matrix[(r_pot, c_pot)]

        score = new_probs - old_probs
        max_score_id = np.argmax(score)
        max_score = score[max_score_id]

        if max_score > 0:
            swapped = True
            other_edge_tbr_row, other_edge_tbr_col = r_pot[max_score_id], c_pot[max_score_id]

            new_ints = {(r_i, other_edge_tbr_col), (other_edge_tbr_col, r_i),
                        (c_i, other_edge_tbr_row), (other_edge_tbr_row, c_i)}
            # change this for directed graphs f!!!!!!!!!!!!!!
            olds_int = {(r_i, c_i), (c_i, r_i),
                        (other_edge_tbr_row, other_edge_tbr_col),
                        (other_edge_tbr_col, other_edge_tbr_row)}

            new_interactions = set(zip(current_rows, current_cols)).difference(olds_int) \
                .union(new_ints)

            new_interactions = tuple(zip(*list(new_interactions)))

            print('Swapped %s with %s to increase the score with %f.' % ((r_i, c_i),
                                                                         (other_edge_tbr_row, other_edge_tbr_col),
                                                                         max_score))

            return swapped, (np.array(new_interactions[0]), np.array(new_interactions[1])), \
                   (np.array([r_i, other_edge_tbr_row]), np.array([other_edge_tbr_row, c_i]))

        else:
            print('No swaps found.')
            return swapped, (current_rows, current_cols), None

    else:
        print('No swaps found.')
        return swapped, (current_rows, current_cols), None


def filter_graph_by_LP_swaps_fastest(prob_matrix, A, current_interactions, n_swaps=1000, strict=False,
                                     max_attempts=100000):
    """
    Filters a graph by performing LP swaps on the graph
    This assures that the degree distribution of the graph remains unchanged.
    The algorithm starts by finding the lowest probability edge and performs a swap,
    i.e. both interaction node get another interaction with a different pair of nodes that had
    an interaction as well. This is also removed such that the degree of each nodes remains the same:

    O-----O          O     O
                ->   |     |
    O-----O          O     O

    The swaps are optimal, in the sense that this function will only allow a swap such that

    :param prob_matrix: a n_node x n_node np array that contains the probability for each possible edge
    :param A: the adjacency matrix
    :param current_interactions: a tuple containing the row and col indices of the current interaction in the prob_matrix
    :param n_attempts: the number of swaps that are attempted
    :return:
    """

    n_successes = 0
    start = time.time()

    probs_current_edges = prob_matrix[current_interactions]
    original_likelihood = np.sum(probs_current_edges)

    sorted_id = np.argsort(probs_current_edges)
    probs_current_edges = probs_current_edges[sorted_id]
    r_is, c_is = current_interactions[0][sorted_id], current_interactions[1][sorted_id]

    id, attempt = 0, 0
    n_interactions = len(current_interactions[0])

    n_swaps = np.minimum(n_swaps, n_interactions)

    while (n_successes < n_swaps) & (attempt < max_attempts):
        attempt += 1
        r_i, c_i = r_is[id], c_is[id]

        t_ = (np.hstack((c_is, r_is)), np.hstack((r_is, c_is)))
        swapped, old_int, new_int, edge_tbr_id = _perform_single_LP_swap_fastest(prob_matrix,
                                                                                 A,
                                                                                 r_i=r_i,
                                                                                 c_i=c_i,
                                                                                 current_interactions=t_,
                                                                                 strict=strict)

        if swapped:
            new_probs = prob_matrix[new_int]

            ii = np.array([id, edge_tbr_id])
            probs_current_edges = np.delete(probs_current_edges, ii)
            r_is, c_is = np.delete(r_is, ii), np.delete(c_is, ii)

            ii = np.searchsorted(probs_current_edges, new_probs)
            probs_current_edges = np.insert(probs_current_edges, ii, new_probs)
            r_is, c_is = np.insert(r_is, ii, new_int[0]), np.insert(c_is, ii, new_int[1])

        id += 1

        if id == (n_interactions - 1):
            id = 0  # perform a complete restart if all edges have been investigated once

        n_successes += swapped

    stop = time.time()

    print('Finished swaps!')
    print('Elapsed time: %f' % (stop - start))
    print('Number of attempted swaps: %i' % attempt)
    print('Number of successful swaps: %i' % n_successes)
    print('Total probability increase: %f' % (np.sum(prob_matrix[(r_is, c_is)]) - original_likelihood))

    return r_is, c_is


def _perform_single_LP_swap_fastest(prob_matrix, A, current_interactions, r_i, c_i, strict=False):
    '''
    Only for undirected graphs
    :param prob_matrix:
    :param current_rows: the row indices of the current interactions in prob_matrix
    :param current_cols: the row indices of the current interactions in prob_matrix
    :param r_i: the row index of the edge for which a swap is attempted
    :param c_i: the col index of the edge for which a swap is attempted
    :param i: the index r_i and c_i have in current_rows and current_cols
    :return: the indices of the node that should be swapped with
    '''

    swapped = False
    edge_tbr_id = np.arange(len(current_interactions[0]))

    assert A[r_i, c_i] == 1, 'Please make sure that c_i and r_i correspond to existing nodes in the graph!'

    lowest_prob = prob_matrix[r_i, c_i]
    all_node_ints = np.arange(prob_matrix.shape[1])

    c_pot = all_node_ints[
        (prob_matrix[r_i, :] > lowest_prob) & (A[r_i, :] == 0)]  # make the Adjecency matrix self-connected
    r_pot = all_node_ints[(prob_matrix[:, c_i] > lowest_prob) & (A[:, c_i] == 0)]

    # only for undirected graphs
    mask = np.isin(current_interactions[0], r_pot) & np.isin(current_interactions[1], c_pot)

    r_pot, c_pot = current_interactions[0][mask], current_interactions[1][mask]
    edge_tbr_id = edge_tbr_id[mask]

    # optionally impose the additional constraint that the other edge should also benefit from the swap
    if strict:
        mask = prob_matrix[r_pot, c_i] > prob_matrix[(r_pot, c_pot)]
        r_pot, c_pot = r_pot[mask], c_pot[mask]
        edge_tbr_id = edge_tbr_id[mask]

    if len(r_pot) > 0:
        new_probs = prob_matrix[r_i, c_pot] + prob_matrix[r_pot, c_i]
        old_probs = prob_matrix[r_i, c_i] + prob_matrix[(r_pot, c_pot)]

        score = new_probs - old_probs
        max_score_id = np.argmax(score)
        max_score = score[max_score_id]
        edge_tbr_id = edge_tbr_id[max_score_id]

        if max_score > 0:
            swapped = True
            other_edge_tbr_row, other_edge_tbr_col = r_pot[max_score_id], c_pot[max_score_id]

            new_ints = (np.array([np.minimum(r_i, other_edge_tbr_col), np.minimum(c_i, other_edge_tbr_row)]),
                        np.array([np.maximum(r_i, other_edge_tbr_col), np.maximum(c_i, other_edge_tbr_row)]))

            old_ints = (np.array([r_i, np.minimum(other_edge_tbr_row, other_edge_tbr_col)]),
                        np.array([c_i, np.maximum(other_edge_tbr_row, other_edge_tbr_col)]))

            A[new_ints] = 1
            A[(new_ints[1], new_ints[0])] = 1

            A[old_ints] = 0
            A[(old_ints[1], old_ints[0])] = 0

            print('Swapped %s with %s to increase the score with %f.' % ((r_i, c_i),
                                                                         (other_edge_tbr_row, other_edge_tbr_col),
                                                                         max_score))

            return swapped, old_ints, new_ints, np.mod(edge_tbr_id, len(current_interactions[0]) / 2).astype(int)

        else:
            print('No swaps found.')
            return swapped, None, None, None

    else:
        print('No swaps found.')
        return swapped, None, None, None


def from_df(self, df, weighted=False, directed=False, weights_col=None):
    self.G = nx.DiGraph()
    src_col = df.columns[0]
    dst_col = df.columns[1]
    if directed:
        def read_weighted(src, dst, w):
            self.G.add_edge(src, dst)
            self.G[src][dst]['weight'] = w

    else:
        def read_weighted(src, dst, w):
            self.G.add_edge(src, dst)
            self.G.add_edge(dst, src)
            self.G[src][dst]['weight'] = w
            self.G[dst][src]['weight'] = w

    if weights_col is None:
        weights = [1.0 for row in range(df.shape[0])]
    else:
        try:
            weights = df[weights_col].values.astype(float)
        except:
            raise IOError('The weight column is not known.')

    for src, dst, w in zip(df[src_col].values, df[dst_col].values, weights):
        read_weighted(src, dst, w)

    self.encode_node()


def adj_dict_to_df(adj_dict):
    data = np.array([(k, v) for k, vals in adj_dict.items() for v in vals])
    return pd.DataFrame(data, columns=['Gene_A', 'Gene_B'])


def getSecondOrderNeighbors(adj_dict, adj_dict0=None, incl_first_order=True):
    # slwo
    if adj_dict0 is None:
        adj_dict0 = adj_dict

    if incl_first_order:
        return {k: set([l for v_i in list(v) + [k] for l in adj_dict0[v_i]]) for k, v in adj_dict.items()}
    else:
        return {k: set([l for v_i in v for l in adj_dict0[v_i]]) for k, v in adj_dict.items()}


def extractIDsfromBioGRID(path_df, one2one=True):
    '''
    Creates a map (dictionary) between entrez and gene symbols (including aliases) from a BioGRID interaction file.
    :param path_df: path or dataframe containing the interaction data from BioGRID
    :param one2one: whether the resulting map should map str to str or str to tuple
    :return:
    '''

    if isinstance(path_df, str):
        biogrid_df = pd.read_csv(path_df, sep='\t', header=0)

    elif isinstance(path_df, pd.DataFrame):
        biogrid_df = path_df

    else:
        raise IOError('Input type not understood.')

    try:
        biogrid_df = biogrid_df[['Official Symbol Interactor A', 'Official Symbol Interactor B', 'Synonyms Interactor A',
                                 'Synonyms Interactor B', 'Entrez Gene Interactor A', 'Entrez Gene Interactor B']]

    except KeyError:
        raise IOError('The dataframe does not contain the BioGRID column names.')

    # make the entrez to symbol map:
    map_df = pd.DataFrame({'Entrez': biogrid_df['Entrez Gene Interactor A'].append(biogrid_df['Entrez Gene Interactor B']),
                        'Gene Symbol': biogrid_df['Official Symbol Interactor A'].append(biogrid_df['Official Symbol Interactor B'])})

    map_df = map_df.drop_duplicates()
    map_dict = dict(zip(map_df.Entrez, map_df['Gene Symbol']))

    # to make tuple maps
    # symbols = biogrid_df['Official Symbol Interactor A'].append(biogrid_df['Official Symbol Interactor B'])
    # aliases = biogrid_df['Synonyms Interactor A'].append(biogrid_df['Synonyms Interactor B'])

    # tuples = (symbols.astype(str) + '|' + aliases.astype(str)).apply(lambda x: tuple(x.replace('|-', '').split('|')))

    return map_dict


def select_random_sets(equivalency_dict, node_group, nsets, seed=42):
    '''
    draws samples from the equivalency dict such that we end up with nsets sets.
    Each set contains unique elements sampled from the equivalency dict.
    :param equivalency_dict: a dict with elements from nsets as key and valid replacements as values (list)
    :param node_group:
    :param nsets: the number of sets available
    :return: an (len(node_group, nsets) np.array
    '''

    np.random.seed(seed)

    group_size = len(node_group)
    bin_sizes = [len(v) for v in equivalency_dict.values()]
    min_bin_size = min(bin_sizes)
    most_frequent_bin, max_bin_freq = np.unique(bin_sizes, return_counts=True)
    max_bin_freq = max_bin_freq[np.argmax(most_frequent_bin)]

    #p_unique = np.prod(np.arange(min_bin_size, min_bin_size - max_bin_freq, -1)/min_bin_size)
    #p_unique = (min_bin_size - group_size + 1)/(min_bin_size - 1)
    #ndraws = np.int_(nsets / p_unique * 1.1)


    rand_sets = np.array([np.random.choice(equivalency_dict[g], nsets, replace=True)
                          for g in node_group])

    rand_sets = np.transpose(rand_sets)
    mask = np.array([len(np.unique(gs)) == group_size for gs in rand_sets])
    rand_sets = rand_sets[mask]

    while rand_sets.shape[0] < nsets:
        still_needed = 2*(nsets - rand_sets.shape[0])
        rand_sets_ = np.array([np.random.choice(equivalency_dict[g], still_needed, replace=True)
                              for g in node_group])

        rand_sets_ = np.transpose(rand_sets_)
        mask = np.array([len(np.unique(gs)) == group_size for gs in rand_sets])
        rand_sets = rand_sets[mask]
        rand_sets = np.vstack((rand_sets, rand_sets_))

    return rand_sets[:nsets, :]


def check_node_dict(net_node_names, node_dict, type_dict=""):
    """
    expects a dict where keys = nodes in the network, values are the corresponding types
    """

    nodes = list(node_dict.keys())

    nodes_missing_in_dict = set(net_node_names) - set(nodes)

    if len(nodes_missing_in_dict) > 0:
        print("The following genes have no annotation:")
        print(nodes_missing_in_dict)

        raise IOError("There are genes that are missing in the %s dict." %type_dict)

    nodes_missing_in_network = set(nodes) - set(net_node_names)

    if len(nodes_missing_in_network) > 0:
        # print("The following genes are missing in the network and will be removed:")
        # print(nodes_missing_in_network)
        return {k: v for k, v in node_dict.items() if k not in nodes_missing_in_network}
    else:
        return node_dict


def checkInputDict(input, allowed_type, modelnames, inputname, allowed_type_name='', custom=False):
    '''
    Checks if an input is dict consisting of values of allowed_type and keys equal to modelnames,
    else if the input is a scalar, it is converted to a dict, where all the keys are modelnames,
    all of which map to input.
    :param input: a dict or an object of allowed_type
    :param allowed_type: The allowed type() for an instance of input (when a dict), or input dict (when a scalar)
    :param modelnames: keys of the new dict when input is a scalar
    :param inputname: a string speficying the name of the input, used for more informative errors
    :param allowed_type_name:
    :param custom: a boolean indicating whether input is a custom function
    :return: a dictionary, all elements of which belong to allowed_type and keys are given by modelnames
    '''

    if isinstance(input, types.FunctionType) or custom:
        # return {modelname_: input for modelname_ in modelnames}
        return input

    elif isinstance(input, dict):
        assert set(list(input.keys())) == set(modelnames), \
            'The modelnames of ' + inputname + ' are not consistent with the model'

        bools = [isinstance(v, allowed_type) for model, v in input.items()]

        assert sum(bools) == len(bools), \
            'The types of ' + inputname + ' are not all ' + allowed_type_name

        return input
    elif isinstance(input, allowed_type):
        return {modelname_: input for modelname_ in modelnames}
    else:
        raise IOError('The input ' + inputname + ' should be a dictionary or a ' + allowed_type_name \
                      + ", while it is a " + str(type(input)))