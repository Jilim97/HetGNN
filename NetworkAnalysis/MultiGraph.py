from NetworkAnalysis.UndirectedInteractionNetwork import UndirectedInteractionNetwork#, DirectedInteractionNetwork
from NetworkAnalysis.Graph import Graph, checkInputDict
import pandas as pd
import numpy as np
import warnings
import networkx as nx
from collections import defaultdict


# TODO: write a function to assign types to each dataset, based on the order in which the genes are added
class MultiGraph(Graph):

    def __init__(self, graph_dict, colnames=None, verbose=True, keeplargestcomponent=False,
                 allow_self_connected=False, node_types=None, directed=False, gene2int=None):

        if isinstance(graph_dict, dict):
            interaction_types = []
            counter = 0
            int2edgetype = {}
            interaction_df = []

            for k, g in graph_dict.items():

                if isinstance(g, UndirectedInteractionNetwork):
                    df = g.getInteractionNamed(return_both_directions=directed)

                elif isinstance(g, pd.DataFrame):  # if directed, this df should be directed as well!
                    df = parse_df(g)

                # elif isinstance(g, DirectedInteractionNetwork):
                    # df = g.getInteractionNamed()

                else:
                    raise IOError("The input df corresponding to %s is not understood." %k)

                int2edgetype[counter] = k
                interaction_types += [counter] * df.shape[0]
                interaction_df += [df]
                counter += 1

            interaction_df = pd.concat(interaction_df, axis=0, ignore_index=True)

        elif isinstance(graph_dict, pd.DataFrame):
            interaction_df = parse_df(graph_dict)

            if "type" in interaction_df.columns:
                interaction_types = interaction_df["type"].values
                uniq_edge_types = np.unique(interaction_types)
                int2edgetype = {i: e_type for i, e_type in enumerate(uniq_edge_types)}
                edgetype2int = {v: k for k, v in int2edgetype.items()}
                interaction_types = interaction_df["type"].apply(lambda x: edgetype2int[x])
                interaction_df = interaction_df[["Gene_A", "Gene_B"]]

            else:
                int2edgetype = None
                warnings.warn("There was no edge type provided, please use an UndirectedInteractionNetwork instead.")

        else:
            raise IOError("The input data should be a dictionary or a pandas DataFrame.")

        super().__init__(interaction_df, colnames, verbose=verbose, keeplargestcomponent=keeplargestcomponent,
                         allow_self_connected=allow_self_connected, drop_duplicates=False, node_types=node_types,
                         gene2int=gene2int)

        self.interactions["type"] = interaction_types  # DANGEROUS
        self.interactions = self.interactions.drop_duplicates()

        if not allow_self_connected:
            self.interactions = self.interactions.loc[self.interactions.Gene_A != self.interactions.Gene_B]

        if keeplargestcomponent:
            self.keepLargestComponent()

        self.int2edgetype = int2edgetype
        self.directed = directed

        if not self.directed:
            self.interactions.loc[:, ['Gene_A', 'Gene_B']] = np.sort(self.interactions[['Gene_A', 'Gene_B']].values,
                                                                     axis=1)

    @property
    def isConnected(self):
        return self.get_UndirectedInteractionNetwork().isConnected

    @property
    def N_edge_types(self):
        return len(self.int2edgetype.keys())

    @property
    def is_heterogeneous(self):
        return len(self.node_types.keys()) > 1

    @property
    def edge_counts(self):
        count_df = self.interactions.groupby(["Gene_A", "Gene_B"]).nunique()
        count_df = count_df.reset_index()
        return count_df

    @property
    def edges_to_type(self):
        edges = self.edge_list(return_names=True)
        odict = {e: [] for e in edges}

        for e, e_type in zip(edges, self.edge_types_named):
            odict[e].append(e_type)

        return odict

    @property
    def edge_types_named(self):
        return self.interactions.type.apply(lambda x: self.int2edgetype[x]).values

    def edge_to_edge_type(self):
        odict = defaultdict(list)

        if self.directed:
            for t in self.interactions.itertuples(name=None, index=False):
                odict[(t[0], t[1])] += [t[2]]

        else:
            for t in self.interactions.itertuples(name=None, index=False):
                odict[(t[0], t[1])] += [t[2]]
                odict[(t[1], t[0])] += [t[2]]

        odict = {k: list(set(v)) for k, v in odict.items()}

        return odict

    def get_UndirectedInteractionNetwork(self):
        return UndirectedInteractionNetwork(self.interactions[["Gene_A", "Gene_B"]])

    # def get_DirectedInteractionNetwork(self):
        # return DirectedInteractionNetwork(self.interactions[["Gene_A", "Gene_B"]])

    def get_duplicate_edges(self):
        df = self.edge_counts
        return df.loc[df.type > 1]

    def set_node_types(self, node_types):
        self.node_types = node_types

    def interactions_as_set(self, return_names=True):
        if return_names:
            df = self.getInteractionNamed()
        else:
            df = self.interactions

        return set(df.itertuples(index=False, name=None))

    def change_edge_type_name(self, old_2_new_dict):
        for k, v in self.int2edgetype.items():
            self.int2edgetype[k] = old_2_new_dict[v] if v in old_2_new_dict.keys() else v

    def getInteractionNamed(self, return_both_directions=False):
        if return_both_directions:
            df = self.interactions[["Gene_A", "Gene_B"]].applymap(lambda x: self.int2gene[x])
            df["type"] = self.interactions.type.apply(lambda x: self.int2edgetype[x])

            df2 = df.copy(deep=True).rename(columns={'Gene_B': 'Gene_A', 'Gene_A': 'Gene_B'})
            return pd.concat([df, df2], axis=0, ignore_index=True)

        else:
            df = self.interactions[["Gene_A", "Gene_B"]].applymap(lambda x: self.int2gene[x])
            df["type"] = self.interactions.type.apply(lambda x: self.int2edgetype[x])

            return df

    def getnxGraph(self, return_names=True):
        '''return a graph instance of the networkx module'''
        if return_names:
            df = self.getInteractionNamed()
        else:
            df = self.interactions

        return nx.from_pandas_edgelist(df, source='Gene_A', target='Gene_B', create_using=nx.MultiGraph,
                                       edge_attr="type")

    def getMinimmumSpanningTree(self, as_edge_list=True, return_names=True):
        edge_list = super().getMinimmumSpanningTree(return_names=return_names)

        if as_edge_list:
            return edge_list

        else:
            edges = np.array(edge_list)
            df = pd.DataFrame(edges, columns=["Gene_A", "Gene_B"])
            mapper = self.edges_to_type
            df["type"] = [mapper[tuple(e)] for e in edges]
            df = df.explode(column="type")
            return MultiGraph(df, node_types=self.node_type_names)

    def getEdgeType_subset(self, edge_type, allow_self_connected=False):

        df = self.getInteractionNamed()
        df = df.loc[df.type == edge_type]

        if self.directed:
            # return DirectedInteractionNetwork(df,
            #                                   node_types=self.node_type_names,
            #                                   allow_self_connected=allow_self_connected,
            #                                   gene2int=self.gene2int)
            pass
        else:
            print("Returning UndirectedInteractionNetwork object.")
            return UndirectedInteractionNetwork(df,
                                                node_types=self.node_type_names,
                                                allow_self_connected=allow_self_connected,
                                                gene2int=self.gene2int)

    def getAdjMatrix(self, sort='first', as_df=False):

        count_df = self.interactions.groupby(["Gene_A", "Gene_B"]).nunique()
        count_df = count_df.reset_index()

        row_ids = list(count_df['Gene_A'])
        col_ids = list(count_df['Gene_B'])

        A = np.zeros((self.N_nodes, self.N_nodes), dtype=np.uint8)
        A[(row_ids, col_ids)] = count_df["type"].values.astype(int)

        if as_df:
            return pd.DataFrame(A, index=self.node_names, columns=self.node_names)
        else:
            return A, np.array(self.node_names)

    def sample_positives_negatives_in_train_test_validation(self, test_fraction=0.3, validation_fraction=None, excluded_sets=None,
                                                            neg_pos_ratio=5., random_state=42, debug_mode=True):
        '''
        Constructs a train, test and validation set for multiloss training.
        This function can also be used for single loss training,
        especially when the different datasets have overlapping edges.
        :param: train_ratio: The fraction of samples used for training
        :param: neg_pos_ratio: The ratio of negative examples to positive examples
        :param: assumption: Whether we work in the open world or  closed world assumption
        :param: exclude_sets: should be list of tuples
        :return: positive and negative pairs for both train and test set (4 lists in total)
        '''

        edge_types = np.unique(self.edge_types_named)

        test_fraction = checkInputDict(test_fraction, float, edge_types, 'test_fraction', allowed_type_name='float')
        # directed = checkInputDict(directed, bool, model_names, 'directed', allowed_type_name='bool')
        neg_pos_ratio = checkInputDict(neg_pos_ratio, float, edge_types, 'neg_pos_ratio', allowed_type_name='float')

        if (validation_fraction is not None) and (validation_fraction > 1e-5):
            validation_fraction = checkInputDict(validation_fraction, float, edge_types,
                                                 'validation_fraction',
                                                 allowed_type_name='float')

        else:
            validation_fraction = {modelname_: None for modelname_ in edge_types}

        if excluded_sets is None:
            excluded_sets = []

        excluded_sets = set(excluded_sets)

        np.random.seed(random_state)
        mst = set(self.getMinimmumSpanningTree(as_edge_list=True, return_names=False))
        duplicate_edges = set(self.get_duplicate_edges()[["Gene_A", "Gene_B"]].itertuples(name=None, index=None))

        test_fractions_new = {k: calculate_corrected_fraction(self.interactions.shape[0], len(mst), v)
                              for k, v in test_fraction.items()}

        validation_fractions_new = {k: calculate_corrected_fraction(self.interactions.shape[0], len(mst), v)
                                     for k, v in validation_fraction.items()}

        duplicate_edges = duplicate_edges.difference(mst)

        excluded_sets = excluded_sets.union(mst).union(duplicate_edges)
        all_positives = set(zip(self.interactions.Gene_A.values, self.interactions.Gene_B.values))

        pos_train, pos_validation, pos_test = [], [], []
        neg_train, neg_validation, neg_test = [], [], []

        smallest_edge_type, n_edges_smallest = "", 1e15

        for i, edge_type in enumerate(edge_types):
            print("Starting with edge type %i: %s" % (i, edge_type))
            subnet = self.getEdgeType_subset(edge_type=edge_type)
            subnet_positives = set(zip(subnet.interactions.Gene_A.values, subnet.interactions.Gene_B.values))
            other_positives = all_positives.difference(subnet_positives)
            # necessary to avoid positives of another being sampled as negatives for this one

            pos, negs = subnet.sample_positives_and_negatives(neg_pos_ratio=neg_pos_ratio[edge_type],
                                                              excluded_sets=excluded_sets.union(other_positives))

            assert len(pos) == len(set(subnet.edge_list(return_names=False)).difference(excluded_sets.union(other_positives)))
            assert not set(list(map(tuple, pos))) & set(list(map(tuple, negs))),\
                "getTrainTestPairs_MStree: overlap negatives train - test"

            pos_train_, pos_validation_, pos_test_ = get_random_rows(pos,
                                                                     fraction1=test_fractions_new[edge_type],
                                                                     fraction2=validation_fractions_new[edge_type],
                                                                     as_list_of_tuples=True)

            neg_train_, neg_validation_, neg_test_ = get_random_rows(negs,
                                                                     fraction1=test_fractions_new[edge_type],
                                                                     fraction2=validation_fractions_new[edge_type],
                                                                     as_list_of_tuples=True)
            pos_train += pos_train_
            pos_validation += pos_validation_
            pos_test += pos_test_

            neg_train += neg_train_
            neg_validation += neg_validation_
            neg_test += neg_test_

            excluded_sets = excluded_sets.union(set(neg_train_ + neg_validation_ + neg_test_))\
                .union(set(pos_train_ + pos_validation_ + pos_test_))

            assert not set(neg_train_) & set(neg_test_), "getTrainTestPairs_MStree: overlap negatives train - test"
            assert not set(pos_train_) & set(pos_test_), "getTrainTestPairs_MStree: overlap pos train - test"
            assert not set(pos_train) & set(pos_test), "getTrainTestPairs_MStree: overlap pos test - train"

            if n_edges_smallest > subnet.N_nodes:
                smallest_edge_type = edge_type
                n_edges_smallest = subnet.N_nodes

        pos_train += list(mst)
        assert not set(pos_train) & set(pos_test), "getTrainTestPairs_MStree: overlap pos test - train"

        if len(duplicate_edges) > 0:
            pos_train_, pos_validation_, pos_test_ = get_random_rows(np.asarray(list(duplicate_edges)),
                                                                     fraction1=test_fractions_new[smallest_edge_type],
                                                                     fraction2=validation_fractions_new[smallest_edge_type],
                                                                     as_list_of_tuples=True)

            pos_train += pos_train_
            pos_validation += pos_validation_
            pos_test += pos_test_
            assert not set(pos_train) & set(pos_test), "getTrainTestPairs_MStree: overlap pos test - train"

        if debug_mode:
            assert not set(neg_train) & set(neg_test), "getTrainTestPairs_MStree: overlap negatives train - test"
            assert not set(neg_validation) & set(neg_test), "getTrainTestPairs_MStree: overlap negatives val - test"
            assert len(set(self.edge_list(return_names=False))) == \
                   (len(pos_train) + len(pos_test) + len(pos_validation))
            pos_genes = np.unique(np.asarray(pos_train))
            assert np.all(pos_genes == np.arange(self.N_nodes))

        return list(pos_train), neg_train, list(pos_validation), neg_validation, \
               list(pos_test), neg_test

    def getTrainTestData(self, train_ratio=0.7, neg_pos_ratio=5., train_validation_ratio=None, excluded_sets=None,
                         random_state=42, mode="MLT", debug_mode=True):
        as_dicts = True
        if mode.upper() == "SLT":
            as_dicts = False

        pos_train, neg_train, pos_validation, neg_validation, \
                pos_test, neg_test = self.sample_positives_negatives_in_train_test_validation(test_fraction=1.-train_ratio,
                                                                                              validation_fraction=train_validation_ratio,
                                                                                              neg_pos_ratio=neg_pos_ratio,
                                                                                              excluded_sets=excluded_sets,
                                                                                              random_state=random_state,
                                                                                              debug_mode=debug_mode)

        pos_genes = np.unique(np.asarray(pos_train))
        assert np.all(pos_genes == np.arange(self.N_nodes))

        X_train = pos_train + neg_train
        X_val = pos_validation + neg_validation
        X_test = pos_test + neg_test

        Y_train = self.get_labels_from_pairs_and_mapper(X_train, as_dicts=as_dicts)
        Y_test = self.get_labels_from_pairs_and_mapper(X_test, as_dicts=as_dicts)
        Y_val = self.get_labels_from_pairs_and_mapper(X_val, as_dicts=as_dicts)

        if debug_mode:
            Y_ = self.get_labels_from_pairs_and_mapper(X_train, as_dicts=False)
            Y_max = Y_.max(axis=1).values.flatten()

            pos_genes = np.unique(np.asarray(X_train)[Y_max > 1e-5])
            assert np.all(pos_genes == np.arange(self.N_nodes))

            all_asserts_X_v2_undir(pos_train, pos_validation, pos_test, neg_train, neg_validation, neg_test, self)
            assert Y_max.sum() == len(pos_train)

        return X_train, X_val, X_test, Y_train, Y_val, Y_test

    def get_labels_from_pairs_and_mapper(self, pairs_list, as_dicts=True):

        labels = np.zeros((len(pairs_list), self.N_edge_types), dtype=int)
        edge_2_edge_type_map = self.edge_to_edge_type()

        non_zero_elements = [(i, edge_type) for i, pair in enumerate(pairs_list) if pair in edge_2_edge_type_map.keys()
                             for edge_type in edge_2_edge_type_map[pair]]

        non_zero_elements = tuple(zip(*non_zero_elements))

        labels[non_zero_elements] = 1
        # TODO: check if the first n rows are positive an all others are negative
        odf = pd.DataFrame(labels, columns=[self.int2edgetype[i] for i in range(self.N_edge_types)])

        if as_dicts:
            return odf.to_dict(orient="list")

        else:
            return odf


def parse_df(df):

    isnumeric = np.all(df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))
    if df.shape[1] == 3:
        df = df.copy()
        df.columns = ["Gene_A", "Gene_B", "type"]

    elif df.shape[1] == 2:
        df = df.copy()
        df.columns = ["Gene_A", "Gene_B"]

    elif isnumeric:
        df = data_df_to_edgelist(df)

    else:
        raise IOError("The input DF is not well understood."
                      "The input should either consist of 2 or 3 columns, i.e. an edgelist, "
                      "or be a completely numeric matrix that can be converted to an edgelist.")

    return df


def data_df_to_edgelist(data_df):

    r, c = np.where(data_df != 0.)

    odict = {"Gene_A": data_df.index.values[r],
             "Gene_B": data_df.columns.values[c]}

    return pd.DataFrame(odict)


def get_edge_id(arr, max_int=None):
    if max_int is None:
        max_int = np.max(arr)

    return (max_int + 1) * arr[:, 0] + arr[:, 1]


def get_random_rows(arr, fraction1, fraction2=None, as_list_of_tuples=False):

    N1 = int(arr.shape[0] * fraction1)

    if fraction2 is not None:
        N2 = int(arr.shape[0] * fraction2)

    else:
        N2 = 0

    random_ints = np.random.permutation(np.arange(arr.shape[0]))

    chunk1_ids = random_ints[:N1]
    chunk2_ids = random_ints[N1:(N1 + N2)]
    chunk3_ids = random_ints[(N1 + N2):]

    if as_list_of_tuples:
        return list(map(tuple, arr[chunk3_ids])), list(map(tuple, arr[chunk2_ids])), list(map(tuple, arr[chunk1_ids]))

    else:
        return arr[chunk3_ids], arr[chunk2_ids], arr[chunk1_ids]


def calculate_corrected_fraction(n_samples, n_MST, orig_fraction):

    if orig_fraction is None:
        return None

    n_test = int(np.ceil(orig_fraction * n_samples))

    if n_MST < n_samples:
        new_fraction = n_test / (n_samples - n_MST)

        if new_fraction > 1:
            warnings.warn("The effective size of the test set is lower then specified by the user."
                          "Typically this is caused by the minimum spanning tree containing most of the training data.")
            new_fraction = 1

        return new_fraction

    else:
        raise IOError("All edges are part of the minimum spanning tree, no test set can be made.")


def merge_labels_MLT(X_dict, Y_dict, fill_with_NaNs=False):

    if X_dict is None and Y_dict is None:
        return None, None

    else:

        assert set(list(X_dict.keys())) == set(list(Y_dict.keys())), 'Label and Pairs have different keys.'

        total_df = None

        for k in X_dict.keys():
            assert X_dict[k].shape[0] == Y_dict[k].shape[0],\
                'The number of  pairs and labels does not match for <' + k + '>'
            int_id = np.char.add(np.char.add(X_dict[k][:, 0], '$$'), X_dict[k][:, 1])
            df = pd.DataFrame({'INT_ID': int_id, 'Y_' + k: Y_dict[k]})

            if total_df is None:
                total_df = df

            else:
                total_df = pd.merge(total_df, df, how='outer', on='INT_ID')

        total_df['Gene_A'] = total_df['INT_ID'].apply(lambda x: x.split('$$')[0])
        total_df['Gene_B'] = total_df['INT_ID'].apply(lambda x: x.split('$$')[1])

        if fill_with_NaNs:
            return total_df[['Gene_A', 'Gene_B']].values, \
                   {k: total_df['Y_' + k].values for k in X_dict.keys()}

        else:
            total_df = total_df.fillna(0)
            return total_df[['Gene_A', 'Gene_B']].values, \
                   {k: total_df['Y_' + k].values.astype(int) for k in X_dict.keys()}


def all_asserts_X_v2_undir(pos_train, pos_validation, pos_test, neg_train, neg_validation, neg_test, multi_graph_object):

    # test that all nodes are present in the training set
    assert set([j for i in pos_train for j in i]) == set(np.arange(multi_graph_object.N_nodes))

    # test that all edges are present as positives
    assert len(set(multi_graph_object.edge_list(return_names=False))) == (len(pos_train) + len(pos_test) + len(pos_validation))

    # make sure there are no duplicate edges
    assert len(pos_train) == len(set([tuple(sorted(t)) for t in pos_train])), "Duplicate edges pos train"
    assert len(pos_validation) == len(set([tuple(sorted(t)) for t in pos_validation])), "Duplicate edges pos validation"
    assert len(pos_test) == len(set([tuple(sorted(t)) for t in pos_test])), "Duplicate edges pos test"

    assert len(neg_train) == len(set([tuple(sorted(t)) for t in neg_train])), "Duplicate edges neg train"
    assert len(neg_validation) == len(set([tuple(sorted(t)) for t in neg_validation])), "Duplicate edges neg validation"
    assert len(neg_test) == len(set([tuple(sorted(t)) for t in neg_test])), "Duplicate edges neg test"

    # make sure there are no overlaps
    assert not set(neg_train) & set(neg_test), "Overlap negatives train - test"
    assert not set(neg_train) & set(neg_validation), "Overlap negatives train - validation"
    assert not set(neg_train) & set(pos_train), "Overlap negatives train - pos train"
    assert not set(neg_train) & set(pos_test), "Overlap negatives train - pos test"
    assert not set(neg_train) & set(pos_validation), "Overlap negatives train - pos validation"

    neg_train2 = set([(t[1], t[0]) for t in neg_train])

    assert not set(neg_train2) & set(neg_test), "Overlap negatives train - test"
    assert not set(neg_train2) & set(neg_validation), "Overlap negatives train - validation"
    assert not set(neg_train2) & set(pos_train), "Overlap negatives train - pos train"
    assert not set(neg_train2) & set(pos_test), "Overlap negatives train - pos test"
    assert not set(neg_train2) & set(pos_validation), "Overlap negatives train - pos validation"

    assert not set(pos_train) & set(neg_test), "Overlap positives train - neg test"
    assert not set(pos_train) & set(neg_validation), "Overlap positives train - neg validation"
    assert not set(pos_train) & set(pos_test), "Overlap positives train - pos test"
    assert not set(pos_train) & set(pos_validation), "Overlap positives train - pos validation"

    pos_train2 = set([(t[1], t[0]) for t in pos_train])

    assert not set(pos_train2) & set(neg_test), "Overlap positives train - neg test"
    assert not set(pos_train2) & set(neg_validation), "Overlap positives train - neg validation"
    assert not set(pos_train2) & set(pos_test), "Overlap positives train - pos test"
    assert not set(pos_train2) & set(pos_validation), "Overlap positives train - pos validation"

    assert not set(neg_test) & set(neg_validation), "Overlap negatives test - validation"
    assert not set(neg_test) & set(pos_test), "Overlap negatives test - pos test"
    assert not set(neg_test) & set(pos_validation), "Overlap negatives test - pos validation"

    neg_test2 = set([(t[1], t[0]) for t in neg_test])

    assert not set(neg_test2) & set(neg_validation), "Overlap negatives test - validation"
    assert not set(neg_test2) & set(pos_test), "Overlap negatives test - pos test"
    assert not set(neg_test2) & set(pos_validation), "Overlap negatives test - pos validation"

    assert not set(neg_validation) & set(pos_test), "Overlap negatives validation - pos test"
    assert not set(neg_validation) & set(pos_validation), "Overlap negatives validation - pos validation"

    neg_validation2 = set([(t[1], t[0]) for t in neg_validation])

    assert not set(neg_validation2) & set(pos_test), "Overlap negatives validation - pos test"
    assert not set(neg_validation2) & set(pos_validation), "Overlap negatives validation - pos validation"

    assert not set(pos_test) & set(pos_validation), "Overlap negatives pos test - pos validation"
    pos_test2 = set([(t[1], t[0]) for t in pos_test])

    assert not set(pos_test2) & set(pos_validation), "Overlap negatives pos test - pos validation"
    # TODO: add asserts, test this function

    print("Size of pos test: %i" % len(pos_test))
    print("Size of pos train: %i" % len(pos_train))
    print("Size of pos validation: %i" % len(pos_validation))
    print("Size of neg test: %i" % len(neg_test))
    print("Size of neg train: %i" % len(neg_train))
    print("Size of neg validation: %i" % len(neg_validation))


def all_asserts_X_v2_dir(pos_train, pos_validation, pos_test, neg_train, neg_validation, neg_test, multi_graph_object):

    # test that all nodes are present in the training set
    assert set([j for i in pos_train for j in i]) == set(np.arange(multi_graph_object.N_nodes))

    # test that all edges are present as positives
    assert len(set(multi_graph_object.edge_list(return_names=False))) == (len(pos_train) + len(pos_test) + len(pos_validation))

    # make sure there are no duplicate edges
    assert len(pos_train) == len(set(pos_train)), "Duplicate edges pos train"
    assert len(pos_validation) == len(set(pos_validation)), "Duplicate edges pos validation"
    assert len(pos_test) == len(set(pos_test)), "Duplicate edges pos test"

    assert len(neg_train) == len(set(neg_train)), "Duplicate edges neg train"
    assert len(neg_validation) == len(set(neg_validation)), "Duplicate edges neg validation"
    assert len(neg_test) == len(set(neg_test)), "Duplicate edges neg test"

    # make sure there are no overlaps
    assert not set(neg_train) & set(neg_test), "Overlap negatives train - test"
    assert not set(neg_train) & set(neg_validation), "Overlap negatives train - validation"
    assert not set(neg_train) & set(pos_train), "Overlap negatives train - pos train"
    assert not set(neg_train) & set(pos_test), "Overlap negatives train - pos test"
    assert not set(neg_train) & set(pos_validation), "Overlap negatives train - pos validation"

    assert not set(pos_train) & set(neg_test), "Overlap positives train - neg test"
    assert not set(pos_train) & set(neg_validation), "Overlap positives train - neg validation"
    assert not set(pos_train) & set(pos_test), "Overlap positives train - pos test"
    assert not set(pos_train) & set(pos_validation), "Overlap positives train - pos validation"

    assert not set(neg_test) & set(neg_validation), "Overlap negatives test - validation"
    assert not set(neg_test) & set(pos_test), "Overlap negatives test - pos test"
    assert not set(neg_test) & set(pos_validation), "Overlap negatives test - pos validation"

    assert not set(neg_validation) & set(pos_test), "Overlap negatives validation - pos test"
    assert not set(neg_validation) & set(pos_validation), "Overlap negatives validation - pos validation"

    assert not set(pos_test) & set(pos_validation), "Overlap negatives pos test - pos validation"

    # TODO: add asserts, test this function

    print("Size of pos test: %i" % len(pos_test))
    print("Size of pos train: %i" % len(pos_train))
    print("Size of pos validation: %i" % len(pos_validation))
    print("Size of neg test: %i" % len(neg_test))
    print("Size of neg train: %i" % len(neg_train))
    print("Size of neg validation: %i" % len(neg_validation))


def all_asserts_X_undir(X_train, X_val, X_test, Y_train, Y_val, Y_test, multi_graph_object):
    X_train, X_val, X_test = np.asarray(X_train), np.asarray(X_val), np.asarray(X_test)

    assert np.all(X_train == np.sort(X_train, axis=1))
    if len(X_val) > 0:
        assert np.all(X_val == np.sort(X_val, axis=1))
    assert np.all(X_test == np.sort(X_test, axis=1))

    # test that all nodes are present in the training set
    assert np.all(np.unique(X_train) == np.arange(multi_graph_object.N_nodes))

    single_labels = np.zeros(X_train.shape[0])

    for arr in Y_train.values():
        single_labels = np.maximum(single_labels, np.asarray(arr))

    # multi_graph_object.isConnected
    mst_genes = np.unique(np.asarray(multi_graph_object.getMinimmumSpanningTree(return_names=False)))
    assert np.all(np.unique(X_train[single_labels > 1e-5]) == np.arange(multi_graph_object.N_nodes))
    assert np.all(mst_genes == np.arange(multi_graph_object.N_nodes))

    for k in Y_train.keys():

        pos_train, neg_train = list(map(tuple, X_train[Y_train[k] == 1])), list(map(tuple, X_train[Y_train[k] == 0]))
        pos_validation, neg_validation = list(map(tuple, X_val[Y_val[k] == 1])), list(map(tuple, X_val[Y_val[k] == 0]))
        pos_test, neg_test = list(map(tuple, X_test[Y_test[k] == 1])), list(map(tuple, X_test[Y_test[k] == 0]))

        # make sure there are no overlaps
        assert not set(neg_train) & set(neg_test), "Overlap negatives train - test"
        assert not set(neg_train) & set(neg_validation), "Overlap negatives train - validation"
        assert not set(neg_train) & set(pos_train), "Overlap negatives train - pos train"
        assert not set(neg_train) & set(pos_test), "Overlap negatives train - pos test"
        assert not set(neg_train) & set(pos_validation), "Overlap negatives train - pos validation"

        neg_train2 = set([(t[1], t[0]) for t in neg_train])

        assert not set(neg_train2) & set(neg_test), "Overlap negatives train - test"
        assert not set(neg_train2) & set(neg_validation), "Overlap negatives train - validation"
        assert not set(neg_train2) & set(pos_train), "Overlap negatives train - pos train"
        assert not set(neg_train2) & set(pos_test), "Overlap negatives train - pos test"
        assert not set(neg_train2) & set(pos_validation), "Overlap negatives train - pos validation"

        assert not set(pos_train) & set(neg_test), "Overlap positives train - neg test"
        assert not set(pos_train) & set(neg_validation), "Overlap positives train - neg validation"
        assert not set(pos_train) & set(pos_test), "Overlap positives train - pos test"
        assert not set(pos_train) & set(pos_validation), "Overlap positives train - pos validation"

        pos_train2 = set([(t[1], t[0]) for t in pos_train])

        assert not set(pos_train2) & set(neg_test), "Overlap positives train - neg test"
        assert not set(pos_train2) & set(neg_validation), "Overlap positives train - neg validation"
        assert not set(pos_train2) & set(pos_test), "Overlap positives train - pos test"
        assert not set(pos_train2) & set(pos_validation), "Overlap positives train - pos validation"

        assert not set(neg_test) & set(neg_validation), "Overlap negatives test - validation"
        assert not set(neg_test) & set(pos_test), "Overlap negatives test - pos test"
        assert not set(neg_test) & set(pos_validation), "Overlap negatives test - pos validation"

        neg_test2 = set([(t[1], t[0]) for t in neg_test])

        assert not set(neg_test2) & set(neg_validation), "Overlap negatives test - validation"
        assert not set(neg_test2) & set(pos_test), "Overlap negatives test - pos test"
        assert not set(neg_test2) & set(pos_validation), "Overlap negatives test - pos validation"

        assert not set(neg_validation) & set(pos_test), "Overlap negatives validation - pos test"
        assert not set(neg_validation) & set(pos_validation), "Overlap negatives validation - pos validation"

        neg_validation2 = set([(t[1], t[0]) for t in neg_validation])

        assert not set(neg_validation2) & set(pos_test), "Overlap negatives validation - pos test"
        assert not set(neg_validation2) & set(pos_validation), "Overlap negatives validation - pos validation"

        assert not set(pos_test) & set(pos_validation), "Overlap negatives pos test - pos validation"
        pos_test2 = set([(t[1], t[0]) for t in pos_test])

        assert not set(pos_test2) & set(pos_validation), "Overlap negatives pos test - pos validation"


def all_asserts_X_dir(X_train, X_val, X_test, Y_train, Y_val, Y_test, multi_graph_object):
    X_train, X_val, X_test = np.asarray(X_train), np.asarray(X_val), np.asarray(X_test)

    # test that all nodes are present in the training set
    assert np.all(np.unique(X_train) == np.arange(multi_graph_object.N_nodes))

    single_labels = np.zeros(X_train.shape[0])

    for arr in Y_train.values():
        single_labels = np.maximum(single_labels, np.asarray(arr))

    mst_genes = np.unique(np.asarray(multi_graph_object.getMinimmumSpanningTree(return_names=False)))
    assert np.all(np.unique(X_train[single_labels > 1e-5]) == np.arange(multi_graph_object.N_nodes))
    assert np.all(mst_genes == np.arange(multi_graph_object.N_nodes))

    for k in Y_train.keys():
        pos_train, neg_train = list(map(tuple, X_train[Y_train[k] == 1])), list(
            map(tuple, X_train[Y_train[k] == 0]))
        pos_validation, neg_validation = list(map(tuple, X_val[Y_val[k] == 1])), list(
            map(tuple, X_val[Y_val[k] == 0]))
        pos_test, neg_test = list(map(tuple, X_test[Y_test[k] == 1])), list(map(tuple, X_test[Y_test[k] == 0]))

        # make sure there are no overlaps
        assert not set(neg_train) & set(neg_test), "Overlap negatives train - test"
        assert not set(neg_train) & set(neg_validation), "Overlap negatives train - validation"
        assert not set(neg_train) & set(pos_train), "Overlap negatives train - pos train"
        assert not set(neg_train) & set(pos_test), "Overlap negatives train - pos test"
        assert not set(neg_train) & set(pos_validation), "Overlap negatives train - pos validation"

        assert not set(pos_train) & set(neg_test), "Overlap positives train - neg test"
        assert not set(pos_train) & set(neg_validation), "Overlap positives train - neg validation"
        assert not set(pos_train) & set(pos_test), "Overlap positives train - pos test"
        assert not set(pos_train) & set(pos_validation), "Overlap positives train - pos validation"

        assert not set(neg_test) & set(neg_validation), "Overlap negatives test - validation"
        assert not set(neg_test) & set(pos_test), "Overlap negatives test - pos test"
        assert not set(neg_test) & set(pos_validation), "Overlap negatives test - pos validation"

        assert not set(neg_validation) & set(pos_test), "Overlap negatives validation - pos test"
        assert not set(neg_validation) & set(pos_validation), "Overlap negatives validation - pos validation"

        assert not set(pos_test) & set(pos_validation), "Overlap negatives pos test - pos validation"

# TODO: add asserts, test this function
