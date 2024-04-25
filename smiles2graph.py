import os
import random

#import dgl
from torch_geometric.utils import from_dgl
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures, AllChem
import torch_geometric
import networkx as nx
import cugraph
import matplotlib.pyplot as plt
import pprint

import warnings
warnings.simplefilter('ignore', FutureWarning)

MAX_NUM_PP_GRAPHS = 8


def sample_probability(elment_array, plist, N):
    Psample = []
    n = len(plist)
    index = int(random.random() * n)
    mw = max(plist)
    beta = 0.0
    for i in range(N):
        beta = beta + random.random() * 2.0 * mw
        while beta > plist[index]:
            beta = beta - plist[index]
            index = (index + 1) % n
        Psample.append(elment_array[index])

    return Psample


def six_encoding(atom):
    # actually seven
    orgin_phco = [0, 0, 0, 0, 0, 0, 0, 0]
    for j in atom:
        orgin_phco[j] = 1
    return orgin_phco[1:]


def cal_dist(mol, start_atom, end_tom):
    list_ = []
    list_.append(start_atom)
    seen = set()
    seen.add(start_atom)
    parent = {start_atom: None}
    nei_atom = []
    bond_num = mol.GetNumBonds()
    while (len(list_) > 0):
        vertex = (list_[0])
        del (list_[0])
        nei_atom = ([n.GetIdx() for n in mol.GetAtomWithIdx(vertex).GetNeighbors()])
        for w in nei_atom:
            if w not in seen:
                list_.append(w)
                seen.add(w)
                parent[w] = vertex
    path_atom = []
    while end_tom != None:
        path_atom.append(end_tom)
        end_tom = parent[end_tom]
    nei_bond = []
    for i in range(bond_num):
        nei_bond.append((mol.GetBondWithIdx(i).GetBondType().name, mol.GetBondWithIdx(i).GetBeginAtomIdx(),
                         mol.GetBondWithIdx(i).GetEndAtomIdx()))
    bond_collection = []
    for idx in range(len(path_atom) - 1):
        bond_start = path_atom[idx]
        bond_end = path_atom[idx + 1]
        for bond_type in nei_bond:
            if len(list(set([bond_type[1], bond_type[2]]).intersection(set([bond_start, bond_end])))) == 2:
                bond_ = bond_type[0]
                if [bond_, bond_type[1], bond_type[2]] not in bond_collection:
                    bond_collection.append([bond_, bond_type[1], bond_type[2]])
    dist = 0
    for elment in bond_collection:
        if elment[0] == 'SINGLE':
            dist = dist + 1
        elif elment[0] == 'DOUBLE':
            dist = dist + 0.87
        elif elment[0] == 'AROMATIC':
            dist = dist + 0.91
        else:
            dist = dist + 0.78
    return dist


def smiles_code_(smiles, g, e_list):
    smiles = smiles
    dgl = g
    e_elment = e_list
    mol = Chem.MolFromSmiles(smiles)
    atom_num = mol.GetNumAtoms()
    atom_index_list = []
    smiles_code = np.zeros((atom_num, MAX_NUM_PP_GRAPHS))
    for elment_i in range(len(e_elment)):  ##定位这个元素在第几个药效团
        elment = e_elment[elment_i]
        for e_i in range(len(elment)):
            e_index = elment[e_i]
            for atom in mol.GetAtoms():  ##定位这个原子在分子中的索引
                if e_index == atom.GetIdx():
                    list_ = ((dgl.ndata['type'])[elment_i]).tolist()
                    for list_i in range(len(list_)):
                        if list_[list_i] == 1:
                            smiles_code[atom.GetIdx(), elment_i] = 1.0
    return smiles_code


def smiles2ppgraph(smiles:str, y, cpu=True):
    '''
    :param smiles: a molecule
    :return: (pp_graph, mapping)
        pp_graph: DGLGraph, the corresponding **random** pharmacophore graph
        mapping: np.Array ((atom_num, MAX_NUM_PP_GRAPHS)) the mapping between atoms and pharmacophore features
    '''
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=10000)
    mol = Chem.MolFromSmiles(smiles)
    query = Chem.MolFromSmiles("[Dy]NC(=O)")
    mol = AllChem.DeleteSubstructs(mol, query)
    #query = Chem.MolFromSmiles("")
   #mol = AllChem.DeleteSubstructs(mol, query)
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    atom_index_list = []
    pharmocophore_all = []

    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    n_feats = factory.GetNumMolFeatures(mol)
    for i in range(n_feats):
        f = factory.GetMolFeature(mol, i)
        phar = f.GetFamily()
        atom_index = f.GetAtomIds()
        atom_index = tuple(sorted(atom_index))
        atom_type = f.GetType()
        mapping = {'Aromatic': 1, 'Hydrophobe': 2, 'PosIonizable': 3,
                   'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6}
        phar_index = mapping.setdefault(phar, 7)
        pharmocophore_ = [phar_index, atom_index]  # some pharmacophore feature
        pharmocophore_all.append(pharmocophore_)  # all pharmacophore features within a molecule
        atom_index_list.append(atom_index)  # atom indices of one pharmacophore feature
    random.shuffle(pharmocophore_all)
    num = [3, 4, 5, 6, 7]
    num_p = [0.086, 0.0864, 0.389, 0.495, 0.0273]  # P(Number of Pharmacophore points)
    num_ = sample_probability(num, num_p, 1)

    ## The randomly generated clusters are obtained,
    # and the next step is to perform a preliminary merging of these randomly generated clusters with identical elements
    #if len(pharmocophore_all) >= int(num_[0]):
    if False:
        mol_phco = pharmocophore_all[:int(num_[0])]
    else:
        mol_phco = pharmocophore_all
    #print(mol_phco)
    for pharmocophore_all_i in range(len(mol_phco)):
        for pharmocophore_all_j in range(len(mol_phco)):
            if mol_phco[pharmocophore_all_i][1] == mol_phco[pharmocophore_all_j][1] \
                    and mol_phco[pharmocophore_all_i][0] != mol_phco[pharmocophore_all_j][0]:  # ファーマコフォアのatomインデックスが一緒なのにファーマコフォアが異なる場合→mol_phoco[0]には2つのphar_indexが格納されたlistにする
                index_ = [min(mol_phco[pharmocophore_all_i][0], mol_phco[pharmocophore_all_j][0]),
                          max(mol_phco[pharmocophore_all_i][0], mol_phco[pharmocophore_all_j][0])]
                mol_phco[pharmocophore_all_j] = [index_, mol_phco[pharmocophore_all_i][1]]
                mol_phco[pharmocophore_all_i] = [index_, mol_phco[pharmocophore_all_i][1]]
            else:
                index_ = mol_phco[pharmocophore_all_i][0]
    #print(mol_phco)
    unique_index_filter = []
    unique_index = []
    for mol_phco_candidate_single in mol_phco:   ## mol_phcoの要素が[[phar_index], atom_id]になるように整形? + 重複削除
        if mol_phco_candidate_single not in unique_index:
            if type(mol_phco[0]) == list:
                unique_index.append(mol_phco_candidate_single)
            else:
                unique_index.append([[mol_phco_candidate_single[0]], mol_phco_candidate_single[1]])
    #print(unique_index)

    for unique_index_single in unique_index:
        if unique_index_single not in unique_index_filter:
            unique_index_filter.append(unique_index_single)  ## The following is the order of the pharmacophores by atomic number
    #print(unique_index_filter)
    sort_index_list = []
    for unique_index_filter_i in unique_index_filter:  ## Collect the mean of the participating elements
        sort_index = sum(unique_index_filter_i[1]) / len(unique_index_filter_i[1])
        sort_index_list.append(sort_index)
    sorted_id = sorted(range(len(sort_index_list)), key=lambda k: sort_index_list[k])
    #print("sort_index", sort_index_list)
    #print("sorted_id", sorted_id)
    unique_index_filter_sort = []
    for index_id in sorted_id:                                                # atom_idの平均の小さい順になるようにunique_indexをソート
        unique_index_filter_sort.append(unique_index_filter[index_id])
    #print("unique_index_filter_sort", unique_index_filter_sort)

    adj_matrix = np.array(Chem.GetAdjacencyMatrix(mol))
    dist_matrix = np.zeros_like(adj_matrix, dtype=float)
    if cpu:
        g = nx.Graph()
        g.add_nodes_from(list(range(len(adj_matrix))))
    for i in range(len(adj_matrix[0])):
        for j in range(i+1, len(adj_matrix[1])):
            if adj_matrix[i,j] == 1:
                bond = mol.GetBondBetweenAtoms(i, j)
                bond_type = str(bond.GetBondType())
                if bond_type == 'SINGLE':
                    dist = 1
                elif bond_type == 'DOUBLE':
                    dist = 0.87
                elif bond_type == 'AROMATIC':
                    dist = 0.91
                else:
                    dist = 0.78
                if cpu:
                    g.add_edge(i,j,weight=dist)
                else:

                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist

    if not cpu:
        g = cugraph.Graph(directed=False)
        g.from_numpy_array(dist_matrix)
    #short_path = dict(nx.shortest_path_length(g))

    position_matrix = np.zeros((len(unique_index_filter_sort), len(unique_index_filter_sort)))
    node_feat = []
    for mol_phco_i in range(len(unique_index_filter_sort)):
        mol_phco_i_elment = list(unique_index_filter_sort[mol_phco_i][1]) # mol_phco_i の　atom_ids
        if type(unique_index_filter_sort[mol_phco_i][0]) == list:  # mol_phco_i[0]をリストに
            enc = six_encoding(unique_index_filter_sort[mol_phco_i][0])
        else:
            enc = six_encoding([unique_index_filter_sort[mol_phco_i][0]])

        siz = len(mol_phco_i_elment)  # size_ -> ファーマコフォアの原子数
        enc.append(siz)
        node_feat.append(enc)

        median_atom_i_element = int(np.median(mol_phco_i_elment))
        if cpu:
            short_path = nx.shortest_path_length(g, source=median_atom_i_element)
        else:
            short_path = cugraph.shortest_path_length(g, source=median_atom_i_element)
            short_path = short_path.set_index("vertex")

        for mol_phco_j in range(mol_phco_i+1, len(unique_index_filter_sort)):
            mol_phco_j_elment = list(unique_index_filter_sort[mol_phco_j][1])
            if mol_phco_i_elment == mol_phco_j_elment:
                position_matrix[mol_phco_i, mol_phco_j] = 0.01
                position_matrix[mol_phco_j, mol_phco_j] = 0.01
            elif str(set(mol_phco_i_elment).intersection(set(mol_phco_j_elment))) == 'set()':
                dist_set = []
                #for atom_i in mol_phco_i_elment:
                #    for atom_j in mol_phco_j_elment:
                #print(mol_phco_i_elment)
                median_atom_j_element = int(np.median(mol_phco_j_elment))
                if cpu:
                    dist = short_path[median_atom_j_element]
                else:
                    dist = short_path.at[int(median_atom_j_element), "distance"]
                        #dist_set.append(dist)
                #min_dist = min(dist_set)
                min_dist=dist
                if max(len(mol_phco_i_elment), len(mol_phco_j_elment)) == 1:
                    position_matrix[mol_phco_i, mol_phco_j] = min_dist
                    position_matrix[mol_phco_j, mol_phco_i] = min_dist
                else:
                    position_matrix[mol_phco_i, mol_phco_j] = min_dist + max(len(mol_phco_i_elment),
                                                                             len(mol_phco_j_elment)) * 0.2
                    position_matrix[mol_phco_j, mol_phco_i] = min_dist + max(len(mol_phco_i_elment),
                                                                             len(mol_phco_j_elment)) * 0.2
            else:
                for type_elment_i in mol_phco_i_elment:
                    for type_elment_j in mol_phco_j_elment:
                        if type_elment_i == type_elment_j:
                            position_matrix[mol_phco_i, mol_phco_j] = max(len(mol_phco_i_elment),
                                                                          len(mol_phco_j_elment)) * 0.2
                            position_matrix[mol_phco_j, mol_phco_i] = max(len(mol_phco_i_elment),
                                                                          len(mol_phco_j_elment)) * 0.2
                        ##The above is a summary of the cases where the two pharmacophores have direct elemental intersection.
    weights = []
    u_list = []
    v_list = []
    phco_single = []

    #print(position_matrix)
    
    for u in range(position_matrix.shape[0]):
        for v in range(position_matrix.shape[1]):
            if u != v:
                u_list.append(u)
                v_list.append(v)
                if position_matrix[u, v] >= position_matrix[v, u]:
                    weights.append(position_matrix[v, u])
                else:
                    weights.append(position_matrix[u, v])
    u_list_tensor = torch.tensor(u_list)
    v_list_tensor = torch.tensor(v_list)
    edges = torch.tensor(np.array([u_list_tensor, v_list_tensor]))#, dtype=torch.long)
    edge_attr = torch.tensor(weights).unsqueeze(1)
    embed = torch.tensor(node_feat, dtype=torch.float32)

    #print(embed.shape)
    #print(edges.shape)
    #print(edge_attr.shape)

    y = torch.tensor([y], dtype=torch.float32)
    g = torch_geometric.data.Data(x=embed, edge_index=edges, edge_attr=edge_attr, smiles=smiles, y=y)

    #g.ndata['type'] = type_list_tensor
    #g.ndata['size'] = torch.HalfTensor(size_)

    return g

if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm 
    smiles = 'CC1=C(C(C)=O)C(N(C(CC)CC)C2=NC(NC3=NC=C(N4CCNCC4)C=C3)=NC=C12)=O'
    train = pd.read_csv("data/fold/sEH/train_fold0.csv", usecols=["molecule_smiles"])
    for i in tqdm(range(len(train))):
        row = train.iloc[i,:]
        data = smiles2ppgraph(row["molecule_smiles"], 1, cpu=True)  ##g_药效团；smiles_code_res：smiles编码（矩阵格式）
    
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    nx.draw(g)
    plt.show()