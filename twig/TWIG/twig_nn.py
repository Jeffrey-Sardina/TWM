'''
==========================
Neural Network Definitions
==========================
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torcheval.metrics.functional import r2_score
import torch.nn.functional as F

'''
===============
Reproducibility
===============
'''
torch.manual_seed(17)

'''
==========================
Neural Network Definitions
==========================
'''
class NeuralNetwork_HPs_v2(nn.Module):
    '''From umls-recommender-nn-TWM-batch_v3'''
    def __init__(self, n_struct, n_hps, n_graph):
        super().__init__()
        self.n_struct = n_struct
        self.n_hps = n_hps
        self.n_graph = n_graph

        # struct parts are from the version with no hps included
        # we now want to cinclude hps, hwoever
        self.linear_struct_1 = nn.Linear(
            in_features=n_struct,
            out_features=10
        )
        self.relu_1 = nn.ReLU()
        
        self.linear_struct_2 = nn.Linear(
            in_features=10,
            out_features=10
        )
        self.relu_2 = nn.ReLU()

        self.linear_hps_1 = nn.Linear(
            in_features=n_hps,
            out_features=6
        )
        self.relu_3 = nn.ReLU()

        self.linear_integrate_1 = nn.Linear(
            in_features=6 + 10,
            out_features=8
        )
        self.relu_4 = nn.ReLU()

        self.linear_final = nn.Linear(
            in_features=8,
            out_features=1
        )
        # self.relu_final = nn.ReLU()

    def forward(self, X):
        X_graph, X_struct_and_hps = X[:, :self.n_graph], X[:, self.n_graph:]
        X_struct, X_hps = X_struct_and_hps[:, :self.n_struct], X_struct_and_hps[:, self.n_struct:]

        X_struct = self.linear_struct_1(X_struct)
        X_struct = self.relu_1(X_struct)

        X_struct = self.linear_struct_2(X_struct)
        X_struct = self.relu_2(X_struct)

        X_hps = self.linear_hps_1(X_hps)
        X_hps = self.relu_3(X_hps)

        X = self.linear_integrate_1(
            torch.concat(
                [X_struct, X_hps],
                dim=1
            ),
        )
        X = self.relu_4(X)

        R_pred = self.linear_final(X)
        # R_pred = self.relu_final(R_pred) + 1 #min rank is 1, not 0. Only do this on the last ReLU, I think

        mrr_pred = (1 / R_pred).mean() #R_pred.round()
        return R_pred, mrr_pred

class NeuralNetwork_HPs_v3(nn.Module):
    def __init__(self, n_struct, n_hps, n_graph):
        super().__init__()
        self.n_struct = n_struct
        self.n_hps = n_hps
        self.n_graph = n_graph

        # struct parts are from the version with no hps included
        # we now want to cinclude hps, hwoever
        self.linear_struct_1 = nn.Linear(
            in_features=n_struct,
            out_features=10
        )
        self.relu_1 = nn.ReLU()
        
        self.linear_struct_2 = nn.Linear(
            in_features=10,
            out_features=10
        )
        self.relu_2 = nn.ReLU()

        self.linear_hps_1 = nn.Linear(
            in_features=n_hps,
            out_features=6
        )
        self.relu_3 = nn.ReLU()

        self.linear_integrate_1 = nn.Linear(
            in_features=6 + 10,
            out_features=8
        )
        self.relu_4 = nn.ReLU()

        self.linear_final = nn.Linear(
            in_features=8,
            out_features=1
        )
        self.sigmoid_final = nn.Sigmoid()

    def forward(self, X):
        X_graph, X_struct_and_hps = X[:, :self.n_graph], X[:, self.n_graph:]
        X_struct, X_hps = X_struct_and_hps[:, :self.n_struct], X_struct_and_hps[:, self.n_struct:]

        X_struct = self.linear_struct_1(X_struct)
        X_struct = self.relu_1(X_struct)

        X_struct = self.linear_struct_2(X_struct)
        X_struct = self.relu_2(X_struct)

        X_hps = self.linear_hps_1(X_hps)
        X_hps = self.relu_3(X_hps)

        X = self.linear_integrate_1(
            torch.concat(
                [X_struct, X_hps],
                dim=1
            ),
        )
        X = self.relu_4(X)

        R_pred = self.linear_final(X)
        R_pred = self.sigmoid_final(R_pred)

        return R_pred, None