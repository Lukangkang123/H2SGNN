import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_sparse import matmul as torch_sparse_matmul
from utils import add


class H2SGNN(nn.Module):
    def __init__(self, in_dim, num_classes,lis, lis_t, args, bns=False):
        super(H2SGNN, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.emb_dim = args.emb_dim
        self.h_dim = args.hidden
        self.K = args.K
        self.lis =lis
        self.lis_t =lis_t
        self.alpha =  args.alpha
        self.alpha_list = args.alpha_list
        self.poly = args.poly
        self.a = args.a
        self.b = args.b
        self.u = args.u
        self.dataset = args.dataset
        self.feat_project = nn.Linear(self.in_dim, self.emb_dim, bias=bns)
        self.lin1 = nn.Linear(self.emb_dim, self.h_dim)
        self.lin2 = nn.Linear(self.h_dim, self.num_classes)

        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

        self.beta = torch.nn.Parameter(torch.FloatTensor(len(lis), 1).fill_(1.0 / len(lis)), requires_grad=True)
        torch.nn.init.uniform_(self.beta, a=0, b=0.1)

        TEMP = self.alpha*(1-self.alpha)**np.arange(self.K+1)
        TEMP[-1] = (1-self.alpha)**self.K
        self.temp = torch.nn.Parameter(torch.tensor(TEMP)) # gamma in paper
        
        TEMP_LIST = self.alpha_list*(1-self.alpha_list)**np.arange(self.K+1)
        TEMP_LIST[-1] = (1-self.alpha_list)**self.K
    
        self.temp_list = nn.ParameterList([nn.Parameter(torch.tensor(TEMP_LIST, dtype=torch.float32)) for _ in range(len(lis))])
        self.reset_parameters()
        self.temp = torch.nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def temp_init(self,alpha):
        TEMP = alpha*(1-alpha)**np.arange(self.K+1)
        TEMP[-1] = (1-alpha)**self.K
        temp = torch.nn.Parameter(torch.tensor(TEMP, dtype=torch.float32))
        return temp


    def normalize(self, x):
        means = x.mean(1, keepdim=True)
        deviations = x.std(1, keepdim=True)
        x = (x - means) / deviations
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        return x

    def forward(self, adjs, features_list,adjs_list):

        beta = torch.softmax(self.beta,0)
        A = add(adjs_list,beta)
        x = self.input_drop(self.feat_project(features_list[0]))

        x = F.relu(self.lin1(x))
        if self.dataset in ['IMDB','AMiner']:
            x = self.normalize(x)
        x = self.dropout(x)
        x = self.lin2(x)

      
        if self.poly =='GPR_GPR':
            res = self.GPR_GPR(adjs_list,A,x)
        if self.poly == 'GPR_legendre':
            res = self.GPR_legendre(adjs_list,A,x)
        if self.poly =='GPR_jaco':
            res = self.GPR_jaco(adjs_list,A,x,self.a,self.b)
    
        return res
    

    def GPR_legendre(self,adjs_list,A, x):
        # Global Hybrid Filtering
        res  =  self.temp[0]
        for k in range(self.K):
            x = torch_sparse_matmul(A,x)
            res  = res  +  self.temp[k+1]*x

        # Local Individual Filtering
        for adj, temp_i in zip(adjs_list, self.temp_list):
            P_prev2 = x  # T_0(x)
            P_prev1 = torch_sparse_matmul(adj, x)  # T_1(x)
            res_i = temp_i[0] * P_prev2 + temp_i[1] * P_prev1
            for k in range(2,self.K+1):
                P_current = ((2 * k - 1) * torch_sparse_matmul(adj, P_prev1) - (k - 1) * P_prev2)/k
                P_prev2 = P_prev1
                P_prev1 = P_current
                res_i  = res_i  + temp_i[k] * P_current
            res = res + res_i
        return res
    


    def GPR_GPR(self,adjs_list,A, x):
        res  =  self.temp[0]*x
        # Global Hybrid Filtering
        for k in range(self.K):
            x = torch_sparse_matmul(A,x)
            res  = res  +self.temp[k+1]*x
        
        # Local Individual Filtering
        for i,(adj, temp_i) in enumerate(zip(adjs_list, self.temp_list)):
            res_i  = temp_i[0]*x
            for k in range(self.K):
                x = torch_sparse_matmul(adj,x)
                res_i  = res_i  +  temp_i[k+1]*x
            res = res + res_i
        return res
    
    
    def GPR_jaco(self, adjs_list, A, x, a=0.0, b=0.0):
        res = self.temp[0]
        # Global Hybrid Filtering
        for k in range(self.K):
            x = torch_sparse_matmul(A, x)
            res = res + self.temp[k + 1] * x

        # Local Individual Filtering
        for adj, temp_i in zip(adjs_list, self.temp_list):
            P_prev2 = x  # P_0(x)
            P_prev1 = (a + b+ 2) / 2 * torch_sparse_matmul(adj, x) + (a - b) / 2 * x  # P_1(x)
            res_i = temp_i[0] * P_prev2 + temp_i[1] * P_prev1
            
            for L in range(2, self.K+1):
                coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
                coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
                coef_lm1_2 = (2 * L + a + b - 1) * (a**2 - b**2)
                coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
                tmp1 =  (coef_lm1_1 / coef_l)
                tmp2 =  (coef_lm1_2 / coef_l)
                tmp3 =  (coef_lm2 / coef_l)
        
                P_current = tmp1 * (torch_sparse_matmul(adj, P_prev1)) - tmp2 * P_prev1 - tmp3 * P_prev2 
                P_prev2 = P_prev1
                P_prev1 = P_current
                res_i  = res_i  + temp_i[L] * P_current

            res = res + res_i

        return res
    