import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.models import BilinearFusion, MultiheadAttention, GatedAttention
   
    
# ----------- attention-based MIL model ----------- #

class AttnMIL(nn.Module):
    def __init__(self, 
                 in_feat_dim_he, 
                 in_feat_dim_imc,
                 hidden_feat_dim=256, 
                 out_feat_dim=256, 
                 dropout=None, 
                 n_cls=4,
                 fusion='concat'):
        """
        Args:
            in_feat_dim (int): Input feature dimension.
            hidden_feat_dim (int, optional): Hidden layer feature dimension. Defaults to 256.
            out_feat_dim (int, optional): Output feature dimension. Defaults to 256.
            dropout (float, optional): Dropout. Defaults to None.
            n_cls (int, optional): Number of output classes. Defaults to 4.
        """        
        super(AttnMIL, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fusion = fusion
        
        fc1= [nn.Linear(in_feat_dim_he, hidden_feat_dim), nn.ReLU()]
        fc2= [nn.Linear(in_feat_dim_imc, hidden_feat_dim), nn.ReLU()]
        if dropout is not None:
            fc1.append(nn.Dropout(dropout))
            fc2.append(nn.Dropout(dropout))
        attention_net = GatedAttention(L=hidden_feat_dim, D=out_feat_dim, dropout=dropout, n_cls=1)
        fc1.append(attention_net)
        fc2.append(attention_net)
        
        self.attention_net_rho = nn.Sequential(*fc1)
        self.rho = nn.Sequential(*[nn.Linear(hidden_feat_dim, out_feat_dim), nn.ReLU(), nn.Dropout(dropout)])
        
        self.attention_net_phi = nn.Sequential(*fc2)
        self.phi = nn.Sequential(*[nn.Linear(hidden_feat_dim, out_feat_dim), nn.ReLU(), nn.Dropout(dropout)])
        
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(out_feat_dim*2, out_feat_dim), nn.ReLU(), nn.Linear(out_feat_dim, out_feat_dim), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            raise ValueError("Invalid fusion method. Choose between 'concat' and 'bilinear'.")
        
        self.classify_head = nn.Sequential(*[nn.Linear(out_feat_dim, n_cls)])
        
        self.to(self.device)

    def forward(self, x, output_attention=False):
        x1, x2 = x # x1: H&E features, x2: IMC features
        
        A_1, x1 = self.attention_net_rho(x1)
        A_1 = A_1.permute(0, 2, 1)
        A_1 = F.softmax(A_1, dim=-1)
        M_1 = A_1 @ x1
        M_1 = self.rho(M_1).squeeze()
        
        A_2, x2 = self.attention_net_phi(x2)
        A_2 = A_2.permute(0, 2, 1)
        A_2 = F.softmax(A_2, dim=-1)
        M_2 = A_2 @ x2
        M_2 = self.phi(M_2).squeeze()
        
        if self.fusion == 'concat':
            M = torch.cat((M_1, M_2), axis=0)
            M = self.mm(M)
        elif self.fusion == 'bilinear':
            M = self.mm(M_1.unsqueeze(dim=0), M_2.unsqueeze(dim=0)).squeeze()

        logits = self.classify_head(M).unsqueeze(0)  
        Y_hat = torch.topk(logits, 1, dim = -1)[-1]
        Y_prob = F.softmax(logits, dim = -1)
        
        if output_attention:
            return logits, Y_hat, Y_prob, A_1, A_2
        else:
            return logits, Y_hat, Y_prob

# ----------- co attention-based MIL model ----------- #

class MCATMIL(nn.Module):
    def __init__(self, 
                 in_feat_dim_he,
                 in_feat_dim_imc, 
                 hidden_feat_dim=256, 
                 out_feat_dim=256, 
                 dropout=None, 
                 n_cls=4,
                 fusion='concat'):
        """
        Args:
            in_feat_dim (int): Input feature dimension.
            hidden_feat_dim (int, optional): Hidden layer feature dimension. Defaults to 256.
            out_feat_dim (int, optional): Output feature dimension. Defaults to 256.
            dropout (float, optional): Dropout. Defaults to None.
            n_cls (int, optional): Number of output classes. Defaults to 4.
        """        
        super(MCATMIL, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fusion = fusion
        
        fc1= [nn.Linear(in_feat_dim_he, hidden_feat_dim), nn.ReLU()]
        fc2= [nn.Linear(in_feat_dim_imc, hidden_feat_dim), nn.ReLU()]
        if dropout is not None:
            fc1.append(nn.Dropout(dropout))
            fc2.append(nn.Dropout(dropout))
        self.wsi_net_rho = nn.Sequential(*fc1)
        self.wsi_net_phi = nn.Sequential(*fc2)
        
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.transformer_rho = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_phi = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attention_head_rho = GatedAttention(L=256, D=256, dropout=dropout, n_cls=1)
        self.attention_head_phi = GatedAttention(L=256, D=256, dropout=dropout, n_cls=1)
        self.rho = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])
        self.phi = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])
        
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(out_feat_dim*2, out_feat_dim), nn.ReLU(), nn.Linear(out_feat_dim, out_feat_dim), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            raise ValueError("Invalid fusion method. Choose between 'concat' and 'bilinear'.")
        
        self.classify_head = nn.Sequential(*[nn.Linear(out_feat_dim, n_cls)])
        
        self.to(self.device)
        
    def forward(self, x, output_attention=False):
        x1, x2 = x # x1: H&E features, x2: IMC features
        
        x1 = self.wsi_net_rho(x1)
        x2 = self.wsi_net_phi(x2)
        
        x1_coattn, A_1_coattn = self.coattn(x2, x1, x1)
        x2_coattn, A_2_coattn = self.coattn(x1, x2, x2)
        
        x1_coattn = self.transformer_rho(x1_coattn)
        A_1, x1 = self.attention_head_rho(x1_coattn.squeeze(1))
        A_1 = A_1.permute(0, 2, 1)
        A_1 = F.softmax(A_1, dim=-1)
        M_1 = A_1 @ x1
        M_1 = self.rho(M_1).squeeze()
        
        x2_coattn = self.transformer_phi(x2_coattn)
        A_2, x2 = self.attention_head_phi(x2_coattn.squeeze(1))
        A_2 = A_2.permute(0, 2, 1)
        A_2 = F.softmax(A_2, dim=-1)
        M_2 = A_2 @ x2
        M_2 = self.phi(M_2).squeeze()
        
        if self.fusion == 'concat':
            M = torch.cat((M_1, M_2), axis=0)
            M = self.mm(M)
        elif self.fusion == 'bilinear':
            M = self.mm(M_1.unsqueeze(dim=0), M_2.unsqueeze(dim=0)).squeeze()
            
        logits = self.classify_head(M).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = -1)[-1]
        Y_prob = F.softmax(logits, dim = -1)
        
        if output_attention:
            return logits, Y_hat, Y_prob, A_1_coattn, A_2_coattn
        else:
            return logits, Y_hat, Y_prob 
                                                             
# ----------- attention-based MIL survival model ----------- #

class AttnSurvMIL(nn.Module):
    def __init__(self, 
                 in_feat_dim_he, 
                 in_feat_dim_imc,
                 hidden_feat_dim=256, 
                 out_feat_dim=256, 
                 dropout=None, 
                 n_cls=4,
                 fusion='concat'):
        """
        Args:
            in_feat_dim (int): Input feature dimension.
            hidden_feat_dim (int, optional): Hidden layer feature dimension. Defaults to 256.
            out_feat_dim (int, optional): Output feature dimension. Defaults to 256.
            dropout (float, optional): Dropout. Defaults to None.
            n_cls (int, optional): Number of output classes. Defaults to 4.
        """        
        super(AttnSurvMIL, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fusion = fusion
        
        fc1= [nn.Linear(in_feat_dim_he, hidden_feat_dim), nn.ReLU()]
        fc2= [nn.Linear(in_feat_dim_imc, hidden_feat_dim), nn.ReLU()]
        if dropout is not None:
            fc1.append(nn.Dropout(dropout))
            fc2.append(nn.Dropout(dropout))
        attention_net = GatedAttention(L=hidden_feat_dim, D=out_feat_dim, dropout=dropout, n_cls=1)
        fc1.append(attention_net)
        fc2.append(attention_net)
        
        self.attention_net_rho = nn.Sequential(*fc1)
        self.rho = nn.Sequential(*[nn.Linear(hidden_feat_dim, out_feat_dim), nn.ReLU(), nn.Dropout(dropout)])
        
        self.attention_net_phi = nn.Sequential(*fc2)
        self.phi = nn.Sequential(*[nn.Linear(hidden_feat_dim, out_feat_dim), nn.ReLU(), nn.Dropout(dropout)])
        
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(out_feat_dim*2, out_feat_dim), nn.ReLU(), nn.Linear(out_feat_dim, out_feat_dim), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            raise ValueError("Invalid fusion method. Choose between 'concat' and 'bilinear'.")
        
        self.classify_head = nn.Sequential(*[nn.Linear(out_feat_dim, n_cls)])
        
        self.to(self.device)

    def forward(self, x, output_attention=False):
        x1, x2 = x # x1: H&E features, x2: IMC features
        
        A_1, x1 = self.attention_net_rho(x1)
        A_1 = A_1.permute(0, 2, 1)
        A_1 = F.softmax(A_1, dim=-1)
        M_1 = A_1 @ x1
        M_1 = self.rho(M_1).squeeze()
        
        A_2, x2 = self.attention_net_phi(x2)
        A_2 = A_2.permute(0, 2, 1)
        A_2 = F.softmax(A_2, dim=-1)
        M_2 = A_2 @ x2
        M_2 = self.phi(M_2).squeeze()
        
        if self.fusion == 'concat':
            M = torch.cat((M_1, M_2), axis=0)
            M = self.mm(M)
        elif self.fusion == 'bilinear':
            M = self.mm(M_1.unsqueeze(dim=0), M_2.unsqueeze(dim=0)).squeeze()

        logits = self.classify_head(M).unsqueeze(0)                  
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        if output_attention:
            return hazards, S, Y_hat, A_1, A_2
        else:
            return hazards, S, Y_hat
        
        
# ----------- co attention-based MIL survival model ----------- #
class MCATSurvMIL(nn.Module):
    def __init__(self, 
                 in_feat_dim_he,
                 in_feat_dim_imc, 
                 hidden_feat_dim=256, 
                 out_feat_dim=256, 
                 dropout=None, 
                 n_cls=4,
                 fusion='concat'):
        """
        Args:
            in_feat_dim (int): Input feature dimension.
            hidden_feat_dim (int, optional): Hidden layer feature dimension. Defaults to 256.
            out_feat_dim (int, optional): Output feature dimension. Defaults to 256.
            dropout (float, optional): Dropout. Defaults to None.
            n_cls (int, optional): Number of output classes. Defaults to 4.
        """        
        super(MCATSurvMIL, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fusion = fusion
        
        fc1= [nn.Linear(in_feat_dim_he, hidden_feat_dim), nn.ReLU()]
        fc2= [nn.Linear(in_feat_dim_imc, hidden_feat_dim), nn.ReLU()]
        if dropout is not None:
            fc1.append(nn.Dropout(dropout))
            fc2.append(nn.Dropout(dropout))
        self.wsi_net_rho = nn.Sequential(*fc1)
        self.wsi_net_phi = nn.Sequential(*fc2)
        
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.transformer_rho = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_phi = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attention_head_rho = GatedAttention(L=256, D=256, dropout=dropout, n_cls=1)
        self.attention_head_phi = GatedAttention(L=256, D=256, dropout=dropout, n_cls=1)
        self.rho = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])
        self.phi = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])
        
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(out_feat_dim*2, out_feat_dim), nn.ReLU(), nn.Linear(out_feat_dim, out_feat_dim), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            raise ValueError("Invalid fusion method. Choose between 'concat' and 'bilinear'.")
        
        self.classify_head = nn.Sequential(*[nn.Linear(out_feat_dim, n_cls)])
        
        self.to(self.device)
        
    def forward(self, x, output_attention=False):
        x1, x2 = x # x1: H&E features, x2: IMC features
        
        x1 = self.wsi_net_rho(x1)
        x2 = self.wsi_net_phi(x2)
        
        x1_coattn, A_1_coattn = self.coattn(x2, x1, x1)
        x2_coattn, A_2_coattn = self.coattn(x1, x2, x2)
        
        x1_coattn = self.transformer_rho(x1_coattn)
        A_1, x1 = self.attention_head_rho(x1_coattn.squeeze(1))
        A_1 = A_1.permute(0, 2, 1)
        A_1 = F.softmax(A_1, dim=-1)
        M_1 = A_1 @ x1
        M_1 = self.rho(M_1).squeeze()
        
        x2_coattn = self.transformer_phi(x2_coattn)
        A_2, x2 = self.attention_head_phi(x2_coattn.squeeze(1))
        A_2 = A_2.permute(0, 2, 1)
        A_2 = F.softmax(A_2, dim=-1)
        M_2 = A_2 @ x2
        M_2 = self.phi(M_2).squeeze()
        
        if self.fusion == 'concat':
            M = torch.cat((M_1, M_2), axis=0)
            M = self.mm(M)
        elif self.fusion == 'bilinear':
            M = self.mm(M_1.unsqueeze(dim=0), M_2.unsqueeze(dim=0)).squeeze()
            
        logits = self.classify_head(M).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        if output_attention:
            return hazards, S, Y_hat, A_1_coattn, A_2_coattn
        else:
            return hazards, S, Y_hat
        
        
        