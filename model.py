import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# end to end network
class SiameseNetwork_ete(nn.Module):
    def __init__(self):
        super(SiameseNetwork_ete, self).__init__()
        self.dropout_rate = 0.5
        self.dropout = nn.Dropout(p = 0.5)

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=13, stride=1, padding=6),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(p = self.dropout_rate),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(64, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(p = self.dropout_rate),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(32, 16, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            #nn.Dropout(p = self.dropout_rate),
            nn.MaxPool1d(kernel_size=4),
        )

        self.rr_fc = nn.Sequential(
            nn.Linear(10, 5), #3
        )

        self.dropout = nn.Dropout(p = self.dropout_rate)

        self.concat_fc = nn.Sequential(
            nn.Linear(138,64), # rr -> 326
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p = self.dropout_rate),
            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p = self.dropout_rate),
            #nn.Linear(64,5),
        )

        self.concat_final = nn.Sequential(
            nn.Linear(32,5),
        )

    def forward_once(self, x, rr):
        output = self.conv(x)
        output_ = nn.Flatten(start_dim=1)(output)
        rr_out = self.rr_fc(rr)
        output1 = torch.cat((output_, rr_out), dim=1)
        return output1
    
    def forward(self, x, x_avg, rr, rr_avg):

        output1_ = self.forward_once(x, rr)
        output2_ = self.forward_once(x_avg, rr_avg)
    
        output1 = self.dropout(output1_)
        output2 = self.dropout(output2_)

        output = torch.cat((output1, output2), dim=1)
        #print('output : ',output.shape)
        output = self.concat_fc(output) # for center loss
        output_final = self.concat_final(output)
        return output1_, output2_, output, output_final
