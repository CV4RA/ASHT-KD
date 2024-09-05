import torch
import torch.nn as nn
from models.teacher_model import TeacherModel

class MultiTeacherModel(nn.Module):
    def __init__(self):
        super(MultiTeacherModel, self).__init__()
        self.teacher1 = TeacherModel()
        self.teacher2 = TeacherModel()
        self.teacher3 = TeacherModel()

    def forward(self, x):
        
        out1 = self.teacher1(x)
        out2 = self.teacher2(x)
        out3 = self.teacher3(x)
        
        
        out = (out1 + out2 + out3) / 3.0
        return out
