class MultiTeacherKnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=3):
        super(MultiTeacherKnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_output, teacher_outputs, labels):
        avg_teacher_output = sum(teacher_outputs) / len(teacher_outputs)
        
        kd_loss = self.kldiv_loss(
            torch.log_softmax(student_output / self.temperature, dim=1),
            torch.softmax(avg_teacher_output / self.temperature, dim=1)
        )
        
        ce_loss = self.ce_loss(student_output, labels)
        return kd_loss + ce_loss
