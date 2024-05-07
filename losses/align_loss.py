import torch
from losses.style.style_loss import StyleLoss

class AlignLossBuilder(torch.nn.Module):

    # Changes number of classes to classes in the segmantation model
    def __init__(self, opt, num_classes=16, device="cuda"):
        super(AlignLossBuilder, self).__init__()

        self.opt = opt
        self.parsed_loss = [[opt.l2_lambda, 'l2'], [opt.percept_lambda, 'percep']]

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.style = StyleLoss(distance="l2", VGG16_ACTIVATIONS_LIST=[3, 8, 15, 22], normalize=False).to(device)
        self.style.eval()

        self.binary_ce = torch.nn.BCEWithLogitsLoss()
        
        tmp = torch.zeros(num_classes).to(device)
        tmp[0] = 1
        self.cross_entropy_wo_background = torch.nn.CrossEntropyLoss(weight=1 - tmp)
        self.cross_entropy_only_background = torch.nn.CrossEntropyLoss(weight=tmp)

        self.cosine_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    def cross_entropy_loss(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy(down_seg, target_mask)
        return loss

    def binary_ce_loss(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.binary_ce(down_seg, target_mask)
        return loss

    def style_loss(self, im1, im2, mask1, mask2):
        loss = self.opt.style_lambda * self.style(im1 * mask1, im2 * mask2, mask1=mask1, mask2=mask2)
        return loss

    def cross_entropy_loss_wo_background(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy_wo_background(down_seg, target_mask)
        return loss

    def cross_entropy_loss_only_background(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy_only_background(down_seg, target_mask)
        return loss

    def cosine_similarity_loss(self, hair_perc, target_perc):
        non_zero_indexs = torch.flatten(target_perc != 0)

        flat_hair_perc = torch.flatten(hair_perc)[non_zero_indexs]
        flat_target_perc = torch.flatten(target_perc)[non_zero_indexs]
        
        loss = 1 - self.opt.hair_perc_lambda * self.cosine_loss(flat_hair_perc, flat_target_perc)
        return loss / flat_hair_perc.shape[0]
        