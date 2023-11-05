import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch import  optim
from torch.utils.data import DataLoader
import torchvision.models as models 
import glob
from PIL import Image as PILImage 
import natsort

class ResidualDenseBlock(nn.Module):
    def __init__(self,nf=64,gc=32,bias=True):
        super(ResidualDenseBlock,self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, padding=1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, padding=1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, padding=1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x,x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x,x1,x2),dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x,x1,x2,x3),dim=1)))
        x5 = self.lrelu(self.conv5(torch.cat((x,x1,x2,x3,x4),dim=1)))

        return x5 * 0.2 + x

class Generator(nn.Module):
    def __init__(self, nf=64):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, nf, kernel_size=3,padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.blockLayer = nn.Sequential(
            ResidualDenseBlock(),
            ResidualDenseBlock(),
            ResidualDenseBlock()
        )

        self.pixelShuffle = nn.Sequential(
            nn.Conv2d(nf,nf,kernel_size=3,padding=1),
            nn.Conv2d(nf,nf*4,kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(nf,nf*4,kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(nf,nf,kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(nf, 3, kernel_size=3, stride=1, padding=1)
        )
    def forward(self, x):
        x = self.conv1(x)
        skip = self.relu(x)
        x = self.blockLayer(skip)
        x = self.pixelShuffle(x + skip)
        return x

class Discreminator(nn.Module):
    def __init__(self):
        super(Discreminator, self).__init__()
        self.group_size = 4
        self.num_channels = 1

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.epilouge = nn.Sequential(
            nn.Conv2d(513, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.net(x)
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F
        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        x = self.epilouge(x)
        return x
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
        blocks.append(models.vgg16(pretrained=True).features[23:30].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)

    def forward(self, fakeFrame, frameY):
        fakeFrame_ = (fakeFrame - self.mean) / self.std
        frameY = (frameY - self.mean) / self.std
        loss = 0.0
        x = fakeFrame_
        y = frameY
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + torch.nn.functional.l1_loss(x, y)
        return loss
    
def relative(real, fake):
    real = real - fake.mean(0, keepdim=True)
    fake = fake - real.mean(0, keepdim=True)
    return real, fake

def main():
    num_epochs=2000
    DiscriminatorLR=1.5e-4
    GeneratorLR=1.5e-4
    batch_size = 48
    hr_transform = transforms.Compose([transforms.RandomCrop(size=(256, 256)), 
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                    transforms.RandomGrayscale(p=0.1),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.ToTensor()])
    hr_img = torchvision.datasets.ImageFolder(root="./data", transform=hr_transform)
    eval_img = torchvision.datasets.ImageFolder(root="./valid_data",transform=hr_transform)
    hrimg_loader = DataLoader(hr_img, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_img, batch_size=batch_size, shuffle=True, drop_last=True)
    torch.autograd.set_detect_anomaly(True)
    D = Discreminator()
    G = Generator()
    D.train().cuda()
    G.train().cuda()

    D_optimizer = torch.optim.Adam(D.parameters(), lr=DiscriminatorLR, betas=(0.9, 0.999))
    G_optimizer = torch.optim.Adam(G.parameters(), lr=GeneratorLR, betas=(0.9, 0.999))

    D_scheduler = optim.lr_scheduler.StepLR(D_optimizer, step_size=1, gamma=0.9)
    G_scheduler = optim.lr_scheduler.StepLR(G_optimizer, step_size=1, gamma=0.9)

    realLabel = torch.ones(batch_size, 1, dtype=torch.float16).cuda()
    fakeLabel = torch.zeros(batch_size, 1, dtype=torch.float16).cuda()
    BCE = torch.nn.BCELoss()
    VggLoss = VGGPerceptualLoss().cuda()
    scaler = torch.cuda.amp.GradScaler()
    adversarial_loss = torch.nn.BCEWithLogitsLoss().cuda()
    torch.cuda.empty_cache()
    G_label_loss_result = []
    G_loss_result = []
    v_loss_result = []
    d_loss_result = []
    for i in range(num_epochs):
        print('epoch:',i+1,'/',num_epochs)
        for hr_data, _ in hrimg_loader:
            hr_data = hr_data.cuda()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                lr_data = torch.nn.functional.interpolate(hr_data, (64,64), mode='bicubic').cuda()
                fakeFrame = G(lr_data)
                
                DReal = D(hr_data)
                DFake = D(fakeFrame)
                DReal, DFake = relative(DReal, DFake)
                real_loss = adversarial_loss(DReal ,realLabel)
                fake_loss = adversarial_loss(DFake ,fakeLabel)
                
                D_loss =  (real_loss + fake_loss)/2
            with torch.autograd.detect_anomaly():
                scaler.scale(D_loss).backward(retain_graph=True)
                scaler.step(D_optimizer)
                scaler.update()
                D_optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                lr_data = torch.nn.functional.interpolate(hr_data, (64,64), mode='bicubic').cuda()
                fakeFrame = G(lr_data)
                DReal = D(hr_data.detach()) + 1e-8
                DFake = D(fakeFrame.detach()) + 1e-8

                DReal, DFake = relative(DReal, DFake)
                G_label_loss = adversarial_loss(DFake.clone(), realLabel.clone())

                v_loss = VggLoss(fakeFrame, hr_data)

                G_loss = v_loss + 1e-3 * G_label_loss.clone()
                
            scaler.scale(G_loss.clone()).backward()
            scaler.step(G_optimizer)
            scaler.update()
            G_optimizer.zero_grad()
        
        for eval_data, _ in eval_loader:
            eval_data = eval_data.cuda()
            lr_data = torch.nn.functional.interpolate(eval_data, (64,64), mode='bicubic').cuda()
            fakeFrame = G(lr_data)
            DReal = D(eval_data.detach())
            DFake = D(fakeFrame.detach())
            DReal, DFake = relative(DReal, DFake)
            D_loss = (adversarial_loss(DFake, fakeLabel) + adversarial_loss(DReal, realLabel)) / 2
            G_label_loss = adversarial_loss(DFake.clone(), realLabel.clone())
            v_loss = VggLoss(fakeFrame, hr_data)
            G_loss = v_loss + 1e-3 * G_label_loss.clone()

            
        print(f"D_loss:{D_loss}")
        print(f"G_loss:{G_loss}")
        fout = open("./loss/esrgan_gene.txt",'a')
        fout.write(str(G_loss.cpu().detach().numpy().copy())+'\n')
        fout.close()
        fout = open("./loss/esrgan_disc.txt",'a')
        fout.write(str(D_loss.cpu().detach().numpy().copy())+'\n')
        fout.close()
        torch.save(G.state_dict(), "./ckpt/esrgan_gene_ckpt.pth")
        torch.save(D.state_dict(), "./ckpt/esrgan_disc_ckpt.pth")
        if i == str(1200):
            torch.save(G.state_dict(), "./ckpt/esrgan_gene_1200.pth")
            torch.save(D.state_dict(), "./ckpt/esrgan_disc1200.pth")
        if i == str(1500):
            torch.save(G.state_dict(), "./ckpt/esrgan_gene_1500.pth")
            torch.save(D.state_dict(), "./ckpt/esrgan_disc_1500.pth")
        if i == str(2000):
            torch.save(G.state_dict(), "./ckpt/esrgan_gene_2000.pth")
            torch.save(D.state_dict(), "./ckpt/esrgan_disc_2000.pth")
            
        j = 0
    torch.save(G.state_dict(), "./esrsan_gene_final.pth")
    torch.save(D.state_dict(), "./esrsan_disc_final.pth")

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms.functional as F
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    from torch import  optim
    from torch.utils.data import DataLoader
    import torchvision.models as models 
    import glob
    from PIL import Image as PILImage 
    import natsort

    class ResidualDenseBlock(nn.Module):
        def __init__(self,nf=64,gc=32,bias=True):
            super(ResidualDenseBlock,self).__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, padding=1, bias=bias)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding=1, bias=bias)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, padding=1, bias=bias)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, padding=1, bias=bias)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, padding=1, bias=bias)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x,x1), dim=1)))
            x3 = self.lrelu(self.conv3(torch.cat((x,x1,x2),dim=1)))
            x4 = self.lrelu(self.conv4(torch.cat((x,x1,x2,x3),dim=1)))
            x5 = self.lrelu(self.conv5(torch.cat((x,x1,x2,x3,x4),dim=1)))

            return x5 * 0.2 + x

    class Generator(nn.Module):
        def __init__(self, nf=64):
            super(Generator, self).__init__()
            self.conv1 = nn.Conv2d(3, nf, kernel_size=3,padding=1)
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

            self.blockLayer = nn.Sequential(
                ResidualDenseBlock(),
                ResidualDenseBlock(),
                ResidualDenseBlock()
            )

            self.pixelShuffle = nn.Sequential(
                nn.Conv2d(nf,nf,kernel_size=3,padding=1),
                nn.Conv2d(nf,nf*4,kernel_size=3,padding=1),
                nn.LeakyReLU(negative_slope=0.2,inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(nf,nf*4,kernel_size=3,padding=1),
                nn.LeakyReLU(negative_slope=0.2,inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(nf,nf,kernel_size=3,padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(nf, 3, kernel_size=3, stride=1, padding=1)
            )
        def forward(self, x):
            x = self.conv1(x)
            skip = self.relu(x)
            x = self.blockLayer(skip)
            x = self.pixelShuffle(x + skip)
            return x

    class Discreminator(nn.Module):
        def __init__(self):
            super(Discreminator, self).__init__()
            self.group_size = 4
            self.num_channels = 1

            self.net = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 256, kernel_size=3, stride=2,padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),

                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2)
            )
            self.epilouge = nn.Sequential(
                nn.Conv2d(513, 512, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Flatten(),
                nn.Linear(2048, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512,1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = self.net(x)
            N, C, H, W = x.shape
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
            F = self.num_channels
            c = C // F
            y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
            y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
            y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
            y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
            y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
            y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
            y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
            x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
            x = self.epilouge(x)
            return x
        
    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.shape[0], -1)

    class VGGPerceptualLoss(torch.nn.Module):
        def __init__(self):
            super(VGGPerceptualLoss, self).__init__()
            blocks = []
            blocks.append(models.vgg16(pretrained=True).features[:4].eval())
            blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
            blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
            blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
            blocks.append(models.vgg16(pretrained=True).features[23:30].eval())
            for bl in blocks:
                for p in bl:
                    p.requires_grad = False
            self.blocks = torch.nn.ModuleList(blocks)
            self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
            self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)

        def forward(self, fakeFrame, frameY):
            fakeFrame_ = (fakeFrame - self.mean) / self.std
            frameY = (frameY - self.mean) / self.std
            loss = 0.0
            x = fakeFrame_
            y = frameY
            for block in self.blocks:
                x = block(x)
                y = block(y)
                loss = loss + torch.nn.functional.l1_loss(x, y)
            return loss
        
    def relative(real, fake):
        real = real - fake.mean(0, keepdim=True)
        fake = fake - real.mean(0, keepdim=True)
        return real, fake

    def main():
        num_epochs = 1000
        DiscriminatorLR=1.5e-4
        GeneratorLR=1.5e-4
        batch_size = 64
        hr_transform = transforms.Compose([transforms.RandomCrop(size=(256, 256)), 
                                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                        transforms.RandomGrayscale(p=0.1),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.ToTensor()])
        hr_img = torchvision.datasets.ImageFolder(root="./data", transform=hr_transform)
        eval_img = torchvision.datasets.ImageFolder(root="./eval_data",transform=hr_transform)
        hrimg_loader = DataLoader(hr_img, batch_size=batch_size, shuffle=True, drop_last=True)
        eval_loader = DataLoader(eval_img, batch_size=batch_size, shuffle=True, drop_last=True)
        torch.autograd.set_detect_anomaly(True)
        D = Discreminator()
        G = Generator()
        D.train().cuda()
        G.train().cuda()

        D_optimizer = torch.optim.Adam(D.parameters(), lr=DiscriminatorLR, betas=(0.9, 0.999))
        G_optimizer = torch.optim.Adam(G.parameters(), lr=GeneratorLR, betas=(0.9, 0.999))

        D_scheduler = optim.lr_scheduler.StepLR(D_optimizer, step_size=1, gamma=0.9)
        G_scheduler = optim.lr_scheduler.StepLR(G_optimizer, step_size=1, gamma=0.9)

        realLabel = torch.ones(batch_size, 1, dtype=torch.float16).cuda()
        fakeLabel = torch.zeros(batch_size, 1, dtype=torch.float16).cuda()
        BCE = torch.nn.BCELoss()
        VggLoss = VGGPerceptualLoss().cuda()
        scaler = torch.cuda.amp.GradScaler()
        adversarial_loss = torch.nn.BCEWithLogitsLoss().cuda()
        torch.cuda.empty_cache()
        G_label_loss_result = []
        G_loss_result = []
        v_loss_result = []
        d_loss_result = []
        for i in range(num_epochs):
            print('epoch:',i+1,'/',num_epochs)
            for hr_data, _ in hrimg_loader:
                hr_data = hr_data.cuda()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    lr_data = torch.nn.functional.interpolate(hr_data, (64,64), mode='bicubic').cuda()
                    fakeFrame = G(lr_data)
                    
                    DReal = D(hr_data)
                    DFake = D(fakeFrame)
                    DReal, DFake = relative(DReal, DFake)
                    real_loss = adversarial_loss(DReal ,realLabel)
                    fake_loss = adversarial_loss(DFake ,fakeLabel)
                    
                    D_loss =  (real_loss + fake_loss)/2
                with torch.autograd.detect_anomaly():
                    scaler.scale(D_loss).backward(retain_graph=True)
                    scaler.step(D_optimizer)
                    scaler.update()
                    D_optimizer.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    lr_data = torch.nn.functional.interpolate(hr_data, (64,64), mode='bicubic').cuda()
                    fakeFrame = G(lr_data)
                    DReal = D(hr_data.detach()) + 1e-8
                    DFake = D(fakeFrame.detach()) + 1e-8

                    DReal, DFake = relative(DReal, DFake)
                    G_label_loss = adversarial_loss(DFake.clone(), realLabel.clone())

                    v_loss = VggLoss(fakeFrame, hr_data)

                    G_loss = v_loss + 1e-3 * G_label_loss.clone()
                    
                scaler.scale(G_loss.clone()).backward()
                scaler.step(G_optimizer)
                scaler.update()
                G_optimizer.zero_grad()
            
            for eval_data, _ in eval_loader:
                eval_data = eval_data.cuda()
                lr_data = torch.nn.functional.interpolate(eval_data, (64,64), mode='bicubic').cuda()
                fakeFrame = G(lr_data)
                DReal = D(eval_data.detach())
                DFake = D(fakeFrame.detach())
                DReal, DFake = relative(DReal, DFake)
                D_loss = (adversarial_loss(DFake, fakeLabel) + adversarial_loss(DReal, realLabel)) / 2
                G_label_loss = adversarial_loss(DFake.clone(), realLabel.clone())
                v_loss = VggLoss(fakeFrame, hr_data)
                G_loss = v_loss + 1e-3 * G_label_loss.clone()
            print(f"D_loss:{D_loss}")
            print(f"G_loss:{G_loss}")
            fout = open("./loss/esrgan_gene.txt",'a')
            fout.write(str(G_loss.cpu().detach().numpy().copy())+'\n')
            fout.close()
            fout = open("./loss/esrgan_disc.txt",'a')
            fout.write(str(D_loss.cpu().detach().numpy().copy())+'\n')
            fout.close()
            torch.save(G.state_dict(), "./ckpt/esrgan_gene_ckpt.pth")
            torch.save(D.state_dict(), "./ckpt/esrgan_disc_ckpt.pth")
            if i == str(250):
                torch.save(G.state_dict(), "./ckpt/esrgan_gene_250.pth")
                torch.save(D.state_dict(), "./ckpt/esrgan_disc_250.pth")
            if i == str(500):
                torch.save(G.state_dict(), "./ckpt/esrgan_gene_500.pth")
                torch.save(D.state_dict(), "./ckpt/esrgan_disc_500.pth")
            if i == str(750):
                torch.save(G.state_dict(), "./ckpt/esrgan_gene_750.pth")
                torch.save(D.state_dict(), "./ckpt/esrgan_disc_750.pth")
                
            j = 0
        torch.save(G.state_dict(), "./ckpt/esrsan_gene_finish.pth")
        torch.save(D.state_dict(), "./ckpt/esrsan_disc_finish.pth")

if __name__ == "__main__":
    main()