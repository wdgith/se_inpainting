import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, EdgeGenerator, SegGenerator,Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.edge_gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.seg_gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'seggenerator': self.seggenerator.state_dict()
        }, self.self.seg_gen_weights_path)
        torch.save({
            'iteration': self.iteration,
            'edgegenerator': self.edgegenerator.state_dict()
        }, self.edge_gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)

class EdgeModel(nn.Module):
    def __init__(self, config):
        super(EdgeModel, self).__init__()
        self.name = 'edge_Model'
        self.config = config
        self.iteration = 0
        self.gen_weights_path = os.path.join(config.PATH, self.name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        #super(SegEdgeModel, self).__init__('EdgeModel', config)


        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        generator = EdgeGenerator(use_spectral_norm=True)
        ##############channel number
        discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')

        if len(config.GPU) > 1:
            """
            edgegenerator = nn.DataParallel(edgegenerator, config.GPU)
            seggenerator = nn.DataParallel(seggenerator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
            """
            generator = nn.DataParallel(generator, device_ids=range(len(config.GPU)))
            discriminator = nn.DataParallel(discriminator, device_ids=range(len(config.GPU)))
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )
        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.DE),
            betas=(config.BETA1, config.BETA2)
        )
    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)
            if len(self.config.GPU) > 1:
                self.generator.module.load_state_dict(data['generator'])
            else:
                self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            if len(self.config.GPU) > 1:
                self.discriminator.module.load_state_dict(data['discriminator'])
            else :
                self.discriminator.load_state_dict(data['discriminator'])
    def save(self):
        print('\nsaving %s...\n' % self.name)
        if len(self.config.GPU) > 1: 
            torch.save({'iteration': self.iteration,'generator': self.generator.module.state_dict()}, self.gen_weights_path)
            torch.save({'discriminator': self.discriminator.module.state_dict()}, self.dis_weights_path)
        else:
            torch.save({'iteration': self.iteration,'generator': self.generator.state_dict()}, self.gen_weights_path)
            torch.save({'discriminator': self.discriminator.state_dict()}, self.dis_weights_path)
    def process(self, images,segs, edges, masks):
        self.iteration += 1


        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, segs,edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        #dis_input_real = edges
        #dis_input_fake =
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        #gen_input_fake = outputs

        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)* self.config.E_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # gen_l1_loss = self.l1_loss(outputs,segs)*self.config.E_L1_LOSS_WEIGHT/ torch.mean(masks)
        # gen_loss += gen_l1_loss


        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.E_FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss


        # create logs
        logs = [
            ("2_d1", dis_loss.item()),            
            ("2_g1", gen_gan_loss.item()),
            ("2_fm", gen_fm_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs
    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()
    '''
    def process(self, images,images_gray,segs,edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        #images,images_gray, edges,segs, masks
        edgepred= self(images,images_gray,segs, edges, masks)

        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        #dis_input_real = torch.cat((images, segs, edges), dim=1)
        #dis_input_fake = torch.cat((images, segs,edgepred.detach()), dim=1)
        dis_input_real = torch.cat((images_gray, edges), dim=1)
        dis_input_fake = torch.cat((images_gray, edgepred.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = torch.cat((images_gray,edgepred.detach()), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss


        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss


        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]
        return edgepred, gen_loss, dis_loss, logs
        '''
    def forward(self, images, segs,edges, masks):
        edges_masked = (edges * (1 - masks))
        images_gray_masked = (images * (1 - masks)) + masks
        #images_gray_masked = (images_gray * (1 - masks)) + masks
        inputs = torch.cat((images_gray_masked,segs,edges_masked, masks), dim=1)
        edgepred = self.generator(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]      
        return edgepred

class SegModel(nn.Module):
    def __init__(self, config):
        super(SegModel, self).__init__()
        self.name = 'seg_Model'
        self.config = config
        self.iteration = 0
        self.gen_weights_path = os.path.join(config.PATH, self.name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        #super(SegEdgeModel, self).__init__('EdgeModel', config)


        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        generator = SegGenerator(use_spectral_norm=True)
        ##############channel number
        discriminator = Discriminator(in_channels=22, use_sigmoid=config.GAN_LOSS != 'hinge')

        if len(config.GPU) > 1:
            """
            edgegenerator = nn.DataParallel(edgegenerator, config.GPU)
            seggenerator = nn.DataParallel(seggenerator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
            """
            generator = nn.DataParallel(generator, device_ids=range(len(config.GPU)))
            discriminator = nn.DataParallel(discriminator, device_ids=range(len(config.GPU)))
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )
        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.DS),
            betas=(config.BETA1, config.BETA2)
        )

    
    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)
            if len(self.config.GPU) > 1:
                self.generator.module.load_state_dict(data['generator'])
            else:
                self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)
            if len(self.config.GPU) > 1:
                self.discriminator.module.load_state_dict(data['discriminator'])
            else :
                self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        if len(self.config.GPU) > 1: 
            torch.save({'iteration': self.iteration,'generator': self.generator.module.state_dict()}, self.gen_weights_path)
            torch.save({'discriminator': self.discriminator.module.state_dict()}, self.dis_weights_path)
        else:
            torch.save({'iteration': self.iteration,'generator': self.generator.state_dict()}, self.gen_weights_path)
            torch.save({'discriminator': self.discriminator.state_dict()}, self.dis_weights_path)
    
    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()
    '''
    def process(self, images,segs, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        #images,images_gray, edges,segs, masks
        segpred = self(images,segs, masks)

        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = torch.cat((images, segs), dim=1)
        dis_input_fake = torch.cat((images, segpred.detach()), dim=1)
        
        #dis_input_real = segs
        #dis_input_fake = segpred.detach()
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        #gen_input_fake = torch.cat((images,segpred.detach()), dim=1)
        gen_input_fake = segpred
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss


        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss


        # create logs
        logs = [
            ("l_dr", dis_real_loss.item()),
            ("l_df", dis_fake_loss.item()),
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]

        return segpred, gen_loss, dis_loss, logs
    '''
    def process(self, images, segs, masks):
        self.iteration += 1


        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, segs, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = torch.cat((images, segs), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        #dis_input_real = segs
        #dis_input_fake = outputs.detach()
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        #gen_input_fake = outputs
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)*self.config.S_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        #gen_l1_loss=self.l1_loss(outputs,segs)*self.config.S_L1_LOSS_WEIGHT/ torch.mean(masks)
        #gen_loss += gen_l1_loss
        


        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        #gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_fm_loss = gen_fm_loss * self.config.S_FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss


        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_gadv", gen_gan_loss.item()),
            ("l_gfm", gen_fm_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images,segs, masks):
        segs_masked = (segs * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        #images_gray_masked = (images_gray * (1 - masks)) + masks
        inputs = torch.cat((images_masked, segs_masked, masks), dim=1)
        segpred = self.generator(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]      
        return segpred

class InpaintingModel(nn.Module):
    def __init__(self, config):
        super(InpaintingModel, self).__init__()
        self.name = 'Inpainting_Model'
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, self.name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')

        #super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            """
            generator = nn.DataParallel(generator, [config.GPU])
            discriminator = nn.DataParallel(discriminator , config.GPU)"""
            generator = nn.DataParallel(generator, [0,1])
            discriminator = nn.DataParallel(discriminator , [0,1])


        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )
    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading inpainting_generator...' )

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            if len(self.config.GPU) > 1:
                self.generator.module.load_state_dict(data['generator'])
            else:
                self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']
        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading inpainting_discriminator...' )

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            if len(self.config.GPU) > 1:
                self.discriminator.module.load_state_dict(data['discriminator'])
            else :
                self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        if len(self.config.GPU) > 1: 
            torch.save({'iteration': self.iteration,'generator': self.generator.module.state_dict()}, self.gen_weights_path)
            torch.save({'discriminator': self.discriminator.module.state_dict()}, self.dis_weights_path)
        else:
            torch.save({'iteration': self.iteration,'generator': self.generator.state_dict()}, self.gen_weights_path)
            torch.save({'discriminator': self.discriminator.state_dict()}, self.dis_weights_path)

    def process(self, images,  segs,edges,masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, segs,edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss


        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss


        # create logs
        logs = [
            ("3_d2", dis_loss.item()),
            ("3_g2", gen_gan_loss.item()),
            ("3_l1", gen_l1_loss.item()),
            ("3_per", gen_content_loss.item()),
            ("3_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, segs, edges,masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked,segs, edges), dim=1)
        outputs = self.generator(inputs)                                    # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()
