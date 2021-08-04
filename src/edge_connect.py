import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset import Dataset
from src.models import EdgeModel,SegModel, InpaintingModel
from src.utils import Progbar, create_dir, stitch_images, imsave,tensor2label,imsave_np
from src.metrics import PSNR, EdgeAccuracy
from src.metrics import *



class SegEdgeConnect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'seg'   
        elif config.MODEL == 2:
            model_name = 'edge'
        elif config.MODEL == 3:
            model_name = 'inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'
        elif config.MODEL == 6:
            model_name = 'seg_edge_inpaint'

        self.debug = False
        self.model_name = model_name
        

        self.seg_model = SegModel(config).to(config.DEVICE)
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        
        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST,config.TEST_SEG_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_SEG_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.TEST_SEG_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 1:
            self.seg_model.load()

        elif self.config.MODEL == 2:
            self.edge_model.load()

        elif self.config.MODEL == 3:
            self.inpaint_model.load()
        else:
            self.seg_model.load()
            self.edge_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.seg_model.save()

        elif self.config.MODEL == 2:
            self.edge_model.save()

        elif  self.config.MODEL == 3:
            self.inpaint_model.save()
        else:
            print('save 4')
            self.seg_model.save()
            self.edge_model.save()
            self.inpaint_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )
        

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        print(model)
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.seg_model.train()
                self.edge_model.train()
                self.inpaint_model.train()


                images, images_gray, edges, segs, masks = self.cuda(*items)


                # seg model
                if model == 1:
                    # train
                   
                    segpreds, gen_loss, dis_loss, logs = self.seg_model.process(images, segs, masks)

                    #segpreds_merged = (segpreds * masks) + (segs * (1 - masks))
                    segpreds_merged = segpreds * masks
                    segcs = segs * masks



                    pa=0
                    ma=0
                    mi=0
                    fwi=0

                    for b in range(segpreds_merged.size(0)):
                        seg_np,seg_npc = tensor2label(segcs[b],20, tile=False)
                        segp_np,segp_npc = tensor2label(segpreds_merged[b], 20, tile=False)
                        seg_np = np.squeeze(seg_np)
                        segp_np = np.squeeze(segp_np)

                    

                        pa +=pixel_accuracy(segp_np,seg_np)
                        ma +=mean_accuracy(segp_np,seg_np)
                        mi +=mean_IU(segp_np,seg_np)
                        fwi+=frequency_weighted_IU(segp_np,seg_np)
                    
                    #mean = torch.mean(masks).cpu().numpy()

                    logs.append(('1',pa/self.config.BATCH_SIZE))
                    logs.append(('2',ma/self.config.BATCH_SIZE))
                    logs.append(('3',mi/self.config.BATCH_SIZE))
                    logs.append(('4',fwi/self.config.BATCH_SIZE))

                    # metrics
                    """
                    precision, recall = self.segacc(edges * masks, edgepreds * masks)
                    logs.append(('precision', precision.item()))
                    logs.append(('recall', recall.item()))
                    """

                    # backward
                    self.seg_model.backward(gen_loss, dis_loss)
                    iteration = self.seg_model.iteration
                
                # edge model
                elif model == 2:
                    # train
                    edgepreds, gen_loss, dis_loss, logs = self.edge_model.process(images_gray,segs,edges, masks)
                    edgepreds_merged = (edgepreds * masks) + (edges * (1 - masks))
                    # metrics
                    
                    precision, recall = self.edgeacc(edges * masks, edgepreds * masks)
                    logs.append(('precision', precision.item()))
                    logs.append(('recall', recall.item()))
                    

                    # backward
                    #self.edge_model.backward(gen_loss, dis_loss)
                    iteration = self.edge_model.iteration

                # inpaint model
                elif model == 3:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, segs,edges, masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration
                # inpaint with seg and edge model           
                # joint model
                elif model == 4:
                    s_segpreds, s_gen_loss, s_dis_loss, s_logs = self.seg_model.process(images, segs, masks)
                    #e_edgepreds = e_edgepreds * masks + edges * (1 - masks)
                    #s_segpreds = s_segpreds * masks + segs * (1 - masks)

                    e_edgepreds, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, s_segpreds.detach(),edges,  masks)
                    #e_edgepreds = e_edgepreds * masks + edges * (1 - masks)

                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, s_segpreds.detach(),e_edgepreds.detach(), masks)
                    outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    precision, recall = self.edgeacc(edges * masks, e_edgepreds * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = s_logs + e_logs + i_logs

                    # backward
                    self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                    self.seg_model.backward(s_gen_loss, s_dis_loss)
                    self.edge_model.backward(e_gen_loss, e_dis_loss)                  
                    iteration = self.inpaint_model.iteration
                elif model == 6:
                    segpreds, gen_loss, dis_loss, logs = self.seg_model.process(images, segs, masks)
                    segpreds_merged = segpreds * masks + segs * (1 - masks)
                    edgepreds, gen_loss, dis_loss, logs = self.edge_model.process(images_gray,segpreds_merged,edges, masks)
                    edgepreds = edgepreds * masks + edges * (1 - masks)
                    precision, recall = self.edgeacc(edges * masks, edgepreds * masks)
                    logs.append(('precision', precision.item()))
                    logs.append(('recall', recall.item()))
                    iteration = self.seg_model.iteration
                else :
                    # train
                    segpreds = self.seg_model(images, segs, masks)

                    segpreds = segpreds * masks + segs * (1 - masks)

                    edgepreds = self.edge_model(images_gray, segpreds.detach(),edges,  masks)
                    

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edgepreds.detach(),segpreds.detach(), masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                #progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                progbar.add(len(images), values=logs)

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
                    #print("save")

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.edge_model.eval()
        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))


            # inpaint model
            elif model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # inpaint with edge model
            elif model == 3:
                # eval
                outputs = self.edge_model(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # joint model
            else:
                # eval
                e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                e_logs.append(('pre', precision.item()))
                e_logs.append(('rec', recall.item()))
                i_logs.append(('psnr', psnr.item()))
                i_logs.append(('mae', mae.item()))
                logs = e_logs + i_logs


            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def test(self):
        self.seg_model.eval()
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)
        
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:

            
            name = self.test_dataset.load_name(index)
            
            print(str(index)+"：  loading "+name)
            fname, fext = name.split('.')
            images, images_gray, edges, segs, masks = self.cuda(*items)
            index += 1

            if self.debug:
                
                seg_np,seg_npc = tensor2label(segs, 20, tile=False)
                edge = self.postprocess(1-edges)[0]
                mask = self.postprocess(masks)[0]

                masks_np = masks.permute(0, 2, 3, 1).int()[0].cpu().numpy()

                segm_npc = (seg_npc * (1 - masks_np)+ masks_np*255.0)
               
                

                edgem = self.postprocess(1-edges*(1 - masks)-masks)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                
                pathi = os.path.join(self.results_path, fname+"_image."+ fext)
                image = self.postprocess(images)[0]

                imsave(image, pathi)
                imsave_np(seg_np, os.path.join(self.results_path, fname + '_seg.' + fext))
                imsave_np(seg_npc, os.path.join(self.results_path, fname + '_segc.' + fext))
                imsave(edge, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(mask, os.path.join(self.results_path, fname + '_mask.' + fext))
                imsave_np(segm_npc, os.path.join(self.results_path, fname + '_segm.' + fext))
                imsave(edgem, os.path.join(self.results_path, fname + '_edgem.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_imagem.' + fext))

            # edge model
            if model == 1:
                segpreds = self.seg_model(images,segs, masks)
                segpreds_merged = ((segpreds * masks) + (segs * (1 - masks)))
                segps,segpreds_np = tensor2label(segpreds_merged, 20, tile=False)
                '''
                pa=0
                    #ma=0
                    #mi=0
                    #fwi=0

                    for b in range(segpreds_merged.size(0)):
                        seg_np,seg_npc = tensor2label(segcs[b],20, tile=False)
                        segp_np,segp_npc = tensor2label(segpreds_merged[b], 20, tile=False)
                        seg_np = np.squeeze(seg_np)
                        segp_np = np.squeeze(segp_np)

                    

                        pa +=pixel_accuracy(segp_np,seg_np)
                        #ma +=mean_accuracy(segp_np,seg_np)
                        #mi +=mean_IU(segp_np,seg_np)
                        #fwi+=frequency_weighted_IU(segp_np,seg_np)
                    
                    #mean = torch.mean(masks).cpu().numpy()

                    logs.append(('1',pa/self.config.BATCH_SIZE))
                    #logs.append(('2',ma/self.config.BATCH_SIZE)/torch.mean(masks))
                    #logs.append(('3',mi/self.config.BATCH_SIZE)/torch.mean(masks))
                    #logs.append(('4',fwi/self.config.BATCH_SIZE))
                '''

                #segpreds_merged = self.postprocess(segpreds_merged)[0]
                path1 = os.path.join(self.results_path,fname+'_segp.'+ fext) 
                path = os.path.join(self.results_path,fname+'_segpc.'+ fext) 
                
                imsave_np(segps,path1)
                imsave_np(segpreds_np,path)
                #imsave(segpreds_merged, path)

            elif model == 2:
                edgepreds = self.edge_model(images_gray,segs, edges, masks)
                edgepreds_merged = (edgepreds * masks) + (edges * (1 - masks))
                edgepreds_merged = self.postprocess(edgepreds_merged)[0]
                #segpreds_merged = self.postprocess(segpreds_merged)[0]
                               
                path = os.path.join(self.results_path,fname+'_edgep.'+ fext)
                imsave(edgepreds_merged, path)
                


            # inpaint model
            elif model == 3:
                outputs = self.inpaint_model(images,segs, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                output = self.postprocess(outputs_merged)[0]
                path = os.path.join(self.results_path, fname+"_inpainting."+ fext)
                imsave(output, path)

            # inpaint with edge model / joint model
            else:
                segpreds = self.seg_model(images,segs, masks)
                segpreds_merged = ((segpreds * masks) + (segs * (1 - masks)))
                
                edgepreds = self.edge_model(images_gray,segpreds_merged, edges, masks)
                edgepreds_merged = (edgepreds * masks) + (edges * (1 - masks))

                outputs = self.inpaint_model(images,segpreds_merged, edgepreds_merged, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                
                segpreds_np,segpreds_npc = tensor2label(segpreds_merged,20 , tile=False)
                output = self.postprocess(outputs_merged)[0]
                edgepreds_merged = self.postprocess(1-edgepreds_merged)[0]
                
                               
                path1 = os.path.join(self.results_path,fname+"_edgep."+ fext)
                path2 = os.path.join(self.results_path, fname+ '_segp.'+fext)
                path = os.path.join(self.results_path, fname+ "_inpainting."+fext)

                
                imsave(edgepreds_merged, path1)
                imsave_np(segpreds_npc, path2)
                imsave(output, path)



        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks = self.cuda(*items)

        # edge model
        if model == 1:
            iteration = self.edge_model.iteration
            inputs = (images_gray * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks)
            outputs_merged = (outputs * masks) + (edges * (1 - masks))

        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        # inpaint with edge model / joint model
        else:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)
    
    def preprocess(self, label_map):
        pass

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()


