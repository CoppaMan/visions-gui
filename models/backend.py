import threading
import random
import math
import logging

import torch
import torchvision
import torch_optimizer as optim
import torch.nn as nn
import clip
import numpy as np

from worker.thread import ThreadedWorker


class VisionsBackend(ThreadedWorker):
    def __init__(self):
        ThreadedWorker.__init__(self)

        # Backend members
        self.progress = [0,0]
        self.save_interval = 200

        # General model members
        self.texts = [
            {
                "text": '',
                "weight": 1.0,
            },{ # Not really strong enough to remove all signatures... but I'm ok with small ones
                "text":"text",
                "weight":-0.5
            }
        ]

        self.initial_image = None #@param {type:"string"}

        if self.initial_image == "":
            self.initial_image = "None"

        self.seed = None

        #@markdown Image prompts
        target_image = "None" #@param {type:"string"}
        target_image_weight = 0.2 #@param {type:"number"}

        if target_image == "None" or "":
            self.images = []
        else:
            self.images = [
                {
                    "fpath": target_image,
                    "weight": target_image_weight,
                    "cuts": 16,
                    "noise": 0.0
                }
            ]


        #@markdown Number of times to run
        images_n = 1 #@param {type:"number"}

        #@markdown Pixel art works only in Fourier and Direct Visions
        self.pixel_art = False #@param {type:"boolean"}

    def set_texts(self, texts):
        '''

        '''
        if isinstance(texts, str):
            self.texts[0]['text'] = texts
        elif isinstance(texts, dict):
            self.texts[0] = texts
        elif isinstance(texts, list):
            self.texts = texts
        else:
            print('err')

        self.calc_perceptors()

    def set_image_detail(self, stage1, stage2):
        self.stages[0]['cycles'] = stage1
        self.stages[1]['cycles'] = stage2

    def set_seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

    def set_save_interval(self, save_interval):
        self.save_interval = save_interval

    def loadPerceptor(self, name):
        model, preprocess = clip.load(name, device="cuda")

        tokens = []
        imgs = []
        for text in self.texts:
            tok = model.encode_text(clip.tokenize(text["text"]).cuda())
            tokens.append( tok )
        
        perceptor = {"model":model, "size": preprocess.transforms[0].size, "tokens": tokens, }
        for img in self.images:
            image = loadImage(img["fpath"])
            if self.resample_image_prompts:
                imgs.append(image)
            else:
                tokens = self.getClipTokens(image, img["cuts"], img["noise"], False, perceptor )
                imgs.append(tokens)
        perceptor["images"] = imgs
        return perceptor

    def calc_perceptors(self):
        self.perceptors = (
            self.loadPerceptor("ViT-B/32"),
            self.loadPerceptor("ViT-B/16"),
            # loadPerceptor("RN50x16"),
        )


class PiramidVisions(VisionsBackend):
    def __init__(self):
        VisionsBackend.__init__(self)

        self.color_space =  "YCoCg" # "RGB"

        # Params for gaussian init noise
        self.chroma_noise_scale = 0.5 # Saturation (0 - 2 is safe but you can go as high as you want)
        self.luma_noise_mean = -0.0 # Brightness (-3 to 3 seems safe but around 0 seems to work better)
        self.luma_noise_scale = 0.5 # Contrast (0-2 is safe but you can go as high as you want)
        self.init_noise_clamp = 8.0 # Turn this down if you're getting persistent super bright or dark spots.

        # High-frequency to low-frequency initial noise ratio. 
        self.chroma_noise_persistence = 1.0
        self.luma_noise_persistence = 1.0

        # This doesn't seem to matter too much except for 'Nearest', which results in very crisp but pixelated images
        # Lanczos is most advanced but also uses the largest kernel so has the biggest problem with image borders
        # Bilinear is fastest (outside of nearest) but can introduce star-like artifacts
        self.pyramid_scaling_mode = "lanczos" # "lanczos" #'bicubic' "nearest" "bilinear"

        # AdamW is real basic and gets the job done
        # RAdam seems to work *extremely well* but seems to introduce some color instability?, use 0.5x lr
        # Yogi is just really blurry for some reason, use 5x + lr
        # Ranger works great. use 3-4x LR
        self.optimizer_type = "Ranger" # "AdamW", "AccSGD","Ranger","RangerQH","RangerVA","AdaBound","AdaMod","Adafactor","AdamP","AggMo","DiffGrad","Lamb","NovoGrad","PID","QHAdam","QHM","RAdam","SGDP","SGDW","Shampoo","SWATS","Yogi"

        # Resample image prompt vectors every iterations
        # Slows things down a lot for very little benefit, don't bother
        self.resample_image_prompts = False

        # Size of the smallest pyramid layer
        aspect_ratio = (4,3)#(3, 4)

        #Add an extra pyramid layer with dims (1, 1) to control global avg color
        add_global_color = True

        # Max dim of the final output image.
        max_dim = 1024

        # Number of layers at different resolutions combined into the final image
        # "optimal" number is log2(max_dim / max(aspect_ratio))
        # Going below that can make things kinda pixelated but still works fine
        # Seems like you can really go as high as you want tho. Not sure if it helps but you *can*
        pyramid_steps = 14

        # Optimizer settings for different training steps
        self.stages = (
                    { #First stage does rough detail.
                "cuts": 2,
                "cycles": 2000,
                "lr_luma": 1.5e-1, #1e-2 for RAdam #3e-2 for adamw
                "decay_luma": 1e-5,
                "lr_chroma": 7.5e-2, #5e-3 for RAdam #1.5e-2 for adamw
                "decay_chroma": 1e-5,
                "noise": 0.2,
                "denoise": 0.5,
                "checkin_interval": 100,
                # "lr_persistence": 0.95, # ratio of small-scale to large-scale detail
                "pyramid_lr_min" : 0.2, # Percentage of small scale detail
                # "lr_scales": [0.25,0.15,0.15,0.15,0.15,0.05,0.05,0.01,0.01], # manually set lr at each level
            }, { # 2nd stage does fine detail and
                "cuts": 2,
                "cycles": 1000,
                "lr_luma": 1.0e-1,
                "decay_luma": 1e-5,
                "lr_chroma": 7.0e-2,
                "decay_chroma": 1e-5,
                "noise": 0.2,
                "denoise": 1.0,
                "checkin_interval": 100,
                # "lr_persistence": 1.0,
                "pyramid_lr_min" : 1
                # "lr_scales": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
            },
        )

        self.calc_layers(aspect_ratio, add_global_color, max_dim, pyramid_steps)

        self.debug_clip_cuts = False

        self.bilinear = torchvision.transforms.functional.InterpolationMode.BILINEAR
        self.bicubic = torchvision.transforms.functional.InterpolationMode.BICUBIC

        torch.autograd.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        self.calc_perceptors()

    def calc_layers(self, aspect_ratio, add_global_color, max_dim, pyramid_steps):
        pyramid_lacunarity = (max_dim / max(aspect_ratio))**(1.0/(pyramid_steps-1))
        scales = [pyramid_lacunarity**step for step in range(pyramid_steps)]
        self.dims = []
        if add_global_color:
            self.dims.append([1,1])
        for step in range(pyramid_steps):
            scale = pyramid_lacunarity**step
            dim = [int(round(aspect_ratio[0] * scale)), int(round(aspect_ratio[1] * scale))]
            # Ensure that no two levels have the same dims
            if len(self.dims) > 0:
                if dim[0] <= self.dims[-1][0]:
                    dim[0] = self.dims[-1][0]+1
                if dim[1] <= self.dims[-1][1]:
                    dim[1] = self.dims[-1][1]+1
            self.dims.append(dim)
        pyramid_steps = len(self.dims)
        for stage in self.stages:
            if "lr_scales" not in stage:
                if "lr_persistence" in stage:
                    persistence = stage["lr_persistence"]
                elif "pyramid_lr_min" in stage:
                    persistence = stage["pyramid_lr_min"]**(1.0/float(pyramid_steps-1))
                else:
                    persistence = 1.0  
                lrs = [persistence**i for i in range(pyramid_steps)]
                sum_lrs = sum(lrs)
                stage["lr_scales"] = [rate / sum_lrs for rate in lrs]

    def normalize_image(self, image):
        R = (image[:,0:1] - 0.48145466) /  0.26862954
        G = (image[:,1:2] - 0.4578275) / 0.26130258 
        B = (image[:,2:3] - 0.40821073) / 0.27577711
        return torch.cat((R, G, B), dim=1)

    @torch.no_grad()
    def loadImage(self, filename):
        data = open(filename, "rb").read()
        image = torch.ops.image.decode_png(torch.as_tensor(bytearray(data)).cpu().to(torch.uint8), 3).cuda().to(torch.float32) / 255.0
        # image = normalize_image(image)
        return image.unsqueeze(0).cuda()

    def getClipTokens(self, image, cuts, noise, do_checkin, perceptor):
        im = self.normalize_image(image)
        cut_data = torch.zeros(cuts, 3, perceptor["size"], perceptor["size"])
        for c in range(cuts):
            angle = random.uniform(-20.0, 20.0)
            img = torchvision.transforms.functional.rotate(im, angle=angle, expand=True, interpolation=self.bilinear)

            padv = im.size()[2] // 8
            img = torch.nn.functional.pad(img, pad=(padv, padv, padv, padv))

            size = img.size()[2:4]
            mindim = min(*size)

            if mindim <= perceptor["size"]-32:
                width = mindim - 1
            else:
                width = random.randint( perceptor["size"]-32, mindim-1 )

            oy = random.randrange(0, size[0]-width)
            ox = random.randrange(0, size[1]-width)
            img = img[:,:,oy:oy+width,ox:ox+width]

            img = torch.nn.functional.interpolate(img, size=(perceptor["size"], perceptor["size"]), mode='bilinear', align_corners=False)
            cut_data[c] = img

        cut_data += noise * torch.randn_like(cut_data, requires_grad=False)

        if self.debug_clip_cuts and do_checkin:
            self.displayImage(cut_data)

        clip_tokens = perceptor['model'].encode_image(cut_data)
        return clip_tokens

    @torch.no_grad()
    def saveImage(self,image, filename):
        # R = image[:,0:1] * 0.26862954 + 0.48145466
        # G = image[:,1:2] * 0.26130258 + 0.4578275
        # B = image[:,2:3] * 0.27577711 + 0.40821073
        # image = torch.cat((R, G, B), dim=1)
        size = image.size()

        image = (image[0].clamp(0, 1) * 255).to(torch.uint8)
        png_data = torch.ops.image.encode_png(image.cpu(), 6)
        open(filename, "wb").write(bytes(png_data))

    # TODO: Use torchvision normalize / unnormalize
    def unnormalize_image(self, image):
        
        R = image[:,0:1] * 0.26862954 + 0.48145466
        G = image[:,1:2] * 0.26130258 + 0.4578275
        B = image[:,2:3] * 0.27577711 + 0.40821073
        
        return torch.cat((R, G, B), dim=1)

    def paramsToImage(self, params_pyramid):
        pix = []
        for c in range(3):
            pixels = torch.zeros_like(params_pyramid[-1][c])
            for i in range(len(params_pyramid)):
                if self.pyramid_scaling_mode == "lanczos":
                    pixels += self.resample(params_pyramid[i][c], params_pyramid[-1][c].shape[2:])
                else:
                    if self.pyramid_scaling_mode == "nearest" or (params_pyramid[i][c].shape[2] == 1 and params_pyramid[i][c].shape[3] == 1):
                        pixels += torch.nn.functional.interpolate(params_pyramid[i][c], size=params_pyramid[-1][c].shape[2:], mode="nearest")
                    else:
                        pixels += torch.nn.functional.interpolate(params_pyramid[i][c], size=params_pyramid[-1][c].shape[2:], mode=pyramid_scaling_mode, align_corners=True)
            pixels = torch.sigmoid(pixels)
            pix.append(pixels)
        if self.color_space == "YCoCg": 
            luma = pix[0]
            Co = pix[1] * 2 - 1
            Cg = pix[2] * 2 - 1
            tmp = luma - Cg/2
            G = Cg + tmp
            B = tmp - Co/2
            R = B + Co
        elif self.color_space == "RGB":
            R = pix[0]
            G = pix[1]
            B = pix[2]
        im_torch = torch.cat((R, G, B), dim=1)
        return im_torch

    def imageToParams(self, image):
        image = image#.clamp(0,1)
        R, G, B = image[:,0:1], image[:,1:2], image[:,2:3]
        luma = R * 0.25 + G * 0.5 + B * 0.25
        Co = R  - B
        tmp = B + Co / 2
        Cg = G - tmp
        luma = tmp + Cg / 2

        nsize = luma.size()[2:4]
        chroma =  torch.cat([Co,Cg], dim=1)
        chroma = torch.logit((chroma / 2.0 + 0.5), eps=1e-8)
        luma = torch.logit(luma, eps=1e-8)
        return luma, chroma 

    @torch.no_grad()
    def displayImage(self, image):
        size = image.size()

        width = size[0] * size[3] + (size[0]-1) * 4
        image_row = torch.zeros( size=(3, size[2], width), dtype=torch.uint8 )

        nw = 0
        for n in range(size[0]):
            image_row[:,:,nw:nw+size[3]] = (image[n,:].clamp(0, 1) * 255).to(torch.uint8)
            nw += size[3] + 4

        jpeg_data = torch.ops.image.encode_png(image_row.cpu(), 6)
        image = display.Image(bytes(jpeg_data))
        display.display( image )

    def lossClip(self, image, cuts, noise, do_checkin):
        losses = []

        max_loss = 0.0
        for text in self.texts:
            max_loss += abs(text["weight"]) * len(self.perceptors)
        for img in self.images:
            max_loss += abs(img["weight"]) * len(self.perceptors)

        for perceptor in self.perceptors:
            clip_tokens = self.getClipTokens(image, cuts, noise, do_checkin, perceptor)
            for t, tokens in enumerate( perceptor["tokens"] ):
                similarity = torch.cosine_similarity(tokens, clip_tokens)
                weight = self.texts[t]["weight"]
                if weight > 0.0:
                    loss = (1.0 - similarity) * weight
                else:
                    loss = similarity * (-weight)
                losses.append(loss / max_loss)

        for img in self.images:
            for i, prompt_image in enumerate(perceptor["images"]):
                if self.resample_image_prompts:
                    img_tokens = self.getClipTokens(prompt_image, self.images[i]["cuts"], self.images[i]["noise"], False, perceptor)
                else:
                    img_tokens = prompt_image
                weight = self.images[i]["weight"] / float(self.images[i]["cuts"])
                for token in img_tokens:
                    similarity = torch.cosine_similarity(token.unsqueeze(0), clip_tokens)
                    if weight > 0.0:
                        loss = (1.0 - similarity) * weight
                    else:
                        loss = similarity * (-weight)
                    losses.append(loss / max_loss)
        return losses

    def lossTV(self, image, strength):
        Y = (image[:,:,1:,:] - image[:,:,:-1,:]).abs().mean()
        X = (image[:,:,:,1:] - image[:,:,:,:-1]).abs().mean()
        loss = (X + Y) * 0.5 * strength
        return loss

    def cycle(self, c, stage, optimizer, params_pyramid):
        do_checkin = (c+1) % stage["checkin_interval"] == 0 or c == 0
        with torch.enable_grad():
            losses = []
            image = self.paramsToImage(params_pyramid)
            losses += self.lossClip( image, stage["cuts"], stage["noise"], do_checkin )
            losses += [self.lossTV( image, stage["denoise"] )]

            loss_total = sum(losses).sum()
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward(retain_graph=False)
            # if c <= warmup_its:
            #   optimizer.param_groups[0]["lr"] = stage["lr_luma"] * c / warmup_its
            #   optimizer.param_groups[1]["lr"] = stage["lr_chroma"] * c / warmup_its
            optimizer.step()

        # if (c+1) % self.save_interval == 0 or c == 0:
            # nimg = self.paramsToImage(params_pyramid)
            # self.saveImage(nimg, f"images/frame_{stage['n']:02}_{c:05}.png")
        if c % self.save_interval == 0:
            TV = losses[-1].sum().item()
            print( "Cycle:", str(stage["n"]) + ":" + str(c), "CLIP Loss:", loss_total.item() - TV, "TV loss:", TV)
            nimg = self.paramsToImage(params_pyramid)
            #displayImage(torch.nn.functional.interpolate(nimg, size=display_size, mode='nearest'))
            self.saveImage(nimg, 'images/' + self.texts[0]["text"].replace(' ', '_') + ".png" )
            #self.notify_image()

    def sinc(self, x):
        return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

    def lanczos(self, x, a):
        cond = torch.logical_and(-a < x, x < a)
        out = torch.where(cond, self.sinc(x) * self.sinc(x/a), x.new_zeros([]))
        return out / out.sum()

    def ramp(self, ratio, width):
        n = math.ceil(width / ratio + 1)
        out = torch.empty([n])
        cur = 0
        for i in range(out.shape[0]):
            out[i] = cur
            cur += ratio
        return torch.cat([-out[1:].flip([0]), out])[1:-1]

    def resample(self, input, size, align_corners=True):
        n, c, h, w = input.shape
        dh, dw = size

        input = input.reshape([n * c, 1, h, w])

        # if dh < h:
        kernel_h = self.lanczos(self.ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = torch.nn.functional.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = torch.nn.functional.conv2d(input, kernel_h[None, None, :, None])

        # if dw < w:
        kernel_w = self.lanczos(self.ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = torch.nn.functional.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = torch.nn.functional.conv2d(input, kernel_w[None, None, None, :])

        input = input.reshape([n, c, h, w])
        return torch.nn.functional.interpolate(input, size, mode='bicubic', align_corners=align_corners)

    def init_optim(self, params_pyramid, stage):
        lr_scales = stage["lr_scales"]
        params = []
        for i in range(len(lr_scales)):
            if self.color_space == "YCoCg":
                params.append({"params": params_pyramid[i][0], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
                params.append({"params": params_pyramid[i][1], "lr":stage["lr_chroma"] * lr_scales[i], "weight_decay":stage["decay_chroma"]})
                params.append({"params": params_pyramid[i][2], "lr":stage["lr_chroma"] * lr_scales[i], "weight_decay":stage["decay_chroma"]})
            elif self.color_space == "RGB":
                params.append({"params": params_pyramid[i][0], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
                params.append({"params": params_pyramid[i][1], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
                params.append({"params": params_pyramid[i][2], "lr":stage["lr_luma"] * lr_scales[i], "weight_decay":stage["decay_luma"]})
        optimizer = getattr(optim, self.optimizer_type, None)(params)
        return optimizer

    def function(self):
        print('Running')
        params_pyramid = []
        if self.initial_image is not None:
            for dim in self.dims:
                pix = []
                for channel in range(3):
                    pix_c = torch.zeros((1,1,dim[0], dim[1]))
                    param_pix = torch.nn.parameter.Parameter( pix_c.cuda(), requires_grad=True)
                    pix.append(param_pix)
                params_pyramid.append(pix)
            image = loadImage(self.initial_image)
            image = torch.nn.functional.interpolate(image, size=self.dims[-1], mode='self.bicubic', align_corners=False)
            pix_1, pix_2, pix_3 = imageToParams(image)
            pix = []
            for channel in range(3):
                param_pix = torch.nn.parameter.Parameter( pix.cuda(), requires_grad=True)
                pix.append(param_pix)
            params_pyramid[-1] = pix
        else:
            for i, dim in enumerate(self.dims):
                if self.color_space == "YCoCg":
                    luma = (torch.randn(size = (1,1,dim[0], dim[1])) * self.luma_noise_scale * self.luma_noise_persistence**i / len(self.dims)).clamp(-self.init_noise_clamp / len(self.dims), self.init_noise_clamp / len(self.dims)) 
                    Co = (torch.randn(size = (1,1,dim[0], dim[1])) * self.chroma_noise_scale * self.chroma_noise_persistence**i / len(self.dims)).clamp(-self.init_noise_clamp / len(self.dims), self.init_noise_clamp / len(self.dims)) 
                    Cg = (torch.randn(size = (1,1,dim[0], dim[1])) * self.chroma_noise_scale * self.chroma_noise_persistence**i / len(self.dims)).clamp(-self.init_noise_clamp / len(self.dims), self.init_noise_clamp / len(self.dims)) 
                    param_luma = torch.nn.parameter.Parameter( luma.cuda(), requires_grad=True)
                    param_co = torch.nn.parameter.Parameter( Co.cuda(), requires_grad=True)
                    param_cg = torch.nn.parameter.Parameter( Cg.cuda(), requires_grad=True)
                    pix = [param_luma, param_co, param_cg]
                elif self.color_space == "RGB":
                    pix = []
                    for channel in range(3):
                        pix_c = (torch.randn(size = (1,1,dim[0], dim[1])) * chroma_noise_scale * chroma_noise_persistence**i / len(self.dims)).clamp(-self.init_noise_clamp / len(self.dims), self.init_noise_clamp / len(self.dims)) 
                        param_pix = torch.nn.parameter.Parameter( pix_c.cuda(), requires_grad=True)
                        pix.append(param_pix)
                params_pyramid.append(pix)

        optimizer = self.init_optim(params_pyramid, self.stages[0])

        for n, stage in enumerate(self.stages):
            stage["n"] = n
            if n > 0:
                for i in range(len(self.dims)):
                    if self.color_space == "YCoCg":
                        print(len(optimizer.param_groups))
                        print(str(i*3+2))
                        optimizer.param_groups[i*3]["lr"] = stage["lr_luma"] * stage["lr_scales"][i]
                        optimizer.param_groups[i*3+1]["lr"] = stage["lr_chroma"] * stage["lr_scales"][i]
                        optimizer.param_groups[i*3+2]["lr"] = stage["lr_chroma"] * stage["lr_scales"][i]
                    elif self.color_space == "RGB":
                        optimizer.param_groups[i*3]["lr"] = stage["lr_luma"] * stage["lr_scales"][i]
                        optimizer.param_groups[i*3+1]["lr"] = stage["lr_luma"] * stage["lr_scales"][i]
                        optimizer.param_groups[i*3+2]["lr"] = stage["lr_luma"] * stage["lr_scales"][i]

            for c in range(stage["cycles"]):
                if self.stop_signal:
                    return
                self.cycle(c, stage, optimizer, params_pyramid)
                self.progress[n] = c+1


class FourierVisions(VisionsBackend):
    def __init__(self):
        VisionsBackend.__init__(self)

        self.chroma_noise_scale = 1.00000 # Saturation (0 - 2 is safe but you can go as high as you want)
        self.luma_noise_mean = 0.0 # Brightness (-3 to 3 seems safe but around 0 seems to work better)
        self.luma_noise_scale = 1.00000 # Contrast (0-2 is safe but you can go as high as you want)
        self.init_noise_clamp = 0 # Turn this down if you're getting persistent super bright or dark spots.

        self.lr_scale = 100#5e-5
        # hi_freq_decay = 
        self.eq_pow = 1
        self.eq_min = 1e-6

        self.resample_image_prompts = False

        # Size of the smallest pyramid layer
        aspect_ratio = (3,4)#(3, 4)

        # Max dim of the final output image.
        max_dim = 1024
        scale = max_dim // max(aspect_ratio)
        self.dims = (int(aspect_ratio[0] * scale), int(aspect_ratio[1] * scale))

        self.stages = (
            { #First stage does rough detail. It's going to look really coherent but blurry
                "cuts": 2,
                "cycles": 1000,
                "lr_luma": 1,#3e-2,
                "decay_luma": 0.0,
                "lr_chroma": 0.5,#1.5e-2,
                "decay_chroma": 0.0,
                "noise": 0.2,
                "denoise": 10.0,
                "checkin_interval": 100
            }, { # 2nd stage does fine detail. Going to get much clearer
                "cuts": 2,
                "cycles": 1000,
                "lr_luma": 0.5,
                "decay_luma": 0,
                "lr_chroma": 0.25,
                "decay_chroma": 0,
                "noise": 0.2,
                "denoise": 1,
                "checkin_interval": 100,
            }
        )

        self.debug_clip_cuts = False

        self.eq = self.generate_filter(self.dims, self.eq_pow, self.eq_min)

        self.bilinear = torchvision.transforms.functional.InterpolationMode.BILINEAR
        self.bicubic = torchvision.transforms.functional.InterpolationMode.BICUBIC

        torch.autograd.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        self.calc_perceptors()
        
    def generate_filter(self, dims, eq_pow, eq_min):
        eqx = torch.fft.fftfreq(dims[0])
        eqy = torch.fft.fftfreq(dims[1])
        eq = torch.outer(torch.abs(eqx), torch.abs(eqy))
        eq = eq - eq.min()
        eq = 1 - eq / eq.max()
        eq = torch.pow(eq, eq_pow)
        eq = eq * (1-eq_min) + eq_min
        return eq

    def normalize_image(self, image):
        R = (image[:,0:1] - 0.48145466) /  0.26862954
        G = (image[:,1:2] - 0.4578275) / 0.26130258 
        B = (image[:,2:3] - 0.40821073) / 0.27577711
        return torch.cat((R, G, B), dim=1)

    @torch.no_grad()
    def loadImage(self, filename):
        data = open(filename, "rb").read()
        image = torch.ops.image.decode_png(torch.as_tensor(bytearray(data)).cpu().to(torch.uint8), 3).cuda().to(torch.float32) / 255.0
        # image = normalize_image(image)
        return image.unsqueeze(0).cuda()

    def getClipTokens(self, image, cuts, noise, do_checkin, perceptor):
        im = self.normalize_image(image)
        cut_data = torch.zeros(cuts, 3, perceptor["size"], perceptor["size"])
        for c in range(cuts):
            angle = random.uniform(-20.0, 20.0)
            img = torchvision.transforms.functional.rotate(im, angle=angle, expand=True, interpolation=self.bilinear)

            padv = im.size()[2] // 8
            img = torch.nn.functional.pad(img, pad=(padv, padv, padv, padv))

            size = img.size()[2:4]
            mindim = min(*size)

            if mindim <= perceptor["size"]-32:
                width = mindim - 1
            else:
                width = random.randint( perceptor["size"]-32, mindim-1 )

            oy = random.randrange(0, size[0]-width)
            ox = random.randrange(0, size[1]-width)
            img = img[:,:,oy:oy+width,ox:ox+width]

            if self.pixel_art:
                img = torch.nn.functional.interpolate(img, size=(perceptor["size"], perceptor["size"]), mode='nearest')
            else:
                img = torch.nn.functional.interpolate(img, size=(perceptor["size"], perceptor["size"]), mode='bilinear', align_corners=False)
            cut_data[c] = img

        cut_data += noise * torch.randn_like(cut_data, requires_grad=False)

        if self.debug_clip_cuts and do_checkin:
            displayImage(cut_data)

        clip_tokens = perceptor['model'].encode_image(cut_data)
        return clip_tokens


    def loadPerceptor(self, name):
        model, preprocess = clip.load(name, device="cuda")

        tokens = []
        imgs = []
        for text in self.texts:
            tok = model.encode_text(clip.tokenize(text["text"]).cuda())
            tokens.append( tok )

        perceptor = {"model":model, "size": preprocess.transforms[0].size, "tokens": tokens, }
        for img in self.images:
            image = loadImage(img["fpath"])
            if resample_image_prompts:
                imgs.append(image)
            else:
                tokens = getClipTokens(image, img["cuts"], img["noise"], False, perceptor )
                imgs.append(tokens)

        perceptor["images"] = imgs
        return perceptor

    @torch.no_grad()
    def saveImage(self, image, filename):
        # R = image[:,0:1] * 0.26862954 + 0.48145466
        # G = image[:,1:2] * 0.26130258 + 0.4578275
        # B = image[:,2:3] * 0.27577711 + 0.40821073
        # image = torch.cat((R, G, B), dim=1)
        size = image.size()

        image = (image[0].clamp(0, 1) * 255).to(torch.uint8)
        png_data = torch.ops.image.encode_png(image.cpu(), 6)
        open(filename, "wb").write(bytes(png_data))

    # TODO: Use torchvision normalize / unnormalize
    def unnormalize_image(self, image):
        R = image[:,0:1] * 0.26862954 + 0.48145466
        G = image[:,1:2] * 0.26130258 + 0.4578275
        B = image[:,2:3] * 0.27577711 + 0.40821073
        
        return torch.cat((R, G, B), dim=1)

    def paramsToImage(self, params_luma, params_chroma):
        CoCg = torch.fft.irfft2(params_chroma, norm="backward")
        luma = torch.fft.irfft2(params_luma, norm="backward")
        # print(luma.min(), luma.mean(), luma.max())
        # print(CoCg.min(), CoCg.mean(), CoCg.max())
        # luma = torch.sigmoid(luma)
        # CoCg = torch.sigmoid(CoCg)
        # CoCg = CoCg * 2 - 1
        luma = (luma / 2.0 + 0.5).clamp(0,1)
        CoCg = CoCg.clamp(-1,1)
        Co = CoCg[:,0]
        Cg = CoCg[:,1]

        tmp = luma - Cg/2
        G = Cg + tmp
        B = tmp - Co/2
        R = B + Co
        im_torch = torch.cat((R, G, B), dim=1)#.clamp(0,1)
        im_torch = im_torch[:,:,:,:self.dims[1]]
        return im_torch

    def imageToParams(self, image):
        image = image#.clamp(0,1)
        R, G, B = image[:,0:1], image[:,1:2], image[:,2:3]
        luma = R * 0.25 + G * 0.5 + B * 0.25
        Co = R  - B
        tmp = B + Co / 2
        Cg = G - tmp
        luma = tmp + Cg / 2

        nsize = luma.size()[2:4]
        chroma =  torch.cat([Co,Cg], dim=1)
        chroma = chroma / 2.0 + 0.5
        chroma = torch.logit(chroma, eps=1e-8)
        luma = torch.logit(luma, eps=1e-8)
        chroma = torch.fft.rfft2(chroma)
        luma = torch.fft.rfft2(luma)
        return luma, chroma 

    @torch.no_grad()
    def displayImage(self, image):
        size = image.size()

        width = size[0] * size[3] + (size[0]-1) * 4
        image_row = torch.zeros( size=(3, size[2], width), dtype=torch.uint8 )

        nw = 0
        for n in range(size[0]):
            image_row[:,:,nw:nw+size[3]] = (image[n,:].clamp(0, 1) * 255).to(torch.uint8)
            nw += size[3] + 4

        jpeg_data = torch.ops.image.encode_png(image_row.cpu(), 6)
        image = display.Image(bytes(jpeg_data))
        display.display( image )

    def lossClip(self, image, cuts, noise, do_checkin):
        losses = []

        max_loss = 0.0
        for text in self.texts:
            max_loss += abs(text["weight"]) * len(self.perceptors)
        for img in self.images:
            max_loss += abs(img["weight"]) * len(self.perceptors)

        for perceptor in self.perceptors:
            clip_tokens = self.getClipTokens(image, cuts, noise, do_checkin, perceptor)
            for t, tokens in enumerate( perceptor["tokens"] ):
                similarity = torch.cosine_similarity(tokens, clip_tokens)
                weight = self.texts[t]["weight"]
                if weight > 0.0:
                    loss = (1.0 - similarity) * weight
                else:
                    loss = similarity * (-weight)
                losses.append(loss / max_loss)

        for img in self.images:
            for i, prompt_image in enumerate(perceptor["images"]):
                if resample_image_prompts:
                    img_tokens = getClipTokens(prompt_image, images[i]["cuts"], images[i]["noise"], False, perceptor)
                else:
                    img_tokens = prompt_image
                weight = images[i]["weight"] / float(images[i]["cuts"])
                for token in img_tokens:
                    similarity = torch.cosine_similarity(token.unsqueeze(0), clip_tokens)
                    if weight > 0.0:
                        loss = (1.0 - similarity) * weight
                    else:
                        loss = similarity * (-weight)
                    losses.append(loss / max_loss)
        return losses

    def lossTV(self, image, strength):
        Y = (image[:,:,1:,:] - image[:,:,:-1,:]).abs().mean()
        X = (image[:,:,:,1:] - image[:,:,:,:-1]).abs().mean()
        loss = (X + Y) * 0.5 * strength
        return loss

    def cycle(self, c, stage, optimizer, params_luma, params_chroma, eq):
        do_checkin = (c+1) % stage["checkin_interval"] == 0 or c == 0
        with torch.enable_grad():
            image = self.paramsToImage(params_luma, params_chroma)

            optimizer.zero_grad(set_to_none=True)

            losses = self.lossClip( image, stage["cuts"], stage["noise"], do_checkin )

            losses += [self.lossTV( image, stage["denoise"] )]
            
            loss_total = sum(losses).sum()

            loss_total.backward(retain_graph=False)

            optimizer.step()

        if c % self.save_interval == 0:
            TV = losses[-1].sum().item()
            print( "Cycle:", str(stage["n"]) + ":" + str(c), "CLIP Loss:", loss_total.item() - TV, "TV loss:", TV)
            nimg = self.paramsToImage(params_luma, params_chroma)
            # self.displayImage(torch.nn.functional.interpolate(nimg, size=display_size, mode='self.bilinear'))
            self.saveImage(nimg, 'images/' + self.texts[0]["text"].replace(' ', '_') + ".png" )

    def init_optim(self, params_luma, params_chroma, stage):
        params = []
        params.append({"params":params_luma, "lr":stage["lr_luma"] * self.lr_scale, "weight_decay":stage["decay_luma"] * self.lr_scale})
        params.append({"params":params_chroma, "lr":stage["lr_chroma"] * self.lr_scale, "weight_decay":stage["decay_chroma"] * self.lr_scale})
        return torch.optim.AdamW(params)

    def run(self):
        print('Started run with ' + self.texts[0]["text"])
        param_luma = None
        param_chroma = None
        eq = self.generate_filter(self.dims, self.eq_pow, self.eq_min)
        if self.initial_image is not None:
            image = self.loadImage(self.initial_image)
            image = torch.nn.functional.interpolate(image, size=self.dims[-1], mode='self.bicubic', align_corners=False)
            luma, chroma = self.imageToParams(image)
            param_luma = torch.nn.parameter.Parameter( luma.double().cuda(), requires_grad=True)
            param_chroma = torch.nn.parameter.Parameter( chroma.double().cuda(), requires_grad=True)
        else:
            luma = torch.randn(size = (1,1,self.dims[0], self.dims[1])) * self.luma_noise_scale * eq
            chroma = torch.randn(size = (1,2,self.dims[0], self.dims[1])) * self.chroma_noise_scale * eq
            luma = luma.clamp(-self.init_noise_clamp, self.init_noise_clamp)
            chroma = chroma.clamp(-self.init_noise_clamp, self.init_noise_clamp)
            param_luma = torch.nn.parameter.Parameter( luma.cuda(), requires_grad=True)
            param_chroma = torch.nn.parameter.Parameter( chroma.cuda(), requires_grad=True)
        optimizer = self.init_optim(param_luma, param_chroma, self.stages[0])

        print('Setup complete')

        for n, stage in enumerate(self.stages):
            stage["n"] = n
            if n > 0:
                optimizer.param_groups[0]["lr"] = stage["lr_luma"] * self.lr_scale
                optimizer.param_groups[1]["lr"] = stage["lr_chroma"] * self.lr_scale
            for c in range(stage["cycles"]):
                if self.stop_signal:
                    print('Stopping')
                    return
                self.cycle( c, stage, optimizer, param_luma, param_chroma, eq)
                self.progress[n] = c+1


class CLIPCPPN(VisionsBackend):
    class Residual(nn.Module):
        def __init__(self, c):
            super(Residual, self).__init__()
            self.blocks = nn.Linear(c, c)

        def forward(self, x):
            return x + self.blocks(x)

    class CPPN(nn.Module):
        def __init__(self, cfg):
            super(CLIPCPPN.CPPN, self).__init__()
            activation = cfg.get("activation", nn.Tanh)
            self.z_length = cfg.get("z_length", 0)
            self.num_channels = cfg.get("num_channels", 3)
            start_width = cfg.get("hidden_width_start", 128)
            midpoint, mid_width = cfg.get("hidden_width_center", (2, 128))
            end_width = cfg.get("hidden_width_end", 128)
            
            hidden_depth = cfg.get("hidden_depth", 5)
            self.input_dim = self.z_length + 3
            widths = []
            print(start_width, end_width, )
            widths = [start_width + (x * (mid_width - start_width)) // midpoint for x in range(midpoint)] + [mid_width] + \
                    [mid_width + (x * (end_width - mid_width)) // (hidden_depth - midpoint-1)  for x in range(1, hidden_depth - midpoint)]
            print(widths)


            modules = [
                nn.Linear(self.input_dim, start_width),
                # nn.BatchNorm1d(start_width),
                activation()
            ]
            for b in range(hidden_depth-1):
                modules += [
                    nn.Linear(widths[b], widths[b+1]),
                    # nn.BatchNorm1d(widths[b+1]),
                    activation()
                ]
            modules += [
                nn.Linear(end_width, 3),
                nn.Sigmoid()
            ]
            self.module_sequential = nn.Sequential(*modules)

            params = sum([np.prod(p.size()) for p in self.parameters()])
            # self.apply(weights_init)
            
        def forward(self, x):
            # x is of dims [width * height, 3 + z_length]
            # We process every pixel as a batch for efficiency
            # Each input contains [x, y, r, latent vector "z"]
            # x and y should be normalized between 0-1 probably? 
            # latent vector "z" should be the same for every pixel in a single image, if used
            return self.module_sequential(x)

    def __init__(self):
        VisionsBackend.__init__(self)
        #TODO: test different configs
        # Wider in the middle seems to be the best balance of width / depth
        # SELU seems to have the least problem with exploding / vanishing gradients
        cppn_config = {"z_length": 1, # Length of the z vector. Larger = more output variety, but also can confuse the model early
                        "activation": nn.SELU,
                        "hidden_width_start":16,
                        "hidden_width_center":(4, 512), # (layer, width)
                        "hidden_width_end":16,
                        "hidden_depth":9
                        }

        self.scale_min = 0.9
        self.scale_max = 1.1
        self.angle_max = 1
        self.padding = 0.45
        # color_space =  "YCoCg" # "RGB"

        # AdamW is real basic and gets the job done
        # RAdam seems to work *extremely well* but seems to introduce some color instability?, use 0.5x lr
        # Yogi is just really blurry for some reason, use 5x + lr
        # Ranger works great. use 3-4x LR
        self.optimizer_type = "Ranger" # "AdamW", "AccSGD","Ranger","RangerQH","RangerVA","AdaBound","AdaMod","Adafactor","AdamP","AggMo","DiffGrad","Lamb","NovoGrad","PID","QHAdam","QHM","RAdam","SGDP","SGDW","Shampoo","SWATS","Yogi"

        self.checkin_samples = 2

        # Optimizer settings for different training steps
        self.stages = [
                { #First stage does rough detail.
                "cuts": 2,
                "cycles": 1000,
                "lr": 5e-3, #Radam use 1e-2, radam use 1e-4
                "decay": 0.0,
                "noise": 0.1,
                "denoise": 0.0,
                "checkin_interval": 250
            }, { #First stage does rough detail.
                "cuts": 2,
                "cycles": 10000,
                "lr": 5e-3, #Radam use 1e-2, radam use 1e-4
                "lr_decay": 0.9999,
                "decay": 0.0,
                "noise": 0.1,
                "denoise": 0.0,
                "checkin_interval": 250
            }
        ]

        self.display_size = (512, 512)

        self.debug_clip_cuts = True

        torch.autograd.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        self.calc_perceptors()

        self.cppn = CLIPCPPN.CPPN(cppn_config).cuda()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    

    def sample_cppn(self, cppn, n_batch, width, height, scale = None, angle = None, chip_c = None, z = None, debug=False):
        vecs = np.zeros((n_batch, width * height, 3 + cppn.z_length))
        for b in range(n_batch):
            if chip_c is None or b > 0:
                chip_c = np.random.random(2) * (1.0 - self.padding * 2) + self.padding - 0.5
            if scale is None or b > 0:
                scale = np.random.random() * (self.scale_max - self.scale_min) + self.scale_min
            if angle is None or b > 0: 
                angle = np.radians((np.random.random() * 2 - 1) * self.angle_max)
            if z is None or b > 0:
                z = np.random.random(cppn.z_length)
            # Construct pixel coord grid
            xscale = scale * width / max(width, height)
            yscale = scale * height / max(width, height)
            xcoords = np.linspace(-xscale, xscale, width)
            ycoords = np.linspace(-yscale, yscale, height)
            xcoords, ycoords = np.meshgrid(xcoords, ycoords)
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)
            x = cos_a * xcoords - sin_a * ycoords + chip_c[0]
            y = sin_a * xcoords + cos_a * ycoords + chip_c[1]
            # x = x * aspect_ratio[0] / max(aspect_ratio)
            # y = y * aspect_ratio[1] / max(aspect_ratio)
            r = np.square(x) + np.square(y)
            if debug:
                print(x.min().item(), x.max().item())
                print(y.min().item(), y.max().item())
                print(r.min().item(), r.max().item())
                plt.scatter(x.flatten(), y.flatten(), s=1)
                plt.show()
            v = np.stack((y.flatten(), x.flatten(), r.flatten()), 1)
            z = np.tile(z, (width*height, 1))
            vecs[b,:,:] = np.concatenate((v, z), 1)

        vecs = torch.tensor(vecs, dtype=torch.float32).view(n_batch * width * height, -1).cuda()
        # Query the cppn
        image = cppn(vecs).view(n_batch, height, width, cppn.num_channels).moveaxis(3,1)
        
        return image

    class nnsin(nn.Module):
        def __init__(self):
            super(nnsin, self).__init__()
        
        def forward(self, x):
            return torch.sin(x)

    def normalize_image(self, image):
        R = (image[:,0:1] - 0.48145466) /  0.26862954
        G = (image[:,1:2] - 0.4578275) / 0.26130258 
        B = (image[:,2:3] - 0.40821073) / 0.27577711
        return torch.cat((R, G, B), dim=1)

    @torch.no_grad()
    def loadImage(self, filename):
        data = open(filename, "rb").read()
        image = torch.ops.image.decode_png(torch.as_tensor(bytearray(data)).cpu().to(torch.uint8), 3).cuda().to(torch.float32) / 255.0
        # image = normalize_image(image)
        return image.unsqueeze(0).cuda()

    def getClipTokens(self, cppn, cuts, noise, do_checkin, perceptor):
        cut_data = self.sample_cppn(cppn, cuts, perceptor["size"], perceptor["size"])
        cut_data =  self.normalize_image(cut_data)

        cut_data += noise * torch.randn_like(cut_data, requires_grad=False)

        #if self.debug_clip_cuts and do_checkin:
            #displayImage(self.unnormalize_image(cut_data))

        clip_tokens = perceptor['model'].encode_image(cut_data)
        return clip_tokens

    @torch.no_grad()
    def saveImage(self, image, filename):
        # R = image[:,0:1] * 0.26862954 + 0.48145466
        # G = image[:,1:2] * 0.26130258 + 0.4578275
        # B = image[:,2:3] * 0.27577711 + 0.40821073
        # image = torch.cat((R, G, B), dim=1)
        size = image.size()

        image = (image[0].clamp(0, 1) * 255).to(torch.uint8)
        png_data = torch.ops.image.encode_png(image.cpu(), 6)
        open(filename, "wb").write(bytes(png_data))

    # TODO: Use torchvision normalize / unnormalize
    def unnormalize_image(self, image):
        
        R = image[:,0:1] * 0.26862954 + 0.48145466
        G = image[:,1:2] * 0.26130258 + 0.4578275
        B = image[:,2:3] * 0.27577711 + 0.40821073
        
        return torch.cat((R, G, B), dim=1)

    def lossClip(self, cppn, cuts, noise, do_checkin):
        losses = []

        max_loss = 0.0
        for text in self.texts:
            max_loss += abs(text["weight"]) * len( self.perceptors)
        for img in self.images:
            max_loss += abs(img["weight"]) * len( self.perceptors)

        for perceptor in self.perceptors:
            clip_tokens =  self.getClipTokens(cppn, cuts, noise, do_checkin, perceptor)
            for t, tokens in enumerate( perceptor["tokens"] ):
                similarity = torch.cosine_similarity(tokens, clip_tokens)
                weight =  self.texts[t]["weight"]
                if weight > 0.0:
                    loss = (1.0 - similarity) * weight
                else:
                    loss = similarity * (-weight)
                losses.append(loss / max_loss)

            for img in self.images:
                for i, prompt_image in enumerate(perceptor["images"]):
                    if resample_image_prompts:
                        img_tokens = self.getClipTokens(prompt_image, images[i]["cuts"], images[i]["noise"], False, perceptor)
                    else:
                        img_tokens = prompt_image
                    weight = self.images[i]["weight"] / float(self.images[i]["cuts"])
                    for token in img_tokens:
                        similarity = torch.cosine_similarity(token.unsqueeze(0), clip_tokens)
                        if weight > 0.0:
                            loss = (1.0 - similarity) * weight
                        else:
                            loss = similarity * (-weight)
                        losses.append(loss / max_loss)
        return losses

    def lossTV(self, image, strength):
        Y = (image[:,:,1:,:] - image[:,:,:-1,:]).abs().mean()
        X = (image[:,:,:,1:] - image[:,:,:,:-1]).abs().mean()
        loss = (X + Y) * 0.5 * strength
        return loss

    def cycle(self, c, stage, optimizer, cppn):
        do_checkin = (c+1) % stage["checkin_interval"] == 0 or c == 0
        with torch.enable_grad():
            losses = []
            losses +=  self.lossClip( cppn, stage["cuts"], stage["noise"], do_checkin )
            # losses += [lossTV( image, stage["denoise"] )]

        loss_total = sum(losses).sum()
        optimizer.zero_grad(set_to_none=True)
        loss_total.backward(retain_graph=False)
        # if c <= warmup_its:
        #   optimizer.param_groups[0]["lr"] = stage["lr_luma"] * c / warmup_its
        #   optimizer.param_groups[1]["lr"] = stage["lr_chroma"] * c / warmup_its
        optimizer.step()
        if "lr_decay" in stage:
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] *= stage["lr_decay"]

        if do_checkin:
            # TV = losses[-1].sum().item()
            print( "Cycle:", str(stage["n"]) + ":" + str(c), "CLIP Loss:", loss_total.item(), "LR:", optimizer.param_groups[0]["lr"])
            # for name, param in cppn.named_parameters():
            #   print(name, param.min(), param.mean(), param.max())
            nimg =  self.sample_cppn(cppn, self.checkin_samples, *self.display_size, 1, 0, [0.0, 0.0])
            print(nimg.shape, nimg.min().item(), nimg.mean().item(), nimg.max().item())
            self.saveImage(nimg, 'images/' + self.texts[0]['text'].replace(' ', '_') + '.png')

    def function(self):
        params = [{"params": self.cppn.module_sequential.parameters(), "lr":self.stages[0]["lr"], "weight_decay":self.stages[0]["decay"]}]
        optimizer = getattr(optim, self.optimizer_type, None)(params)

        for n, stage in enumerate(self.stages):
            stage["n"] = n
            if n > 0: 
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]["lr"] = stage["lr"]
                    optimizer.param_groups[i]["decay"] = stage["decay"]

            for c in range(stage["cycles"]):
                self.cycle( c, stage, optimizer, self.cppn)
                self.progress[n] = c+1
