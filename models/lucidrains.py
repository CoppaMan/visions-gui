import logging

from worker.thread import ThreadedWorker

from deep_daze import Imagine as DDImagine
from deep_daze.deep_daze import exists

import torch
import torchvision.transforms as T
from tqdm import trange, tqdm

class DeepDazeWrap(DDImagine):
    '''
    Using a generator instead of for loop for cleaner stopping
    '''
    def __init__(self, **kwargs):
        DDImagine.__init__(self, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        tqdm.write('RUNNING WRAPPER ROUTINE')

        if exists(self.start_image):
            tqdm.write('Preparing with initial image...')
            optim = DiffGrad(self.model.model.parameters(), lr = self.start_image_lr)
            pbar = trange(self.start_image_train_iters, desc='iteration')
            try:
                for _ in pbar:
                    loss = self.model.model(self.start_image)
                    loss.backward()
                    pbar.set_description(f'loss: {loss.item():.2f}')

                    optim.step()
                    optim.zero_grad()
            except KeyboardInterrupt:
                print('interrupted by keyboard, gracefully exiting')
                return exit()

            del self.start_image
            del optim

        tqdm.write(f'Imagining "{self.textpath}" from the depths of my weights...')

        with torch.no_grad():
            self.model(self.clip_encoding, dry_run=True) # do one warmup step due to potential issue with CLIP and CUDA

        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        try:
            for epoch in trange(self.epochs, desc='epochs'):
                pbar = trange(self.iterations, desc='iteration')
                for i in pbar:
                    try:
                        _, loss = self.train_step(epoch, i)
                    except RuntimeError as e:
                        self.logger.critical(e)
                        return
                    pbar.set_description(f'loss: {loss.item():.2f}')
                    # Yield after a complete iteration
                    yield [epoch, i]

                # Update clip_encoding per epoch if we are creating a story
                if self.create_story:
                    self.clip_encoding = self.update_story_encoding(epoch, i)
        except KeyboardInterrupt:
            print('interrupted by keyboard, gracefully exiting')
            return

        self.save_image(epoch, i) # one final save at end

        if (self.save_gif or self.save_video) and self.save_progress:
            self.generate_gif()
    
    @torch.no_grad()
    def save_image(self, epoch, iteration, img=None):
        '''
        Write into the same result image without writing new images 
        '''
        sequence_number = self.get_img_sequence_number(epoch, iteration)

        if img is None:
            img = self.model(self.clip_encoding, return_loss=False).cpu().float().clamp(0., 1.)
        self.filename = self.image_output_path(sequence_number=sequence_number)
        
        pil_img = T.ToPILImage()(img.squeeze())
        # Create no new images
        # pil_img.save(self.filename, quality=95, subsampling=0) 
        pil_img.save(f"{'images/' + self.textpath}.png", quality=95, subsampling=0)

        # Remove image update notification
        # tqdm.write(f'image updated at "./{str(self.filename)}"')


class DeepDaze(ThreadedWorker):
    def __init__(self):
        ThreadedWorker.__init__(self)

    def set_texts(self, prompt):
        self.text = prompt

    def set_seed(self, seed):
        pass

    def set_weighted_prompts(self, weighted_prompts):
        pass
        
    def function(self):
        self.progress = [ 0 ] * self.epochs
        model = DeepDazeWrap(
            text=self.text,
            epochs=self.epochs,
            iterations=self.iterations,
            num_layers=self.layers,
            open_folder=False,
            save_every=self.save_interval,
            image_width=self.image_size
        )
        
        try:
            for epoch, iteration in model.forward():
                if self.stop_signal:
                    print('RTR')
                    return
                self.progress[epoch] = iteration+1
        except:
            self.stop()
            return
            
