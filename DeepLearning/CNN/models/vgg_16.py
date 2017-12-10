from DeepLearning.CNN.recognizer import Recognizer

class vgg_16_model(Recognizer):
    def _configure(self):
        """
                configure thr recognizer
                """
        self.w_img = 70
        self.h_img - 70
        self.img_channels = 1
        self.if_invert_img = True
        self.num_classes = 200
        self.batch_size = 64
        self.name_recognizer = 'Lenet'
        self.local_debug = False
        self.normalize_img_mean0 = False
        self.percent_validaction = 0.1