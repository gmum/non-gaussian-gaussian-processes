class Config:
    def __init__(self, args):
        self.kernel_type = args.kernel_type  # spectral' #'bncossim' #linear, rbf, spectral (regression only), matern, poli1, poli2, cossim, bncossim
        self.data_dir = {}
        self.data_dir['miniImagenet'] = './filelists/miniImagenet/'
        self.data_dir['omniglot'] = './filelists/omniglot/'
        self.data_dir['emnist'] = './filelists/emnist/'
        self.data_dir['eeg'] = './filelists/EEG/A001SB1_1.csv'
        self.data_dir['nasdaq'] = './filelists/Nasdaq_100/nasdaq100_padding.csv'
        self.data_dir['qmul'] = 'filelists/QMUL/images/' # the prefix for loading faces in QMUL dataset. See data/qmul_loader.py
        self.data_dir['QMUL'] = 'filelists/QMUL/images/'
        self.data_dir['objects'] = './filelists/objects_pose' # the prefix for loading faces in QMUL dataset. See data/qmul_loader.py
        self.save_dir = args.save_dir

        # config used for nn kernel, if nn kernel is specified as the kernel type. The input_dim is also used to
        # determine the context dimension in context-augmented NGGP approach.
        self.nn_config = {}
        if args.dataset == "sines":
            self.nn_config["input_dim"] = args.output_dim
        elif args.dataset == "nasdaq" or args.dataset == "eeg":
            self.nn_config["input_dim"] = args.output_dim
        elif args.dataset == "QMUL":
            self.nn_config["input_dim"] = 2916
        elif args.dataset == "CUB":
            self.nn_config["input_dim"] = 1600
        elif args.dataset == "objects":
            self.nn_config["input_dim"] = 64
        else:
            raise ValueError("input dim for nn kernel not known for value {}".format(args.dataset))
        
        if args.dataset == "objects":
            self.nn_config["hidden_dim"] = 64
            self.nn_config["output_dim"] = 64
            self.nn_config["num_layers"] = 4     
        else:
            self.nn_config["hidden_dim"] = 16
            self.nn_config["output_dim"] = 16
            self.nn_config["num_layers"] = 1


