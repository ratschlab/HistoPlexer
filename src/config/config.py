class Config:
    """Class for train/test configuration."""

    def __init__(self, config_dict):
        """Initialize configuration.

        Args:
            config_dict: dictionary of configuration parameters
        """
        # paths
        self.base_save_path: str = config_dict.get("base_save_path", "/raid/sonali/project_mvs/nmi_results") # TODO where all results will be saved 
        self.src_folder: str = config_dict.get("src_folder", "/raid/sonali/project_mvs/data/tupro/binary_he_rois") # TODO change as needed
        self.tgt_folder: str = config_dict.get("tgt_folder", "/raid/sonali/project_mvs/data/tupro/binary_imc_processed_11x") # TODO change as needed
        self.split: str = config_dict.get("split", "/raid/sonali/project_mvs/meta/tupro/split3_train-test.csv") # TODO change as needed
        self.vgg_path: str = config_dict.get("vgg_path", "/raid/sonali/project_mvs/results/models/vgg19_model.pth") # TODO change as needed
        self.resume_path: str = config_dict.get("resume_path", None)

        # data 
        self.num_workers: int = config_dict.get("num_workers", 8) 
        self.batch_size: int = config_dict.get("batch_size", 16) 
        self.use_high_res: bool = config_dict.get("use_high_res", True)
        self.p_flip_jitter_hed_affine: list = config_dict.get("p_flip_jitter_hed_affine", [0.5, 0.0, 0.5, 0.5])
        self.patch_size: int = config_dict.get("patch_size", 256) # of target imc
        self.markers: list = config_dict.get("markers", ["CD16", "CD20", "CD3", "CD31", "CD8a", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100", "SOX10"]) 
        self.channels: list = config_dict.get("channels", None) # provide list of channels to use, None for all
        self.cohort: str = config_dict.get("cohort", "tupro") # TODO change as needed -- used for naming experiment
        self.val: bool = config_dict.get("val", False) # TODO change to True is using validation set
        self.use_fm_features: bool = config_dict.get("use_fm_features", False) # TODO change as needed
        self.fm_features_path: str = config_dict.get("fm_features_path", None) # TODO change as needed
        self.fm_feature_size: int = config_dict.get("fm_feature_size", 0) # TODO change as needed

        # model 
        self.input_nc: int = config_dict.get("input_nc", 3) 
        self.output_nc: int = config_dict.get("output_nc", 11) # TODO change based on markers chosen 
        self.use_multiscale: bool = config_dict.get("use_multiscale", True) # TODO True for ours, False for baselines 
        self.ngf: int = config_dict.get("ngf", 32)
        self.depth: int = config_dict.get("depth", 6) 
        self.encoder_padding: int = config_dict.get("encoder_padding", 1)
        self.decoder_padding: int = config_dict.get("decoder_padding", 1)
        self.encoder_init_type = 'xavier'
        self.discriminator_init_type = 'xavier'
        self.method: str = config_dict.get("method", "ours") # TODO used for naming experiment, change as needed pix2pix, pyp2p, ours
        
        # optimizer 
        self.lr_G: float = config_dict.get("lr_G", 0.004) 
        self.lr_D: float = config_dict.get("lr_D", 2.5e-4) 
        self.lr_F: float = config_dict.get("lr_F", 0.004) 
        self.beta_0: float = config_dict.get("beta_0", 0.5) 
        self.beta_1: float = config_dict.get("beta_1", 0.999) 
        
        # loss
        self.use_gp: bool = config_dict.get("use_gp", True) # TODO True for ours and pyp2p, False for p2p
        self.w_L1: float = config_dict.get("w_L1", 1.) # always 1 
        self.w_GP: bool = config_dict.get("w_GP", 0.) # TODO 5 for ours, 1 for pyp2p and 0 for p2p
        self.w_ASP: float = config_dict.get("w_ASP", 1.) # TODO 1 for our, 0 for baselines
        self.w_R1: float = config_dict.get("w_R1", 1.) # always 1 
        self.r1_gamma: float = config_dict.get("r1_gamma", 2e-4)
        
        # misc
        self.blur_gt: bool = config_dict.get("blur_gt", False) 
        self.p_dis_add_noise: float = config_dict.get("p_dis_add_noise", None)
        self.use_feat_enc: bool = config_dict.get("use_feat_enc", True)
        
        # visualization
        self.vis_size: int = config_dict.get("vis_size", 256)
        self.vis_vmin: list = config_dict.get("vis_vmin", None)
        self.vis_vmax: list = config_dict.get("vis_vmax", None)
        
        # schedule 
        self.total_steps: int = config_dict.get("total_steps", 500000) # 500000 
        self.ema_warmup: int = config_dict.get("ema_warmup", 5000)
        self.update_rule: str = config_dict.get("update_rule", "prob")
        self.update_interval: int = config_dict.get("update_interval", 1)
        self.r1_interval: int = config_dict.get("r1_interval", 16) 
        self.log_interval: int = config_dict.get("log_interval", 100)
        self.log_img_interval: int = config_dict.get("log_img_interval", 1000)
        self.save_interval: int = config_dict.get("save_interval", 5000) # 5000
        
        # seed
        self.seed: int = config_dict.get("seed", 0) # TODO change as needed
        self.device: str = config_dict.get("device", 'cuda:0') # TODO change as needed
