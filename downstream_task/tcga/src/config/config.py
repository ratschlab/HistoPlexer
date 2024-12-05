class Config:
    """Contains all configuration parameters."""
    def __init__(self, config_dict):

        # general config
        self.seed: int = config_dict.get('seed', 1)
        self.max_epochs: int = config_dict.get('max_epochs', 100)
        self.save_path: str = config_dict.get('save_path', None)
        self.task: str = config_dict.get('task', None) # "survival" or "classification" or "multi-task"
        self.pretrained_path: str = config_dict.get('pretrained_path', None)
        self.num_folds: int = config_dict.get('num_folds', 5)
        self.fold: int = config_dict.get('fold', None)
        self.sub: int = config_dict.get('sub', 10)
        
        # optimizer config
        self.lr: float = config_dict.get('lr', 1e-4)
        self.reg: float = config_dict.get('reg', 1e-3)
        self.scheduler_decay_rate: float = config_dict.get('scheduler_decay_rate', 0.5)
        self.scheduler_patience: int = config_dict.get('scheduler_patience', 5)
        self.gc: int = config_dict.get('gc', 32)
        self.loss: str = config_dict.get('loss', "nll_surv")
        self.alpha_surv: float = config_dict.get('alpha_surv', 0.) # How much to weigh uncensored patients
        
        # dataset and dataloader config
        self.data_path: str = config_dict.get('data_path', None)
        self.csv_path: str = config_dict.get('csv_path', None)
        self.split_path: str = config_dict.get('split_path', None)
        self.is_weighted_sampler: bool = config_dict.get('is_weighted_sampler', True)
        self.num_workers: int = config_dict.get('num_workers', 8)
        self.label_col: str = config_dict.get('label_col', "immune_subtype")
        self.label_dict: dict = config_dict.get('label_dict', None)
        self.imc_feature_type: str = config_dict.get('imc_feature_type', "resnet")
        self.bins: list = config_dict.get('bins', [0, 1.70349076e+01, 3.79137577e+01, 7.05215606e+01, 3.69675566e+02])
        
        # model
        self.model: str = config_dict.get('model', "MCAT")
        self.n_cls: int = config_dict.get('n_cls', 4)
        self.drop_out: float = config_dict.get('drop_out', None)
        self.in_feat_dim_he: int = config_dict.get('in_feat_dim_he', 512)
        self.in_feat_dim_imc: int = config_dict.get('in_feat_dim_imc', 512)
        self.fusion: str = config_dict.get('fusion', "concat")
        
        # model saver config
        self.save_metric: str = config_dict.get('save_metric', "loss")
        self.early_stop: bool = config_dict.get('early_stop', False)