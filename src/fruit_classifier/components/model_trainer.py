from fastai.vision.all import *
from fastai.vision.widgets import *

from fruit_classifier.config.configuration import ConfigurationManager


class ModelTrainer:

    def __init__(self):
        
        config_manager = ConfigurationManager()
        self.config = config_manager.get_training_config()

    def random_seed(seed_value, use_cuda):
        np.random.seed(seed_value) # cpu vars
        torch.manual_seed(seed_value) # cpu  vars
        random.seed(seed_value) # Python
        if use_cuda: 
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value) # gpu vars
            torch.backends.cudnn.deterministic = True  #needed
            torch.backends.cudnn.benchmark = False

    def create_dls(self):

        path = self.config.training_data

        fruit_db = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2),#, seed=42),
            get_y=parent_label,
            item_tfms=Resize(464),
            batch_tfms=aug_transforms(size=224, min_scale=0.5)
        )

        dls = fruit_db.dataloaders(path, num_workers=0)

        return dls
    
    def train_model(self):

        self.random_seed(seed_value=13, use_cuda=True)

        dls = self.create_dls()

        learn = vision_learner(
            dls, 
            resnet18, 
            loss_func=LabelSmoothingCrossEntropy(), 
            metrics=accuracy, 
            cbs=[
                MixUp, 
                SaveModelCallback(
                    monitor='accuracy',
                    fname=self.config.params_model_name
                ),
                EarlyStoppingCallback(
                    monitor='accuracy', 
                    patience=self.config.params_patience
                )
            ]
        )

        learn.fine_tune(self.config.params_epochs, freeze_epochs=self.config.params_n_freeze_epochs)

        model = vision_learner(dls, resnet18)
        model.load(self.config.params_model_name)
        model.export(f'{self.config.trained_model_path}/{self.config.params_model_name}.pkl')
