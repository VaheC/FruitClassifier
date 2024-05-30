from fruit_classifier.constants import *
from fruit_classifier.utils.utils import read_yaml, create_directories
from fruit_classifier.entity.entity import DataConfig, TrainingConfig


class ConfigurationManager:

    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config['artifacts_root']])


    
    def get_data_config(self) -> DataConfig:
        config = self.config['data_root']

        create_directories([config])

        data_config = DataConfig(
            root_dir=config['data_root']
        )

        return data_config
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config['model_root']
        params = self.params
        training_data = self.config['data_root']
        create_directories([
            Path(training)
        ])

        training_config = TrainingConfig(
            trained_model_path=Path(training),
            training_data=Path(training_data),
            params_epochs=params['N_EPOCHS'],
            params_n_classes=params['N_CLASSES'],
            params_n_freeze_epochs=params['N_FREEZE_EPOCHS'],
            params_patience=params['PATIENCE'],
            params_model_name=params['MODEL_NAME']
        )

        return training_config