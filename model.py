from fastai.vision.all import *
from fastai.vision.widgets import *
import pickle

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

N_EPOCHS = 100
N_CLASSES = 10
N_FREEZE_EPOCHS = 5
PATIENCE = 15
MODEL_NAME = 'resnet18_fruits'

random_seed(seed_value=13, use_cuda=True)

path = Path('/kaggle/input/fruit-classification10-class/MY_data/train')

audio_db = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2),#, seed=42),
    get_y=parent_label,
    item_tfms=Resize(464),
    batch_tfms=aug_transforms(size=224, min_scale=0.5)
)

dls = audio_db.dataloaders(path, num_workers=0)

learn = vision_learner(
    dls, 
    resnet18, 
    loss_func=LabelSmoothingCrossEntropy(), 
    metrics=accuracy, 
    cbs=[
        MixUp, 
        SaveModelCallback(
            monitor='accuracy',
            fname=MODEL_NAME
        ),
        EarlyStoppingCallback(
            monitor='accuracy', 
            patience=PATIENCE
        )
    ]
)

learn.fine_tune(N_EPOCHS, freeze_epochs=N_FREEZE_EPOCHS)

try:
    learn.export('final_model.pkl')
except:
    pass

model = vision_learner(dls, resnet18)

model.load(f'/kaggle/working/models/{MODEL_NAME}')

model.export(f'{MODEL_NAME}.pkl')

# Better model found at epoch 34 with accuracy value: 0.969565212726593.
# https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class
# Apple
# Orange
# Avocado
# Kiwi
# Mango
# Pinenapple
# Strawberries
# Banana
# Cherry
# Watermelon