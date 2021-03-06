LOSS_GAMMA = 10  # from authors, not optimized
LOSS_NEED_INTERMEDIATE_LAYERS = False
CLASSES_CNT = 31
UNK_VALUE = -100  # torch default
IS_UNSUPERVISED = True

MODEL_BACKBONE = "alexnet" # alexnet resnet50 vanilla_dann
BACKBONE_PRETRAINED = True
GRADIENT_REVERSAL_LAYER_ALPHA = 1.0
FREZE_BACKBONE_FEATURES = True

IMAGE_SIZE = 224
BATCH_SIZE = 32
DATASET = "office-31"
SOURCE_DOMAIN = "amazon"
TARGET_DOMAIN = "dslr"
NUM_WORKERS = 4
N_EPOCHS = 20
STEPS_PER_EPOCH = 20
VAL_FREQ = 1
SAVE_MODEL_FREQ = 19
