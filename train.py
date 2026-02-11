import os 
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models import VGG19_Weights
import constants as c
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Tuple
from sklearn.metrics import(
    precision_recall_fscore_support, matthews_corrcoef, accuracy_score,
    confusion_matrix, classification_report
)

class TrainingConfiguration:
    DATA_DIR: str = "Datasets/main_dataset_no_combination"
    BATCH_SIZE: int = c.DefaultConstant.DEFAULT_BATCH_SIZE.value
    NUM_EPOCHS: int = c.DefaultConstant.DEFAULT_NUM_EPOCHS.value
    LEARNING_RATE: float = c.DefaultConstant.DEFAULT_TRAINING_LR.value
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_FILEDIR: str = "saved_model"
    SAVE_FILENAME: str = "batik_vgg19_features"

    def setBatch(this, newBatch: int):
        this.BATCH_SIZE = newBatch
    def setNumEpoch(this, newNum: int):
        this.NUM_EPOCHS = newNum
    def setLR(this, newLR: float):
        this.LEARNING_RATE = newLR
    def setDevice(this, newDevice: str):
        this.DEVICE = newDevice
    def setDataDir(this, newDir: str):
        this.DATA_DIR = newDir

def get_transform_dataset()-> transforms.Compose:
    
    transform = transforms.Compose([
        transforms.Resize((c.DefaultConstant.DEFAULT_SIZE_IMAGE_1.value, c.DefaultConstant.DEFAULT_SIZE_IMAGE_1.value)),
        transforms.ToTensor(),
        transforms.Normalize(mean=c.IMAGENET_MEAN_255, 
                             std=c.IMAGENET_STD_255)
    ])

    return transform

def get_dataloaders(config: TrainingConfiguration) -> Tuple[Tensor, Tensor, List[str]]:
    train_dir: str = os.path.join(config.DATA_DIR, "train")
    test_dir: str = os.path.join(config.DATA_DIR,"test")

    if os.path.exists(test_dir):
        print("Test dataset is found")
        test_dataset = datasets.ImageFolder(test_dir, transform=get_transform_dataset() )
    
    if os.path.exists(train_dir):
        print("Train dataset is found")
        train_dataset = datasets.ImageFolder(train_dir, transform=get_transform_dataset())

    train_loader = DataLoader(train_dataset,
                              batch_size=c.DefaultConstant.DEFAULT_BATCH_SIZE.value,
                              shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=c.DefaultConstant.DEFAULT_BATCH_SIZE.value,
                             shuffle=False,
                             )
    
    class_names: List[str] = train_dataset.classes

    print(f"Loaded {len(train_dataset)} training images. Total class {len(class_names)}\n Including {len(test_dataset)} test images")
    for indx, name in enumerate(class_names):
        print(f"{indx}: {name}")
    
    return train_loader, test_loader, class_names

def create_model_for_training(num_classes: List[str], freeze_layers=False) :
    model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

    if freeze_layers:
        for param in model.features[:20].parameters():
            param.requires_grad = False
        print("First 20 layers are frezzed")

    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, c.DefaultConstant.LINEAR_DEFAULT.value),
        nn.ReLU(True),
        nn.Dropout(c.DefaultConstant.DROPOUT_DEFAULT.value),
        nn.Linear(c.DefaultConstant.LINEAR_DEFAULT.value, c.DefaultConstant.LINEAR_DEFAULT.value),
        nn.ReLU(True),
        nn.Dropout(c.DefaultConstant.DROPOUT_DEFAULT.value),
        nn.Linear(c.DefaultConstant.LINEAR_DEFAULT.value, len(num_classes))
    )
    return model
def train_model(model, train_loader:DataLoader,
                test_loader: DataLoader, config: TrainingConfiguration,
                class_names: List[str]):
    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    #optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                               factor=c.DefaultConstant.SCHEDULER_FACTOR_DEFAULT.value,
                                                patience=c.DefaultConstant.SCHEDULER_PATIENCE_DEFAULT.value)
    
    history: Dict[str, List[float]] = {c.DefaultConstant.M_TRAIN_LOSS.value: [],
                      c.DefaultConstant.M_TEST_LOSS.value: [],
                      c.DefaultConstant.M_TRAIN_ACCURACY.value: [],
                      c.DefaultConstant.M_TEST_ACCURACY.value: [],
                      c.DefaultConstant.M_TRAIN_RECALL.value: [],
                      c.DefaultConstant.M_TEST_RECALL.value: [],
                      c.DefaultConstant.M_TRAIN_F1.value: [],
                      c.DefaultConstant.M_TEST_F1.value: [],
                      c.DefaultConstant.M_TRAIN_MCC.value: [],
                      c.DefaultConstant.M_TEST_MCC.value: [],
                      c.DefaultConstant.M_TRAIN_PREC.value: [],
                      c.DefaultConstant.M_TEST_PREC.value: [],
                      c.DefaultConstant.M_TRAIN_SUPPORT.value: [],
                      c.DefaultConstant.M_TEST_SUPPORT.value: []
                    }


    print(f"Training on {config.DEVICE}")
    print(f"Classes ({len(class_names)})")
    print(f"Epochs: {config.NUM_EPOCHS} || Batch Size: {config.BATCH_SIZE}")
    print("-" *100)

    e_best_f1: float = 0.0
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        batch_loss: float  = 0.0

        all_train_preds: List = []
        all_test_preds: List = []

        all_train_labels: List = []
        all_test_labels: List = []
        xsa = 0
        for b_inputs, b_labels in train_loader:
            b_inputs, b_labels = b_inputs.to(config.DEVICE), b_labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs: Tensor = model(b_inputs)
            loss: Tensor = criterion(outputs, b_labels)
            loss.backward()
            optimizer.step()

            _, tr_preds = torch.max(outputs, dim=1) 
            #print(outputs)
            #print(outputs.size())
            all_train_labels.extend(b_labels.cpu().numpy())
            all_train_preds.extend(tr_preds.cpu().numpy())

            batch_loss += loss.item() * b_inputs.size(0)
            #print(all_train_preds)
            #print("="*100)
            #print(all_train_labels)
            #print(batch_loss)
            xsa+=1
            if xsa == 2:
                break
        
        e_loss_train = batch_loss / len(train_loader.dataset)
        print(all_train_labels)
        print(all_train_preds)
        
        e_acc_train = accuracy_score(all_train_labels, all_train_preds)
        e_prec_train, e_recall_train, e_f1_train, e_support_train =  precision_recall_fscore_support(all_train_labels, all_train_preds,
                                                                                                      average="micro", zero_division="warn")
        history[c.DefaultConstant.M_TRAIN_LOSS.value].append(e_loss_train)
        history[c.DefaultConstant.M_TRAIN_F1.value].append(e_f1_train)
        history[c.DefaultConstant.M_TRAIN_RECALL.value].append(e_recall_train)
        history[c.DefaultConstant.M_TRAIN_ACCURACY.value].append(e_acc_train)
        history[c.DefaultConstant.M_TRAIN_PREC.value].append(e_prec_train)
        history[c.DefaultConstant.M_TRAIN_SUPPORT.value].append(e_support_train)
        
        # ======== test =+++=+++++=+++=
        tsa= 0
        model.eval()
        test_batch_loss: float = 0.0
        for b_inputs, b_labels in test_loader:
            b_inputs, b_labels = b_inputs.to(config.DEVICE), b_labels.to(config.DEVICE)
            outputs = model(b_inputs)
            loss: Tensor = criterion(outputs, b_labels)
            
            test_batch_loss += loss.item() * b_inputs.size(0)
            _, test_preds = torch.max(outputs, dim=1)
            all_test_preds.extend(test_preds.cpu().numpy())
            all_test_labels.extend(b_labels.cpu().numpy())
            tsa+=1
            if tsa == 2:
                break
        e_loss_test = test_batch_loss / len(test_loader.dataset)

        e_acc_test = accuracy_score(all_test_labels, all_test_preds)
        e_prec_test, e_recall_test, e_f1_test, e_support_test =  precision_recall_fscore_support(all_test_labels, all_test_preds,
                                                                                            average="micro", zero_division="warn")
        
        history[c.DefaultConstant.M_TEST_LOSS.value].append(e_loss_test)
        history[c.DefaultConstant.M_TEST_F1.value].append(e_f1_test)
        history[c.DefaultConstant.M_TEST_RECALL.value].append(e_recall_test)
        history[c.DefaultConstant.M_TEST_ACCURACY.value].append(e_acc_test)
        history[c.DefaultConstant.M_TEST_PREC.value].append(e_prec_test)
        history[c.DefaultConstant.M_TEST_SUPPORT.value].append(e_support_test)

        scheduler.step(e_loss_test)

        if e_f1_test > e_best_f1:
            print(f"Saving new loss, previously {e_best_f1} -> now {e_f1_test}")           
            e_best_f1 = e_f1_test
            torch.save(model.state_dict(), os.path.join(config.SAVE_FILEDIR, config.SAVE_FILENAME +"_"+str(epoch) + ".pth"))
        
        print("\n\n")
        print(f"  Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"  Train Loss: {e_loss_train:.4f} | Test Loss: {e_loss_test:.4f}")
        print(f"  Accuracy Train: {e_acc_train:.2%} | Accuracy Test : {e_acc_test:.3f} ")
        print(f"  Precision Train: {e_prec_train:.2%} | Precision Test : {e_prec_test:.3f} ")
        print(f"  Recall Train: {e_acc_train:.2%} | Recall Test : {e_recall_test:.3f} ")
        print(f"  F1 Train: {e_acc_train:.3f} | F1 Test: {e_f1_test:.3f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

    print("\n" + "="*80)
    print(f"Training complete! Best validation F1: {e_best_f1:.3f}")

    return model, history
