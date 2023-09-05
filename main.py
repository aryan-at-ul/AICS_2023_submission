import os
import sys
import argparse
from utils import graph_preperation
from utils import str_to_bool
from utils import make_dirs ,dataloader, get_gnn_model, save_model, save_plots, SaveBestModel,download_dataloader
from config import *
import torch
from torch.utils.data import Dataset,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from train import train_epoch, valid_epoch
import csv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt


torch.cuda.empty_cache()

current_file_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
def write_test_results(gnn_model, superpixel_number, cnn_model_name, test_loss, test_acc, spcficity, sensitivity, roc):
    # print("reach here")
    with open('best_test_results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([gnn_model, superpixel_number, cnn_model_name, test_loss, test_acc, spcficity, sensitivity, roc])




def run_epoch(model, train_loader,val_loader, optimizer, criterion, epochs = 10, train = True, cnn_model_name = 'densenet121', gnn_model = 'GCN', superpixel_number = 10):
    epoch_loss = 0
    epoch_acc = 0
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    save_best_model = SaveBestModel()
    if train:
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model,device,train_loader,criterion,optimizer)
            val_loss, val_acc,cfm, spcficity, sensitivity, true, pred = valid_epoch(model,device,val_loader,criterion)
  
            print("Epoch:{}/{} Training Loss:{:.3f}  Val Loss:{:.3f}  Training Acc {:.4f} %  Val Acc {:.4f} %".format(epoch + 1,
                                                                                                                    epochs,
                                                                                                                    train_loss,
                                                                                                                    val_loss,
                                                                                                                    train_acc,
                                                                                                                    val_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(val_acc)

            save_best_model(val_acc, epoch, model, optimizer, criterion, cnn_model_name, gnn_model, superpixel_number)



    else:
        test_loss, test_acc = valid_epoch(model,device,val_loader,criterion)
        
        

    save_plots(history['train_acc'], history['test_acc'], history['train_loss'], history['test_loss'],cnn_model_name, gnn_model, superpixel_number)
    



def main(cnn_model_name = 'densenet', gnn_model = 'GCN', superpixel_number = 10, learning_rate = 0.001, batch_size = 64, epochs = 10, train = True, saved = True):#, pretrained = False
    print("Model name: ", cnn_model_name)

    criterion = torch.nn.CrossEntropyLoss()
    if saved:
        train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset = dataloader(FOLDER_PATH , batch_size, saved, superpixel_number, cnn_model_name)
        print("Using saved state")
        if train:
            print("Training")
            train_dataset = [data.to(device) for data in train_dataset]
            val_dataset = [data.to(device) for data in val_dataset]
            test_dataset = [data.to(device) for data in test_dataset]
            # dataset = ConcatDataset([train_dataset, val_dataset])
            model = get_gnn_model(gnn_model, feature_size[cnn_model_name])
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.001)
            run_epoch(model, train_loader, test_loader, optimizer, criterion, epochs, train, cnn_model_name, gnn_model, superpixel_number)
            print("training complete")
            # explainer = Explainer(
            #     model=model,
            #     explanation_type='node',
            #     algorithm=GNNExplainer(epochs=200),
            #     node_mask_type='attributes')
            print("number of paramteres for this model",sum(p.numel() for p in model.parameters()))


        else:
            print("Testing")
            print(f'outputs/{gnn_model}_{superpixel_number}_{cnn_model_name}_best_model.pt')
            model = get_gnn_model(gnn_model, feature_size[cnn_model_name])
            model.load_state_dict(torch.load(f'outputs/{gnn_model}_{superpixel_number}_{cnn_model_name}_best_model.pt', map_location=torch.device(device)))
            print("number of paramteres for this model",sum(p.numel() for p in model.parameters()))
            test_loss, test_acc,cfm, spcficity, sensitivity, true, pred = valid_epoch(model,device,test_loader,criterion)
            plt.imshow(cfm,cmap = plt.cm.Wistia,
                    interpolation='nearest')
            for i in range(cfm.shape[0]):
                for j in range(cfm.shape[1]):
                    plt.text(j, i, cfm[i, j],
                            ha="center", va="center", color="black",fontsize=14)
            plt.title(f'{gnn_model} model {superpixel_number} superpixels {cnn_model_name} features')
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.xticks([0, 1], ['NORMAL', 'PNEUMONIA'], fontsize = 12)
            plt.yticks([0, 1], ['NORMAL', 'PNEUMONIA'], fontsize = 12)
            # plt.colorbar()
            # plt.show()
            print("label  1 sensitivity",sensitivity)
            plt.savefig(f'outputs/{gnn_model}_{superpixel_number}_{cnn_model_name}_confusion_matrix.png')


            fpr, tpr, thresholds = metrics.roc_curve(true, pred)
            roc_auc = metrics.auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC {gnn_model}')
            plt.legend(loc="lower right")
            plt.savefig(f'outputs/{gnn_model}_{superpixel_number}_{cnn_model_name}_{thresholds}_ROC.png')

            print("Test Loss: {:.4f}".format(test_loss))
            print("Test Acc: {:.4f} %".format(test_acc))
            print("testing complete")
            # specificity_score, acc,specificity_score2 = 0,0,0
            write_test_results(gnn_model, superpixel_number, cnn_model_name, test_loss, test_acc,spcficity, sensitivity,roc_auc)
    
    else:
        print("Not using saved state")
        graph_preperation(superpixel_number,cnn_model_name)
        os.system(f"python saving_dataloader.py --model_name {cnn_model_name} --super_pixels {superpixel_number}")
        saved = True
        main(cnn_model_name, gnn_model, superpixel_number, learning_rate, batch_size, epochs, train, saved)
        return 0

    return 0 


if __name__ == "__main__":

    make_dirs('saved_data_loader')    
    # if not os.path.exists(f'{current_file_path}/chest_xray'):
    #     print("please make sure that chest_xray folder is in the same directory as main.py")
    #     print("expected chest_xray folder with train and test folders inside")
    #     print("run dowload_data.py to download the data and extract it to the chest_xray folder")
    #     sys.exit()
    parser = argparse.ArgumentParser(description='Getting arguments')
    parser.add_argument('--cnn_model_name', type=str, default='densenet121', help='Name of the model')
    parser.add_argument('--gnn_model', type=str, default='GCN', help='Name of the GNN model')
    parser.add_argument('--superpixel_number', type=int, default=10, help='Number of superpixels')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train', type=str_to_bool, nargs='?', const=True, default=True, help='Train or test')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--use_saved_state', type=str_to_bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    cnn_model_name = args.cnn_model_name
    gnn_model = args.gnn_model
    superpixel_number = args.superpixel_number
    learning_rate = args.learning_rate
    train = args.train
    batch_size = args.batch_size
    epochs = args.epochs
    saved = args.use_saved_state
    print(f"starting main.py for saved state = {saved}")

    if not os.path.exists(f'{current_file_path}/chest_xray') and not saved:
        print("please make sure that chest_xray folder is in the same directory as main.py")
        print("expected chest_xray folder with train and test folders inside")
        print("run dowload_data.py to download the data and extract it to the chest_xray folder")
        sys.exit()
    
    if not os.path.exists(f'{current_file_path}/saved_data_loader/train_dataloader_{superpixel_number}_{cnn_model_name}.pkl'):
        download_dataloader(superpixel_number,cnn_model_name)

    ret = main(cnn_model_name,gnn_model, superpixel_number, learning_rate, batch_size, epochs, train, saved)#, pretrained
    if ret == 0:
        print("Done")
    else:
        print("Error")

