from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, AdamW
import torch
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable
import numpy as np
from source.ernie import *
from source.models import RobertaForClaimDetection
import os

# Global Variables:
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def load_ClaimBuster_data(train=True):
    # groundtruth.csv and crowdsourced.csv
    if train:
        df = pd.read_csv("./ClaimBuster_Datasets/datasets/crowdsourced.csv")
    else:
        df = pd.read_csv("./ClaimBuster_Datasets/datasets/groundtruth.csv")
    #df = df1.append(df2, ignore_index=True) # for both...
    texts = df['Text']
    labels = df['Verdict'].copy()
    labels[labels != 1] = 0
    ids = df['Sentence_id']
    return texts.tolist(), labels.tolist(), ids.tolist()

def load_clef20_data(data_file):
    df = pd.read_csv(data_file, sep='\t')
    texts = df['tweet_text'].tolist()
    labels = df['check_worthiness'].tolist()
    topic_ids = df['topic_id'].tolist()
    tweet_ids = df['tweet_id'].tolist()
    return texts, labels, topic_ids, tweet_ids

# to load data from CLEF
def load_clef19_data(datadir, as_docs=False):
    files = glob.glob(datadir)
    texts = list()
    labels = list()
    ids = list()
    for f in files:
        df = pd.read_csv(f, sep='\t', names=['no.', 'speaker', 'statement', 'label'])
        if as_docs:
            texts.append(df['statement'])
            labels.append(df['label']) # for 1 class model
            #lbls = [((0, 1)) if lbl else ((1, 0)) for lbl in df['label']] # for 2 class model
            #labels.append(lbls) # for 2 class model
            ids.append(df['no.'])
        else:
            texts += df['statement'].tolist()
            labels += df['label'].tolist() # for 1 class model
            #lbls = [((0, 1)) if lbl else ((1, 0)) for lbl in df['label']] # for 2 class model
            #labels += lbls # for 2 class model
            ids += df['no.'].tolist()
    return texts, labels, ids

# class describing a torch dataset object
class torchCLEFDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def weighted_sampler(train_y):
    N = float(len(train_y))
    ones = float(np.sum(train_y))
    ones_weight = N/ones
    zeros_weight = N/(N - ones)
    weights = [ones_weight if x else zeros_weight for x in train_y]
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return zeros_weight, ones_weight, sampler

def load_model(mpath, unfreeze):
    model = RobertaForClaimDetection(n_classes=2, unfreeze=unfreeze)
    checkpoint = torch.load(mpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded was trained for this many epochs: ", checkpoint['epoch'])
    for param in model.roberta.base_model.parameters():
        param.requires_grad = unfreeze
    model.eval()
    return model

def format_output(output_file, model_dir, test_data_dir="./clef2019-factchecking-task1/data/test_annotated/"):
    fnames = glob.glob(test_data_dir + "*.tsv")
    df = pd.read_csv(output_file, sep='\t', header=None)
    start_indices = [i for i, d in enumerate(df[0]) if d == 1]
    assert len(fnames) == len(start_indices)

    for i in range(len(start_indices)):
        start = start_indices[i]
        if i + 1 == len(start_indices):
            end = len(df[0])
        else:
            end = start_indices[i+1]

        current_f = "./saved_models/" + model_dir + "/preds/" + fnames[i].split("/")[-1][:-4] + "_out.tsv"
        df[start:end].to_csv(current_f, sep='\t', header=False, index=False, float_format='%.5f')
        og = pd.read_csv(fnames[i], sep='\t', header=None)
        ne = pd.read_csv(current_f, sep='\t', header=None)
        assert og.shape[0] == ne.shape[0]
        assert ne[0][0] == 1
        print("Saved output file %d / %d: %s" % (i+1, len(start_indices), current_f))

def train(model, args):
    if args.clef_year == 19:
        train_x, train_y, _ = load_clef19_data('clef2019-factchecking-task1/data/training/*.tsv')
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=.2)
    elif args.clef_year == 20:
        train_x, train_y, _, _ = load_clef20_data('clef2020-factchecking-task1/data/v2/training_v2.tsv')
        val_x, val_y, _, _ = load_clef20_data('clef2020-factchecking-task1/data/v2/dev_v2.tsv')
    else:
        train_x, train_y, _ = load_ClaimBuster_data(train=True)
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=.2)

    # tokenizer to take sentences and encode them for the model:
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    
    # encode training and validation data:
    train_encodings = tokenizer(train_x, truncation=True, padding=True)
    val_encodings = tokenizer(val_x, truncation=True, padding=True)
    
    # make a torch dataset object out of them
    train_dataset = torchCLEFDataset(train_encodings, train_y)
    val_dataset = torchCLEFDataset(val_encodings, val_y)

    BATCH_SIZE = args.batch_size
    dom_cw, un_cw, sam = weighted_sampler(train_y)

    if args.weights:
        weights = [dom_cw, un_cw]
    else:
        weights = [1.0, 1.0]
        
    class_weights = torch.FloatTensor(weights).to(DEVICE)

    print("Zeros class weight: ", weights[0])
    print("Ones class weight: ", weights[1])
    
    if args.sampler:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sam)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    print("LENGTH OF TRAIN_LOADER: ", len(train_loader))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("LENGTH OF VAL_LOADER: ", len(val_loader))
    data_loaders = dict()
    data_loaders['train'] = train_loader
    data_loaders['val'] = val_loader
    
    best_val_acc = float('-inf')
    best_epoch = -1
    best_state_dict = None 
    best_optimizer = None
    best_loss = float('inf')

    optimizer = AdamW(model.parameters(), lr=1.5e-5)
    if args.load:
        checkpoint = torch.load(args.load)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(args.epochs):
        total_t0 = time.time()
        running_loss = 0.0
        running_corrects = 0
        running_total = 0
        print('\n======== Epoch {:} / {:} ========'.format(epoch + 1, args.epochs))
        for phase in ['train', 'val']:
            print("\nPHASE: ", phase)
            print()
            t0 = time.time()
            n = len(data_loaders[phase])
            if phase == 'train':
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                model.train()
            else:
                model.eval()

            for i, batch in enumerate(data_loaders[phase]):
                if phase == 'train':
                    optimizer.zero_grad()
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                #_, logits = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                else:
                    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                    running_corrects += torch.sum(preds == labels.data)
                    running_total += len(preds)

                if phase == 'train':
                    if i % 75 == 74:    # print every 75 mini-batches
                        elapsed = format_time(time.time() - t0)
                        print('     Batch {:>4,}  of  {:>4,}:     Avg. Running Loss: {:}     Time elapsed: {:}'.format(i+1, n, running_loss / i+1, elapsed))
            
            print()
            if phase == 'val':
                val_acc = torch.true_divide(running_corrects, running_total)
                print('     Validation accuracy: {0:.3f}'.format(val_acc))
                if val_acc > best_val_acc:
                    best_epoch = epoch + 1
                    best_state_dict = model.state_dict()
                    best_optimizer = optimizer.state_dict()
                    best_loss = loss
                    best_val_acc = val_acc
                    print("     Saving the best updated model to: ./saved_models/" + args.model_dir + "/model.pth ...")
                    torch.save({
                        'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        }, ("./saved_models/" + args.model_dir + "/model.pth"))

                    print('     Updated and saved best model...')

            print('     Epoch avg. loss: {0:.3f}'.format(running_loss / len(data_loaders['train'])))
            print('     Epoch time elapsed: {:}'.format(format_time(time.time() - total_t0)))
    print()
    print("=" * 15 + " TRAINING COMPLETED " + "=" * 15)
    print("\nThe model with the highest validation accuracy trained for: %d epoch(s).\nThe validation accuracy was: %.3f\nThe loss was: %.3f\n" % (best_epoch, best_val_acc, best_loss))

    stats = {}
    stats['Epoch with best val accuracy: '] = best_epoch
    stats['Best val accuracy: '] = best_val_acc
    stats['Best loss was: '] = best_loss
    update_model_log(model=model, args=args, stats=stats, train=True)
    model.eval()
    return model

def test(model, args):
    print("\nLoading test data...")
    if args.clef_year == 19:
        test_x, test_y, test_ids = load_clef19_data('clef2019-factchecking-task1/data/test_annotated/*.tsv')
    elif args.clef_year == 20:
        test_x, test_y, topic_ids, tweet_ids = load_clef20_data("clef2020-factchecking-task1/test-input/test-gold.tsv")
    else:
        test_x, test_y, test_ids = load_ClaimBuster_data(train=False)


    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    test_encodings = tokenizer(test_x, truncation=True, padding=True)
    # make a torch dataset object out of them
    test_dataset = torchCLEFDataset(test_encodings, test_y)
    BATCH_SIZE = args.batch_size
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Evaluating...")
    # Evaluate:
    running_corrects = 0
    running_ones = 0
    n = len(test_loader)
    running_preds = []
    running_softs = []
    printProgressBar(0, n, prefix='Progress:', suffix='batch 0 / %d' % (n), length=25)
    for i, batch in enumerate(test_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        with torch.no_grad():
            #_, logits = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        softmaxes = F.softmax(logits, dim=1)
        preds = torch.argmax(softmaxes, dim=1)
        running_softs += softmaxes.tolist()
        running_preds += preds.tolist()
        running_corrects += torch.sum(preds == labels.data)
        running_ones += torch.sum(preds)
        printProgressBar(i+1, n, prefix='Progress:', suffix='batch %d / %d' % (i+1, n), length=25)
    acc = torch.true_divide(running_corrects, len(test_y))
    confusion_mat = confusion_matrix(test_y, running_preds)
    print("Accuracy: ", acc)
    print("Num. ones guessed: ", running_ones)
    print("Confusion Matrix (true label = rows, predicted label = cols): ")
    print(confusion_mat)
    print()
    print("=" * 15 + " TESTING COMPLETED " + "=" * 15)
    
    stats = {}
    stats['Accuracy on test set: '] = acc
    stats['Confusion Matrix (true label = rows, predicted label = cols):\n'] = confusion_mat
    stats['Num. ones guessed: '] = running_ones

    print()
    update_model_log(model=model, args=args, stats=stats, train=False)

    print("Generating files to pass to clef scorer and obtain mAP etc...")
    ranking_scores = []
    for pair in running_softs:
        ranking_scores.append(pair[1] - pair[0])
    
    if args.clef_year == 19:
        ranking = pd.DataFrame.from_records(list(zip(test_ids, ranking_scores))).round(5)
        ranking_fname = './saved_models/' + args.model_dir + '/preds/ranking.tsv'
        ranking.to_csv(ranking_fname, sep='\t', float_format='%.5f', header=False, index=False)
        print("Saved all scores to: ", ranking_fname)
        print("Now splitting " + ranking_fname + " into its proper parts...")
        format_output(output_file=ranking_fname, model_dir=args.model_dir)
    elif args.clef_year == 20:
        run_ids = ["Model_1"] * len(test_y)
        ranking = pd.DataFrame.from_records(list(zip(topic_ids, tweet_ids, ranking_scores, run_ids))).round(5)
        ranking_fname = './saved_models/' + args.model_dir + '/preds/ranking.tsv'
        ranking.to_csv(ranking_fname, sep='\t', float_format='%.5f', header=False, index=False)
        print("Saved all scores to: ", ranking_fname)
    else:
        pass
    
    print()
    print("=" * 15 + " PROGRAM END. NEXT STEP: " + "=" * 15)
    print("\nNow pass those files above^^ into clef score.sh and run it to obtain mAP score.\n")

if __name__ == "__main__":
    # PARSE ARGUMENTS:
    args = get_args()
    
    # make necessary folders...
    if not os.path.isdir('./saved_models/' + args.model_dir):
        os.makedirs('./saved_models/' + args.model_dir)

    if not os.path.isdir('./saved_models/' + args.model_dir + '/preds/'):
        os.makedirs('./saved_models/' + args.model_dir + '/preds/')

    # LOAD MODEL:
    print()
    print("=" * 15 + " INITIALIZING MODEL " + "=" * 15)
    if args.load:
        print("\nLoading saved model from: " + args.load + " ...\n")
        model = load_model(args.load, unfreeze=args.unfreeze)
        print("\nSuccessfully loaded " + args.load + "!\n")
    else:
        print("\nNo model was provided to load, therefore starting from scratch and loading RoBERTa-base...\n")
        #model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
        model = RobertaForClaimDetection(n_classes=2, unfreeze=args.unfreeze)
        print(model)
        print()
        print()
        print('#' * 80)
        print()
        print()
        print("\nSuccessfully loaded RoBERTa-base!\n")
    
    #model = nn.DataParallel(model, device_ids=[0,1,2])
    model.to(DEVICE)
    print("Device: ", DEVICE)

    # TRAINING:
    if args.train:
        print()
        print("=" * 15 + " TRAINING MODEL " + "=" * 15)
        print("\nThe trained model will be saved in: ./saved_models/" + args.model_dir + "/model.pth upon completion of training.")
        model = train(model, args=args)
    
    # TESTING:
    if args.test:
        print()
        print("=" * 15 + " TESTING MODEL " + "=" * 15)
        print("\nThe model's predictions on the test data will be saved in: ./saved_models/" + args.model_dir + "/preds/ upon completion of testing.")
        test(model, args=args)
