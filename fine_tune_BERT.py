from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import defaultdict
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from data.iemocap import IEMOCAPFeatures

RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'


class IEMOCAPDataset(Dataset):

    def __init__(self, utterances, targets, tokenizer, max_len):
        self.utterances = utterances
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
  
    def __len__(self):
        return len(self.utterances)
  
    def __getitem__(self, item):
        utterance = str(self.utterances[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            utterance,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'utterance_text': utterance,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = IEMOCAPDataset(
            utterances=df.text.to_numpy(),
            targets=df.label.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


class EmotionClassifier(nn.Module):

    def __init__(self, n_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
                
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
          input_ids = input_ids,
          attention_mask = attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    f1s = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            targets = targets.cpu().detach().numpy()
            preds = preds.cpu().detach().numpy()
            
            f1s.append(round(f1_score(targets, preds, average='weighted', labels=np.unique(preds)), 2))
            
    return correct_predictions.double() / n_examples, np.mean(f1s), np.mean(losses)


def main():

    #Read data
    df = pd.read_csv('./data/dialoguegcn_utterances.csv')
    df.rename(columns={'Unnamed: 0':'key'}, inplace=True)
    df['session'] = df['key'].map(lambda x: '_'.join(x.split('_')[:-1]))

    trainset = IEMOCAPFeatures()
    testset = IEMOCAPFeatures(train=False)

    #Extract trainVid and TestVid
    test_ids = [t[-1] for t in testset]
    train_ids = [t[-1] for t in trainset]

    train = df[df['session'].isin(train_ids)]
    test = df[df['session'].isin(test_ids)]

    #print(train.shape, test.shape)

    videoIDs, videoSpeakers, videoLabels, videoText,videoAudio, videoVisual, videoSentence, trainVid,testVid = pickle.load(open('data/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

    id_labels = []
    for k, v in videoIDs.items():
        id_labels.extend(list(zip(v, videoLabels[k])))

    df = pd.DataFrame(id_labels, columns =['vid', 'label']) 


    train = train.merge(df, left_on='key', right_on='vid')
    del train['vid']
    test = test.merge(df, left_on='key', right_on='vid')
    del test['vid']

    df_train, df_test = train, test
    print(df_train.shape, df_test.shape)


    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    BATCH_SIZE = 16
    MAX_LEN = 160
    EPOCHS = 5

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


    model = EmotionClassifier(6)
    model = model.to(device)


    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)


    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,    
            loss_fn, 
            optimizer, 
            device, 
            scheduler, 
            len(df_train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        test_acc, test_f1, test_loss = eval_model(
            model,
            test_data_loader,
            loss_fn, 
            device, 
            len(df_test)
        )

        print(f'Test   loss {test_loss} accuracy {test_acc} f1 {test_f1}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc)
        history['test_acc'].append(test_f1)    
        history['test_loss'].append(test_loss)
        
        if test_acc > best_accuracy:
            torch.save(model.state_dict(), './models/fine_tuned_BERT_base_model_state.bin')
            best_accuracy = test_acc


if __name__ == '__main__':
    main()

