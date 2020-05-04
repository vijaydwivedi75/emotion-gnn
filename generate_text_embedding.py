#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch, pickle, warnings, argparse
import transformers as ppb
from data.iemocap import IEMOCAPFeatures
warnings.filterwarnings('ignore')


def generate_text_features(df, tokenizer, model):
    df.reset_index(drop=True, inplace=True)

    if tokenizer:
        tokenized = df['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

        np.array(padded).shape

        attention_mask = np.where(padded != 0, 1, 0)
        attention_mask.shape

        input_ids = torch.tensor(padded)  
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:,0,:].numpy()
    
    else:
        features = model.encode(df['text'].tolist())
        features = np.array(features)

    df_F = pd.DataFrame(features)

    df_combined = pd.concat([df, df_F], axis=1)
    
    feature_dict = {}
    i = 0
    for session in df['session'].unique().tolist():
        feature_dict[session] = df_combined[df_combined['session']==session].iloc[:,3:].to_numpy()
        
    return feature_dict


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="BERT / DISTILBERT / BERT_FT / ROBERTA")
    args = parser.parse_args()

    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = "BERT_FT"
    
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

    print(train.shape, test.shape)


    if MODEL_NAME == 'ROBERTA':
        model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
        tokenizer = None
    elif MODEL_NAME == 'BERT_FT':
        model_state_dict = torch.load("./models/fine_tuned_BERT_base_model_state.bin")
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights, state_dict=model_state_dict)    
    else:
        if MODEL_NAME == 'BERT':
            model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        elif MODEL_NAME == 'BERT_SA':
            model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'lvwerra/bert-imdb')           
        else:
            model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)



    #Generate text embeddings
    #Generate the train/test features separately due to memory constraint
    train_feature_dict = generate_text_features(train, tokenizer, model)
    test_feature_dict = generate_text_features(test, tokenizer, model)

    #Load original pickle file
    videoIDs, videoSpeakers, videoLabels, videoText,videoAudio, videoVisual, videoSentence, trainVid,testVid = pickle.load(open('data/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

    #Replace text embeddings
    videoText_new = {}
    for k, v in videoText.items():
        if k in train_feature_dict:
            videoText_new[k] = train_feature_dict[k]
        else:
            videoText_new[k] = test_feature_dict[k]

    final_output = [videoIDs, videoSpeakers, videoLabels, videoText_new,videoAudio, videoVisual, videoSentence, trainVid,testVid]

    with open('./data/IEMOCAP_features_'+MODEL_NAME+'.pkl', 'wb') as f:
        pickle.dump(final_output, f)

if __name__ == '__main__':
    main()
