#!/usr/bin/env python
# coding: utf-8

# !pip install srsly jsonlines openpyxl en_core_web_lg-3.1.0-py3-none-any.whl
# # pprint #--index-url https://artifactory.alight.com/artifactory/api/pypi/hws-pypi-local/simple --extra-index https://artifactory.alight.com/artifactory/api/pypi/python-pypi-remote/simple
#!pip install ner/payroll-cloud/en_core_web_lg-3.1.0-py3-none-any.whl
#!pip install -U pip setuptools wheel
#!pip install en_core_web_lg-3.1.0-py3-none-any.whl srsly spacy numpy pandas jsonlines

import srsly
import numpy as np
import pandas as pd
import pandera as pa
import re
import jsonlines
import datetime
import copy
import os
import sys
import json

#sys.path.append( './' )
import pandera_schemas as ps

from sklearn.model_selection import train_test_split

import typer
from pathlib import Path

import en_core_web_lg
nlp = en_core_web_lg.load()


def read_excel(filename):
    '''to read excel sheets'''
    xls = pd.ExcelFile(filename)
    #print("Available Sheets: ",xls.sheet_names)

    df_dict = {}
    for sheet in xls.sheet_names:
        df_dict[sheet] = pd.read_excel(filename, sheet_name=sheet)
    return df_dict


def write_excel(df_dict,filename, engine='xlsxwriter'):
    '''to write excel sheets'''
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename, engine=engine)

    for col in df_dict.keys():
        df_dict[col].dropna(axis=1, how='all').to_excel(writer, sheet_name=col)

    writer.save()

def text_to_span(row):
    if pd.isna(row.spans):
        return pd.Series(["", ""])
    _text = str(row.text)
    _beg = row.spans['start']
    _end = row.spans['end']
    return pd.Series([_text[_beg:_end].lower(), row.spans['label']])

def correct_span(row):
    if pd.isna(row.spans):
        return {}
    
    row.spans['label'] = row.label
    return row.spans

def spans_to_list(grp):
    list_spans = []
    for span in grp:
        if not isinstance(span, dict):
            break
        list_spans.append(span)
    return list_spans

def find_new_multilabel_spans(spans_w_more_1_label_df, df_span_adj_assignment):
    span_label_crosstab = pd.crosstab(spans_w_more_1_label_df.span, spans_w_more_1_label_df.label)

    # find new multilabel spans 
    span_label_crosstab = span_label_crosstab.loc[list(set(span_label_crosstab.index)-
                                                       set(df_span_adj_assignment['lower_span']))]

    for span in span_label_crosstab.index:
        dict_labels_count = {k:v for k,v in span_label_crosstab.loc[span].to_dict().items() if v>0}

        df_span_adj_assignment = df_span_adj_assignment.append({'lower_span':span, 'label':max(dict_labels_count, key=dict_labels_count.get), 
                                                                'IS_VERIFIED':'NO', 'remark':str(dict_labels_count)},ignore_index=True)

    return df_span_adj_assignment.set_index('lower_span')
    
def resolve_issue3(df, data_correction_df_dict, DEBUG):
    df.index.name = 'index'
        
    # create dict of spans to be corrected
    df_span_adj_assignment = data_correction_df_dict['corrected_labels'] #shallow copy 
    dict_span_adj_assignment = dict(zip(df_span_adj_assignment['lower_span'],df_span_adj_assignment['label']))
    
    # create new df with text & spans as columns
    spans_df = df[['text', 'spans']].copy()
    spans_df = spans_df.explode('spans')
    spans_df = spans_df.reset_index()
    spans_df.index.name = 'explode_index'
    
    # extract spans and label
    spans_df[['span', 'label']] = spans_df.apply(text_to_span, axis=1)
    
    # extract spans with more than 1 label
    spans_w_more_1_label_df = spans_df[['span', 'label']].groupby('span').filter(lambda g: (g.nunique() > 1).any())
    
    # correct label if its available in dict_span_adj_assignment & find changed rows
    spans_w_more_1_label_df['corrected_label'] = spans_w_more_1_label_df.span.replace(dict_span_adj_assignment)# replace to avoid Nan instaed of map
    spans_w_more_1_label_df['IS_CHANGED'] = np.where(spans_w_more_1_label_df.label!=spans_w_more_1_label_df.corrected_label,
                                                     True, False) #due
    changed = spans_w_more_1_label_df[spans_w_more_1_label_df.IS_CHANGED] #it will give only True
        
    # marking the rows in spans_df where changes are required
    spans_df['IS_CHANGED'] = False
    spans_df.loc[changed.index, 'label'] = changed['corrected_label']
    spans_df.loc[changed.index, 'IS_CHANGED'] = changed['IS_CHANGED']
    spans_df['more_1_label'] = spans_df.index.isin(spans_w_more_1_label_df.index)
    
    # correcting the spans
    spans_df.loc[spans_df[spans_df.IS_CHANGED].index, 'spans'] = spans_df.loc[
                                            spans_df[spans_df.IS_CHANGED].index].apply(correct_span, axis=1)
    
    if DEBUG:
        print('-'*80)
        print(changed.to_string())
        spans_df[spans_df.IS_CHANGED==True].to_csv('issue3_debug_info.csv')
        print('-'*80)
    
    # apply changes in main df
    updated_spans = spans_df.groupby('index')['spans'].apply(spans_to_list) #list
    df.loc[updated_spans.keys(), 'spans'] = updated_spans
    
    # append_new_multilabel_spans to df_span_adj_assignment
    data_correction_df_dict['corrected_labels'] = find_new_multilabel_spans(spans_w_more_1_label_df, df_span_adj_assignment)

    return changed.shape[0]


total_rows_impected_task4 = 0
#---------------------------------

def find_ws(text, token_end):
    true = 'true'
    false = 'false'
    if token_end<len(text): 
        if text[token_end]==' ':
            return true      
    return false

def create_token_dict(tok, start, end, tok_id, text):
    token_dict = {}
    token_dict['text'] = tok
    token_dict['start'] = start
    token_dict['end'] = end
    token_dict['id'] = tok_id
    token_dict['ws'] = find_ws(text, token_dict['end'])
    return token_dict
    
def find_tokens(text):
    list_toks = []
    tok = ''
    for loc, char in enumerate(text):
        if len(tok)==0:
            start = loc
            
        if char.isalnum():
            tok +=char
        else:
            if len(tok)>0:
                list_toks.append(create_token_dict(tok, start, loc, len(list_toks), text))
                tok=''
                
            # append special chars directly
            if char!=' ':
                list_toks.append(create_token_dict(char, loc, loc+1, len(list_toks), text))

    if len(tok)>0:
        list_toks.append(create_token_dict(tok, start, loc+1, len(list_toks), text))
        
    return list_toks


def print_info_util(text, list_spans, list_tokens):
    print("TOKENS: ")
    [print(f'{i}:{t}') for i,t in enumerate(list_tokens)] 
    print('-'*40)
    print(f"Spans: ")
    for i, span in enumerate(list_spans):
        print(f'{i}:{span}')
        print("Span Text: ", text[span['start']:span['end']])
        print("Text from Tokens: ", end='')
        for token in list_tokens:
            if (token['id']>=span['token_start']) & (token['id']<=span['token_end']):
                print(token['text'], end=' ')
        print('\n')
    #print()
    
def print_info(text, list_tokens, list_spans, updated_list_tokens, updated_list_span):
    print("Text: ", text)
    print('-'*40)
    print("Before: ")
    print('-'*20)
    print_info_util(text, list_spans, list_tokens)
    
    print()
    print('-'*80)
    print("After: ")
    print('-'*20)
    print_info_util(text, updated_list_span, updated_list_tokens)
    print()
    print('='*80)

def correct_noun_phrase(text, start, end, dict_corrected_noun_phrase):
    new_start = start
    new_end = end 
    
    span_text = text[start:end].lower()    
    if span_text in dict_corrected_noun_phrase.keys():
        try:
            corrected_span_text = dict_corrected_noun_phrase[span_text]
            
            # return start & end = -1 if corrected_span_text is Nan, empty or single char
            if ((not isinstance(corrected_span_text,str)) or (len(corrected_span_text)<2)):
                return -1, -1
            
            # find corrected_span_text in text
            find_index = text.find(corrected_span_text, start, end)
            #print("find_index ", find_index)
            if find_index>=0:
                new_start = find_index
                new_end = find_index+len(corrected_span_text)
        except:
            pass

    return new_start, new_end

def update_span_task4(row, dict_corrected_noun_phrase, DEBUG):
    global total_rows_impected_task4
    list_spans = []
    list_tokens = copy.deepcopy(row.tokens)
    
    # find if changes are required in any spans list for this row
    flag_update_span = False
    for i, span in enumerate(row.spans):
        # fetch span text
        span_text = row.text[span['start']:span['end']]       
        
        if span_text in dict_corrected_noun_phrase.keys():
            if dict_corrected_noun_phrase[span_text] == span_text:
                continue
            flag_update_span = True
            list_spans = []
            break
        else:
            list_spans.append(span)
       
    # update all spans if flag_update_span is True
    if flag_update_span:
        total_rows_impected_task4+=1
        for i, span in enumerate(row.spans):
            # Check if span is empty
            if len(span)==0:
                continue
            # deepcopy span, just to makesure it doesnt impact orignal one
            span = span.copy() 

            # 1. Correct non-alphanumeric chars at start and end of span text
            #-------------------------------------------------------------
            new_start, new_end = correct_noun_phrase(row.text, span['start'],span['end'], dict_corrected_noun_phrase)
            
            # dont append span if new_start, new_end = -1
            if ((new_start == -1) or (new_end==-1)):
                continue                
                
            # check if span phrase contains leading n-an char, 
            #if row.text[span['start']:new_start]:
            span['start'] = new_start

            # check if span phrase contains Trailing n-an chars
            #if row.text[new_end:span['end']]:
            span['end'] = new_end            
            #---------------------------------------------------

            # 2. Update tokens_start & tokens_end in span(if new tokens are available in tokens list)
            #-------------------------------------------------------------            
            # Update list of tokens based on, new tokenizer(spacy tokenizer is not working fine)         

            # replace tokens list
            list_tokens = find_tokens(row.text)

            list_token_words = [token_dict['text'] for token_dict in list_tokens]

            # find span tokens(just words)
            span_token_words = [token_dict['text'] for token_dict in 
                           find_tokens(row.text[new_start:new_end])]

            for token_id in range(len(list_token_words)):
                # find first token
                if list_token_words[token_id] == span_token_words[0]:
                    # search remaining tokens
                    match = 0
                    for span_token_id in range(len(span_token_words)):
                        if span_token_words[span_token_id] == list_token_words[token_id+span_token_id]:
                            match +=1
                    # update token_start & token_end in span if complete span text found
                    if match == len(span_token_words):
                        span['token_start'] = token_id
                        span['token_end'] = token_id+len(span_token_words)-1

            list_spans.append(span)
        #---------------------------------------------------
        # 3. just for debugging(remove later)    
        if DEBUG:
            print_info(row.text, row.tokens, row.spans, list_tokens, list_spans)
        
    return list_spans, list_tokens


def find_new_incorrect_span_phrases(df, corrected_spans_df):  
    # create new df with text & spans as columns
    spans_df = df[['text', 'spans']].copy()
    spans_df = spans_df.explode('spans')

    # extract spans(in lower) and label
    spans_df[['span', 'label']] = spans_df.apply(text_to_span, axis=1)

    # find span_word_count
    spans_df['span_word_count'] = spans_df['span'].apply(lambda span: len(span.split()))

    # find list_new_spans
    list_new_spans = list(set(spans_df[spans_df.span_word_count > 1].span.unique())-set(corrected_spans_df.lower_span.values))
    df_new_spans = pd.DataFrame({'lower_span':list_new_spans, 'corrected_span': list_new_spans, 'IS_VERIFIED': 'NO'})
    
    # append new spans with corrected_spans_df with IS_VARIFIED='NO'
    corrected_spans_df = pd.concat([corrected_spans_df,df_new_spans], ignore_index=True)    
    
    return corrected_spans_df.set_index('lower_span')


def resolve_issue4(df, data_correction_df_dict, DEBUG):     
    # create dict of spans to be corrected
    corrected_spans_df = data_correction_df_dict['corrected_spans']
    dict_corrected_noun_phrase=dict(zip(corrected_spans_df['lower_span'],corrected_spans_df['corrected_span']))
    
    # Update Spans
    start_time=datetime.datetime.now()
    df[['spans','tokens']] = df.apply(update_span_task4, args=(dict_corrected_noun_phrase, DEBUG, ), 
                                      axis=1, result_type='expand')
    
    # append_new_incorrect_span_phrases to corrected_spans_df
    data_correction_df_dict['corrected_spans'] = find_new_incorrect_span_phrases(df, corrected_spans_df)
    
    end_time=datetime.datetime.now()
    print("Time taken to resolve issue4: (in sec) ",end_time-start_time)
    
    return total_rows_impected_task4


total_rows_impected_task2 = 0
#--------------------------------------

def func_remove_non_alpha_num(text, start, end):
    new_start = start
    new_end = end 
    
    for index in range(start, end):
        if text[index].isalnum():
            new_start = index
            break

    for index in range(end-1, start-1, -1):
        if text[index].isalnum():
            new_end = index
            break
            
    return new_start, new_end+1

def update_span(row, DEBUG):
    global total_rows_impected_task2
    list_spans = []
    list_tokens = copy.deepcopy(row.tokens)
    
    # find if changes are required in any spans list for this row
    flag_update_span = False
    for i, span in enumerate(row.spans):
        # fetch span text
        #print(span)
        span_text = row.text[span['start']:span['end']]
        if (not span_text[0].isalnum()) or (not span_text[-1].isalnum()):
            flag_update_span = True
            list_spans = []
            break
        else:
            list_spans.append(span)
       
    # update all spans if flag_update_span is True
    if flag_update_span:
        total_rows_impected_task2+=1
        for i, span in enumerate(row.spans):
            # Check if span is empty
            if len(span)==0:
                continue
            # deepcopy span, just to makesure it doesnt impact orignal one
            span = span.copy() 

            # 1. Correct non-alphanumeric chars at start and end of span text
            #-------------------------------------------------------------
            new_start, new_end = func_remove_non_alpha_num(row.text, span['start'],span['end'])

            # check if span phrase contains leading n-an char, 
            if row.text[span['start']:new_start]:
                span['start'] = new_start

            # check if span phrase contains Trailing n-an chars
            if row.text[new_end:span['end']]:
                span['end'] = new_end            
            #---------------------------------------------------

            # 2. Update tokens_start & tokens_end in span(if new tokens are available in tokens list)
            #-------------------------------------------------------------            
            # Update list of tokens based on, new tokenizer(spacy tokenizer is not working fine)         

            # replace tokens list
            list_tokens = find_tokens(row.text)

            list_token_words = [token_dict['text'] for token_dict in list_tokens]

            # find span tokens(just words)
            span_token_words = [token_dict['text'] for token_dict in 
                           find_tokens(row.text[new_start:new_end])]

            for token_id in range(len(list_token_words)):
                # find first token
                if list_token_words[token_id] == span_token_words[0]:
                    # search remaining tokens
                    match = 0
                    for span_token_id in range(len(span_token_words)):
                        if span_token_words[span_token_id] == list_token_words[token_id+span_token_id]:
                            match +=1
                    # update token_start & token_end in span if complete span text found
                    if match == len(span_token_words):
                        span['token_start'] = token_id
                        span['token_end'] = token_id+len(span_token_words)-1

            list_spans.append(span)
        #---------------------------------------------------
        # 3. just for debugging(remove later)    
        if DEBUG:
            print_info(row.text, row.tokens, row.spans, list_tokens, list_spans)
        
    return list_spans, list_tokens

def resolve_issue2(df, DEBUG):    
    # Update Spans
    start_time=datetime.datetime.now()
    
    df[['spans','tokens']] = df.apply(update_span, args=(DEBUG, ), axis=1, result_type='expand')
    
    end_time=datetime.datetime.now()
    print("Time taken to resolve issue2: (in sec) ",end_time-start_time)
    
    return total_rows_impected_task2


# fill NaN with "NaN" & true, false in tokens['ws'], "_is_binary" with True, False to avoide error in training pipeline
def other_corrections(df):
    # fill NaN with 'NA' 
    df= df.fillna('NA')
    
    return df


def resolve_all_issues(df, data_correction_df_dict, DEBUG):
    print("Resolving issue3...")
    print('-'*40)
    impacted_rows = resolve_issue3(df, data_correction_df_dict, DEBUG) 
    print('Resolved issue3 (deal with spans have more than 1 label), impacted rows: ',impacted_rows)
    print('='*80)
    print()

    #resolve_spans_with_non_noun_phrases
    print("Resolving issue4...")
    print('-'*40)
    impacted_rows = resolve_issue4(df, data_correction_df_dict, DEBUG)
    print('Resolved issue4 (resolve_spans_with_non_noun_phrases), impacted rows: ',impacted_rows)
    print('='*80)
    print()

    #remove leading trailing_non alphanumeric char from spans
    print("Resolving issue2..")
    print('-'*40)
    impacted_rows = resolve_issue2(df, DEBUG)
    print('Resolved issue2 (remove leading trailing_non alphanumeric char from spans), impacted rows: ',impacted_rows)
    print('='*80)
    print()       

    #TODO
    # fill NaN with "NaN" & true, false in tokens['ws'], "_is_binary" with True, False to avoide error in training pipeline
    df = other_corrections(df)
    
    return df, data_correction_df_dict


def stat_to_json(file_path):
    s_obj = os.stat(file_path)
    return {k: getattr(s_obj, k) for k in dir(s_obj) if k.startswith('st_')}

def save_metadata(data_correction_tobe_verified_file_path, metadata_file_path, current_version):
    metadata_dict = stat_to_json(data_correction_tobe_verified_file_path)
    metadata_dict['version'] = current_version+1

    data_correction_df_dict = read_excel(data_correction_tobe_verified_file_path)
    for sheet_name in data_correction_df_dict.keys():
        metadata_dict[sheet_name] = {'shape': data_correction_df_dict[sheet_name].shape}

    # save metadata
    with open(metadata_file_path, 'w') as outfile:
        json.dump(metadata_dict, outfile)

def read_metdata(metadata_file_path):
    metadata = {}
    if os.path.exists(metadata_file_path):
        # read metadata
        with open(metadata_file_path) as json_file:
            metadata = json.load(json_file)    
    return metadata

# function to calculate hash value of row
#fn_calculate_hash = lambda row: hash(tuple(row))
def fn_calculate_hash(row):
    try:
        tpl = (row['lower_span'], row['label'], row['IS_VERIFIED'])
    except:
        tpl = (row['lower_span'], row['corrected_span'], row['IS_VERIFIED'])

    return hash(tpl)
    
def compare_files(original_file, updated_file):
    is_same = True
    data_correction_df_dict_changed = {}
    data_correction_df_dict_original = read_excel(original_file)
    data_correction_df_dict_updated = read_excel(updated_file)
    
    # verify shape
    for sheet in data_correction_df_dict_original.keys():
        if data_correction_df_dict_original[sheet].shape[0] != data_correction_df_dict_updated[sheet].shape[0]:
            is_same = False
            print(f"Shape of {sheet} is changed")
            
    # find added & removed rows, it will show updated rows in reoved and added both  
    for sheet in data_correction_df_dict_original.keys():
        # find hash values for all rows 
        data_correction_df_dict_original[sheet]['hash_value'] = data_correction_df_dict_original[sheet].apply(fn_calculate_hash, axis = 1)
        data_correction_df_dict_updated[sheet]['hash_value'] = data_correction_df_dict_updated[sheet].apply(fn_calculate_hash, axis = 1)
        
        data_correction_df_dict_original[sheet] = data_correction_df_dict_original[sheet].set_index('hash_value')
        data_correction_df_dict_updated[sheet] = data_correction_df_dict_updated[sheet].set_index('hash_value')
        
        # find hash values for all rows 
        deleted_rows = set(data_correction_df_dict_original[sheet].index) - set(data_correction_df_dict_updated[sheet].index)
        added_rows = set(data_correction_df_dict_updated[sheet].index) - set(data_correction_df_dict_original[sheet].index)

        if len(deleted_rows)>0:
            is_same = False
            data_correction_df_dict_changed[f'deleted_rows_{sheet}'] = data_correction_df_dict_original[sheet].loc[deleted_rows]
            
        if len(added_rows)>0:
            is_same = False
            data_correction_df_dict_changed[f'added_rows_{sheet}'] = data_correction_df_dict_updated[sheet].loc[added_rows]
                  
    # raise if same
    if is_same:
        raise Exception('file is not updated by analyst')
    else:
        return data_correction_df_dict_changed


def verify_input_corrected_labels(df_corrected_labels, model_name):   
    # fetch the specific schema
    schema_corrected_labels = ps.dict_schemas[f'{model_name}_input_corrected_labels']
    
    # Validating the data
    schema_corrected_labels.validate(df_corrected_labels, lazy = True)
    

def verify_input_corrected_spans(df_corrected_spans, model_name):
    # fetch the specific schema
    schema_corrected_spans = ps.dict_schemas[f'{model_name}_input_corrected_spans']

    # Validating the data
    schema_corrected_spans.validate(df_corrected_spans, lazy = True)
        
    
def verify_input_data_correction_file(data_correction_df_dict, current_version, model_name):
    # Input to python script, output from analyst
    
    # data_correction_tobe_verified_file_path(BEFORE)
    data_correction_tobe_verified_file_path = f"{MODEL_INPUT_PATH}/data_correction_tobe_verified_v{current_version}.xlsx"
    # file updated and renamed by analyst (AFTER)
    data_correction_file_path = f"{MODEL_INPUT_PATH}/data_correction_v{current_version}.xlsx"
    
    if os.path.exists(data_correction_tobe_verified_file_path):
        try:
            data_correction_df_dict_changed = compare_files(data_correction_tobe_verified_file_path, data_correction_file_path)
            # write all data_correction_changes to excel
            write_excel(data_correction_df_dict_changed, MODEL_INPUT_PATH+"/report_data_correction_changes.xlsx")
            print(f"data_correction file is updated, details are saved in <report_data_correction_changes.xlsx> file")
        except Exception as e:
            print("data_correction file is not updated", e)
    
    try:
        verify_input_corrected_labels(data_correction_df_dict['corrected_labels'], model_name)
        verify_input_corrected_spans(data_correction_df_dict['corrected_spans'], model_name)
    except Exception as e:
        print("\nPandera exception while validating input from analyst")
        print('='*80)
        print(e)
        print('='*80)

        
def verify_output_corrected_labels(df_corrected_labels, model_name):    
    # fetch the specific schema
    schema_corrected_labels = ps.dict_schemas[f'{model_name}_output_corrected_labels']
    
    # Validating the data
    schema_corrected_labels.validate(df_corrected_labels, lazy = True)
    

def verify_output_corrected_spans(df_corrected_spans, model_name):
    # fetch the specific schema
    schema_corrected_spans = ps.dict_schemas[f'{model_name}_output_corrected_spans']
    
    # Validating the data
    schema_corrected_spans.validate(df_corrected_spans, lazy = True)

def verify_output_data_correction_file(data_correction_df_dict, model_name):
    #input to analyst
    try:
        verify_output_corrected_labels(data_correction_df_dict['corrected_labels'], model_name)
        verify_output_corrected_spans(data_correction_df_dict['corrected_spans'], model_name)
    except Exception as e:
        print("\nPandera exception while validating output to analyst")
        print('='*80)
        print(e)
        print('='*80)


# save if any sheet in data_correction_df_dict is updated
def save_updated_data_correction_df_dict(data_correction_df_dict, data_correction_sheets_actual_size_dict, 
                                         data_correction_file_path, metadata_file_path, current_version, model_name):    
    data_correction_tobe_verified_file_path = f"{MODEL_INPUT_PATH}/data_correction_tobe_verified_v{current_version+1}.xlsx"
    
    for sheet in data_correction_df_dict.keys():
        if data_correction_sheets_actual_size_dict[sheet] != data_correction_df_dict[sheet].shape[0]:
            # save data data_correction_tobe_verified.xlsx
            write_excel(data_correction_df_dict, data_correction_tobe_verified_file_path)
            save_metadata(data_correction_tobe_verified_file_path, metadata_file_path, current_version)
            
            # validating data_correction_tobe_verified.xlsx before it goes as input to anlyst
            verify_output_data_correction_file(read_excel(data_correction_tobe_verified_file_path), model_name) 
            
            print(f'Found some new spans & multilabels to be verified, details are saved in <{data_correction_tobe_verified_file_path}>')
            break

# read_data_correction_df_dict & verify_input_data_correction_file
def read_data_correction_df_dict(data_correction_metadata_file_path, model_name):
    # get data_correction file version
    current_version = 0 # defaut value
    metadata = read_metdata(data_correction_metadata_file_path)
    if len(metadata)>0:
        current_version = metadata['version']

    data_correction_file_path = f"{MODEL_INPUT_PATH}/data_correction_v{current_version}.xlsx"
    try:
        data_correction_df_dict = read_excel(data_correction_file_path)
    except:
        print(data_correction_file_path, " file doesn't Exist!")
        sys.exit()

    # verify_data_correction_file
    verify_input_data_correction_file(data_correction_df_dict, current_version, model_name) 
    
    return data_correction_df_dict, data_correction_file_path, metadata, current_version
    

def clean_data(input_filename, out_filename, model_name, DEBUG):
    #read jsonl to df
    raw_data = list(srsly.read_jsonl(input_filename))
    df = pd.DataFrame(raw_data)
    
    # read data_correction.xlsx
    data_correction_metadata_file_path = MODEL_INPUT_PATH+'/data_correction_metadata.json'
    data_correction_df_dict, data_correction_file_path, metadata, current_version = read_data_correction_df_dict(
                                                                                        data_correction_metadata_file_path, model_name)       

    # keep data_correction_sheets_actual_size, to compare it with updated data_correction_df_dict
    data_correction_sheets_actual_size_dict = {sheet:data_correction_df_dict[sheet].shape[0] 
                                               for sheet in data_correction_df_dict.keys()}
    
    #resolve_all_issues
    df, data_correction_df_dict = resolve_all_issues(df, data_correction_df_dict, DEBUG) 
        
    # dump df to jsonl file
    if not os.path.exists(ASSETS_PATH):
        os.makedirs(ASSETS_PATH)
    with jsonlines.open(out_filename, mode='w') as writer:
        writer.write_all(df.to_dict(orient='records'))
    print(f'Updated the input data & saved at <{out_filename}>')
        
    # save if any sheet in data_correction_df_dict is updated
    save_updated_data_correction_df_dict(data_correction_df_dict, data_correction_sheets_actual_size_dict, 
                                         data_correction_file_path, data_correction_metadata_file_path, current_version, model_name)


def train_test_split_util(cleaned_filename, train_output_path, test_output_path, test_size):
    with jsonlines.open(cleaned_filename) as reader:
        data = list(reader)
        #split_index = int(len(data)*(1-test_size))
        train, test = train_test_split(data, test_size = test_size)
        
        #train_filename = f"{cleaned_filename[:-6]}_train.jsonl"
        with jsonlines.open(train_output_path, mode='w') as writer:
            writer.write_all(train)
          
        #eval_filename = f"{cleaned_filename[:-6]}_eval.jsonl"
        with jsonlines.open(test_output_path, mode='w') as writer:
            writer.write_all(test)


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    train_output_path: Path = typer.Argument(..., dir_okay=False),
    test_output_path: Path = typer.Argument(..., dir_okay=False),
    model_name = typer.Argument(..., dir_okay=False),):

   # define some global vars
    global DEBUG, ASSETS_PATH, MODEL_INPUT_PATH, MODEL_NAME
    MODEL_NAME = model_name.lower()
    
     # folder path
    ASSETS_PATH = f"./assets/{MODEL_NAME}"
    MODEL_INPUT_PATH = ASSETS_PATH+"/model_input"

    cleaned_filename = ASSETS_PATH+"/cleaned_data.jsonl"
    DEBUG = False
    test_size = .1 #10%

    clean_data(input_path, cleaned_filename, MODEL_NAME, DEBUG)
    train_test_split_util(cleaned_filename, train_output_path, test_output_path, test_size)


if __name__ == "__main__":
     typer.run(main)




