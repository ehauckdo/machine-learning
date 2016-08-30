#!  /usr/bin/env python
# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import sys
from shapely.geometry import MultiPoint
import gc
from sklearn.preprocessing import LabelEncoder

def merge_features(df):

    df = merge_phone_brand(df)

    df = merge_number_events(df)

    df = merge_device_centroid(df)

    #df = merge_events(df)
    #df = merge_app_events(df)
    #df = merge_app_labels(df)
    #df = merge_label_categories(df)
    #print_report(df)

    return df

def load_train(subset_size=1):
    df = pd.read_csv("input/gender_age_train.csv", dtype={'device_id':np.str})
    df = df.sample(frac=subset_size)
    df = merge_features(df)
    df = encode_features(df)
    return df

def load_test(subset_size=1):    
    df = pd.read_csv("input/gender_age_test.csv", dtype={'device_id':np.str})
    df = df.sample(frac=subset_size)
    df = merge_features(df)
    df = encode_features(df)
    return df

def encode_features(df):
    encoder = LabelEncoder()
    df["phone_brand"] = encoder.fit_transform(df['phone_brand'])
    df["device_model"] = encoder.fit_transform(df['device_model'])
    return df

def generate_subset():

    train = pd.read_csv("input/gender_age_train.csv", dtype={'device_id':np.str})
    df = train

    df = train.head(5)

    extra_row =  train.iloc[[32880]]
    df = df.append(extra_row)

    df.to_csv("output/train_subset.csv", index=False)
    df = merge_features(df)

    return df

def merge_phone_brand(df):

    print("Loading and merging phone_brand_device_model.csv...")

    phone_brand = pd.read_csv("input/phone_brand_device_model.csv", dtype={'device_id':np.str})
    phone_brand = phone_brand.drop_duplicates(subset=phone_brand.columns[0])
    phone_brand = translate_phone_brands(phone_brand)
    phone_brand = phone_brand.fillna("Unkown")

    df = pd.merge(df,phone_brand, left_on='device_id', right_on='device_id', how='left', suffixes=['','_'])

    df.to_csv("output/train_subset2.csv", index=False)

    print("Done.")

    return df

def merge_events(df):

    print("Loading and merging events.csv...")

    events = pd.read_csv("input/events.csv", dtype={'device_id':np.str, 'event_id':np.str})

    df = pd.merge(df,events,left_on='device_id',right_on='device_id', how='left', suffixes=['','_'])

    df.to_csv("output/train_subset3.csv", index=False)

    print("Done.")

    return df

def merge_app_events(df):

    print("Loading and merging app_events.csv...")

    app_events=pd.read_csv('input/app_events.csv', dtype={'event_id':np.str, 'app_id':np.str})

    df = pd.merge(df,app_events,left_on='event_id',right_on='event_id', how='left', suffixes=['','_'])

    df.to_csv("output/train_subset4.csv", index=False)

    print("Done.")

    return df

def merge_app_labels(df):

    print("Loading and merging app_labels.csv...")

    app_labels= pd.read_csv('input/app_labels.csv', dtype={'label_id':np.str, 'app_id':np.str})

    df = pd.merge(df,app_labels,left_on='app_id',right_on='app_id', how='left', suffixes=['','_'])

    df.to_csv("output/train_subset5.csv", index=False)

    print("Done.")

    return df

def merge_label_categories(df):

    print("Loading and merging app_categories.csv...")

    label_categories = pd.read_csv('input/label_categories.csv', dtype={'label_id':np.str, 'category':np.str})
    label_categories = label_categories.fillna("unkown")

    df = pd.merge(df,label_categories,left_on='label_id',right_on='label_id', how='left', suffixes=['','_'])

    df.to_csv("output/train_subset6.csv", index=False)

    print("Done.")

    return df

def merge_number_events(df):

    print("Merging number_events...")

    events = pd.read_csv("input/events.csv", dtype={'device_id':np.str, 'event_id':np.str})
    
    events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')

    events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')

    df = pd.merge(df, events_small, how='left', on='device_id', left_index=True)
    df['counts'].fillna(0, inplace=True)
    df['counts'] = df['counts'].astype(int)

    print("Done.")

    return df

def merge_installed_active_apps(df):
    
    ape = pd.read_csv("input/app_events.csv", dtype={'event_id':np.str})
    ape['installed'] = ape.groupby(['event_id'])['is_installed'].transform('sum')
    ape['active'] = ape.groupby(['event_id'])['is_active'].transform('sum')
    ape.drop(['is_installed', 'is_active'], axis=1, inplace=True)
    ape.drop_duplicates('event_id', keep='first', inplace=True)
    ape.drop(['app_id'], axis=1, inplace=True)

    events = pd.read_csv("input/events.csv", dtype={'device_id':np.str, 'event_id':np.str})
    events.drop(['timestamp', 'longitude', 'latitude'], axis=1, inplace=True)
 
    events = pd.merge(events, ape, how='left', on='event_id', left_index=True)
    events.drop('event_id', axis=1, inplace=True)
    
    df = pd.merge(df, events, how='left', left_on='device_id', right_on='device_id', left_index=True)
    df['installed'].fillna(0, inplace=True)
    df['installed'] = df['installed'].astype(int)
    df['active'].fillna(0, inplace=True)
    df['active'] = df['active'].astype(int)

    return df    

def merge_device_centroid(df, coord = True):
    
    print("Merging device_centroid...")

    events = pd.read_csv("input/events.csv", dtype={'device_id':np.str, 'event_id':np.str})

    def get_centroid(device_id):

        events_device = events.loc[events['device_id'] == device_id]
        events_device['lat_long'] = events_device[['longitude', 'latitude']].apply(tuple, axis=1)
        events_device = events_device.drop_duplicates(subset=['lat_long'])

        lat_long_list = events_device['lat_long'].tolist()
        points = MultiPoint(lat_long_list)
        
        #True centroid, not necessarily an existing point
        #print points.centroid 
        
        if coord:
            try:
                x = points.representative_point().x
                y = points.representative_point().y
                if (x, y) != (0, 0):
                    return (x, y)
            except:
                pass
        else:
            try:
                x = round(points.representative_point().x, 0)
                y = round(points.representative_point().y, 0)
                if (x, y) != (0, 0):
                    return (x, y)
            except:
                pass
        return 0

    df["Centroid"] = df["device_id"].apply(get_centroid)
    
    print("Done.")

    return df

def print_report(df):

    print("\nFinal shape:")
    unique_devices = [x for x in df["device_id"].unique() if str(x) != 'nan']
    print("Unique devices: {0}.".format(len(unique_devices)))
    unique_events = [x for x in df["event_id"].unique() if str(x) != 'nan']
    print("Unique events: {0}.".format(len(unique_events)))
    unique_app_events = [x for x in df["app_id"].unique() if str(x) != 'nan']
    print("Unique apps: {0}.\n".format(len(unique_app_events)))

    list_unique_devices = df['device_id'].unique()
    if False:
        for device in list_unique_devices:
            print("Device ID: {0}:".format(device))
            events = df.loc[df['device_id'] == device]
            unique_events = [x for x in events["event_id"].unique() if str(x) != 'nan']
            print("Total events: {0}.".format(len(unique_events)))
            if unique_events:
                unique_app_events = [x for x in events["app_id"].unique() if str(x) != 'nan']
                print("Total app_events: {0}.".format(len(unique_app_events)))
                unique_categories = [x for x in events["label_id"].unique() if str(x) != 'nan']
                print("Total categories: {0}.".format(len(unique_categories)))
            print("")

    return df

def translate_phone_brands(df):
    df.phone_brand = df.phone_brand.map(pd.Series(english_phone_brands_mapping), na_action='ignore')
    return df

english_phone_brands_mapping = {
    "三星": "samsung",
    "天语": "Ktouch",
    "海信": "hisense",
    "联想": "lenovo",
    "欧比": "obi",
    "爱派尔": "ipair",
    "努比亚": "nubia",
    "优米": "youmi",
    "朵唯": "dowe",
    "黑米": "heymi",
    "锤子": "hammer",
    "酷比魔方": "koobee",
    "美图": "meitu",
    "尼比鲁": "nibilu",
    "一加": "oneplus",
    "优购": "yougo",
    "诺基亚": "nokia",
    "糖葫芦": "candy",
    "中国移动": "ccmc",
    "语信": "yuxin",
    "基伍": "kiwu",
    "青橙": "greeno",
    "华硕": "asus",
    "夏新": "panosonic",
    "维图": "weitu",
    "艾优尼": "aiyouni",
    "摩托罗拉": "moto",
    "乡米": "xiangmi",
    "米奇": "micky",
    "大可乐": "bigcola",
    "沃普丰": "wpf",
    "神舟": "hasse",
    "摩乐": "mole",
    "飞秒": "fs",
    "米歌": "mige",
    "富可视": "fks",
    "德赛": "desci",
    "梦米": "mengmi",
    "乐视": "lshi",
    "小杨树": "smallt",
    "纽曼": "newman",
    "邦华": "banghua",
    "E派": "epai",
    "易派": "epai",
    "普耐尔": "pner",
    "欧新": "ouxin",
    "西米": "ximi",
    "海尔": "haier",
    "波导": "bodao",
    "糯米": "nuomi",
    "唯米": "weimi",
    "酷珀": "kupo",
    "谷歌": "google",
    "昂达": "ada",
    "聆韵": "lingyun",
    "小米": "Xiaomi",
    "华为": "Huawei",
    "魅族": "Meizu",
    "中兴": "ZTE",
    "酷派": "Coolpad",
    "金立": "Gionee",
    "SUGAR": "SUGAR",
    "OPPO": "OPPO",
    "vivo": "vivo",
    "HTC": "HTC",
    "LG": "LG",
    "ZUK": "ZUK",
    "TCL": "TCL",
    "LOGO": "LOGO",
    "SUGAR": "SUGAR",
    "Lovme": "Lovme",
    "PPTV": "PPTV",
    "ZOYE": "ZOYE",
    "MIL": "MIL",
    "索尼" : "Sony",
    "欧博信" : "Opssom",
    "奇酷" : "Qiku",
    "酷比" : "CUBE",
    "康佳" : "Konka",
    "亿通" : "Yitong",
    "金星数码" : "JXD",
    "至尊宝" : "Monkey King",
    "百立丰" : "Hundred Li Feng",
    "贝尔丰" : "Bifer",
    "百加" : "Bacardi",
    "诺亚信" : "Noain",
    "广信" : "Kingsun",
    "世纪天元" : "Ctyon",
    "青葱" : "Cong",
    "果米" : "Taobao",
    "斐讯" : "Phicomm",
    "长虹" : "Changhong",
    "欧奇" : "Oukimobile",
    "先锋" : "XFPLAY",
    "台电" : "Teclast",
    "大Q" : "Daq",
    "蓝魔" : "Ramos",
    "奥克斯" : "AUX"
}

