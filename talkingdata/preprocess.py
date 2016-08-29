#!  /usr/bin/env python
# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import sys
from shapely.geometry import MultiPoint
from preprocess import translate_phone_brands
import gc

def generate_subset():

    df = load_data()

    df = merge_phone_brand(df)

    df = merge_events(df)

    df = merge_app_events(df)

    df = merge_app_labels(df)

    df = merge_label_categories(df)

    print_report(df)

    return df

def load_data():

    # ====== Load subset of train ======
    print("Loading train set...")

    train = pd.read_csv("input/gender_age_train.csv", dtype={'device_id':np.str})
    df = train

    # get first 5 rows as a smple
    df = train.head(5)

    # get a specific device that has many events/app_events
    extra_row =  train.iloc[[32880]]
    df = df.append(extra_row)

    # save first instance of train_subset: user + phone
    df.to_csv("output/train_subset.csv", index=False)

    print("Done.")

    return df

def merge_phone_brand(df):

    # ====== Load phone_brand and merge ======
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

    # ====== Load events and merge ===========
    print("Loading and merging events.csv...")

    events = pd.read_csv("input/events.csv", dtype={'device_id':np.str, 'event_id':np.str})

    df = pd.merge(df,events,left_on='device_id',right_on='device_id', how='left', suffixes=['','_'])

    # save second instance of train_subset: user + phone + events
    df.to_csv("output/train_subset3.csv", index=False)

    print("Done.")

    return df

def merge_app_events(df):

    # ====== Load app_events and merge ===========
    print("Loading and merging app_events.csv...")

    app_events=pd.read_csv('input/app_events.csv', dtype={'event_id':np.str, 'app_id':np.str})

    df = pd.merge(df,app_events,left_on='event_id',right_on='event_id', how='left', suffixes=['','_'])

    # save third instance of train_subset: user + phone + events + app_events
    df.to_csv("output/train_subset4.csv", index=False)

    print("Done.")

    return df

def merge_app_labels(df):

    # ====== Load app_labels and merge =========
    print("Loading and merging app_labels.csv...")

    app_labels= pd.read_csv('input/app_labels.csv', dtype={'label_id':np.str, 'app_id':np.str})

    df = pd.merge(df,app_labels,left_on='app_id',right_on='app_id', how='left', suffixes=['','_'])

    # save fourth instance of train_subset: user + phone + events + app_events + app_labels
    df.to_csv("output/train_subset5.csv", index=False)

    print("Done.")

    return df
def merge_label_categories(df):

    # ====== Load label_categories and merge ======
    print("Loading and merging app_categories.csv...")

    label_categories = pd.read_csv('input/label_categories.csv', dtype={'label_id':np.str, 'category':np.str})
    label_categories = label_categories.fillna("unkown")

    df = pd.merge(df,label_categories,left_on='label_id',right_on='label_id', how='left', suffixes=['','_'])

    # save fifth instance of train_subset: user + phone + events + app_events + app_labels
    df.to_csv("output/train_subset6.csv", index=False)

    print("Done.")

    return df
def merge_device_centroid(df, coord = True):
    df = df[['device_id']].drop_duplicates()

    events = pd.read_csv("input/events.csv", dtype={'device_id':np.str, 'event_id':np.str})

    def get_centroid(device_id):

        print("DEVICE ID: ", device_id)
        events_device = events.loc[events['device_id'] == device_id]
        events_device['lat_long'] = events_device[['latitude', 'longitude']].apply(tuple, axis=1)
        events_device = events_device.drop_duplicates(subset=['lat_long'])
        print("FOUND:")
        print(events_device)
        print("")

        lat_long_list = events_device['lat_long'].tolist()
        points = MultiPoint(lat_long_list)
        print points.centroid #True centroid, not necessarily an existing point
        print points.representative_point() #A represenative point, not centroid,
                                    #that is guarnateed to be with the geometry
        x = 0;
        y = 0;
        if coord:
            try:
                x = points.representative_point().x
                y = points.representative_point().y
            except:
                pass
        else:
            try:
                x = round(points.representative_point().x, 0)
                y = round(points.representative_point().y, 0)
            except:
                pass
        return (x, y)

    df["Centroid"] = df["device_id"].apply(get_centroid)
    print(df)

def print_report(df):

    # ====== Print report ======

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
    df.phone_brand = df.phone_brand.map(pandas.Series(english_phone_brands_mapping), na_action='ignore')
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

