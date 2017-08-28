# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 21:44:06 2017

@author: PengChen
"""
import pandas as pd
from collections import defaultdict
from pandas import DataFrame
#from sklearn.preprocessing import PolynomialFeatures
from phone_dict_info import phone_info_map,phone_brand_map,term_brand_map
#import gc

def process(input, output,save_data = False):    
    data = pd.read_csv(input,encoding='utf-8',index_col='用户标识')    
    #获取全零的比例
    #a=train_data==0
    #b=a.sum()
    #pro = b/len(train_data)
    
    #删除全零特征
    process1_data = data.drop(['SuningEbuy','爱奇艺动画屋','必应搜索','漫入省份'], axis = 1)
    #填补缺失值(直接补0效果不一定好，后期考虑改进)
    process1_data = process1_data.fillna(0)
    
    process1_data = process1_data.loc[(process1_data['每月的大致刷卡消费次数']<30000)
                                &(process1_data['固定联络圈规模']<3000)
                                &(process1_data['访问餐饮网站的次数']<3000)
                                &(process1_data['访问综合网站的次数']<11000)
                                &(process1_data['访问旅游网站的次数']<10000)
                                &(process1_data['访问生活网站的次数']<80000)
                                &(process1_data['访问金融网站的次数']<40000)
                                &(process1_data['访问IT网站的次数']<20000)
                                &(process1_data['访问教育网站的次数']<20000)
                                &(process1_data['访问通话网站的次数']<15000)
                                &(process1_data['访问社交网站的次数']<20000)
                                &(process1_data['访问汽车网站的次数']<40000)

                                ] 
    #构造特征1：大致消费汇总=大致消费水平*每月的大致刷卡消费次数 （为避免相乘后全为零，两者都加1）
    process1_data['大致消费汇总'] = process1_data['大致消费水平'] * process1_data['每月的大致刷卡消费次数']
    
    #构造特征2：手机信息=手机品牌+手机终端型号
    process1_data['手机信息'] = process1_data['手机品牌'].astype('str') + process1_data['手机终端型号'].astype('str')
    #类别属性转化为数值属性
    #phone_info_map = {label:idx for idx,label in enumerate(set(process1_data['手机信息']))}
    process1_data['手机信息'] = process1_data['手机信息'].map(phone_info_map)
    
    #phone_brand_map = {label:idx for idx,label in enumerate(set(process1_data['手机品牌']))}
    process1_data['手机品牌'] = process1_data['手机品牌'].map(phone_brand_map)
    process1_data.fillna(0,inplace=True)
    #term_brand_map = {label:idx for idx,label in enumerate(set(process1_data['手机终端型号']))}
    process1_data['手机终端型号'] = process1_data['手机终端型号'].map(term_brand_map)
    
    yes_no_map = {'是':1, '否':0}
    process1_data['是否有跨省行为'] = process1_data['是否有跨省行为'].map(yes_no_map)
    process1_data['是否有出境行为'] = process1_data['是否有出境行为'].map(yes_no_map)
    
    #构造特征3: 跨省或出境行为 = 是否跨省行为 + 是否有出境行为
    process1_data['跨省或出境行为'] = process1_data['是否有跨省行为'] + process1_data['是否有出境行为']
    
    #由于不同用户的漫出省份数目不同，所有将该特征转化为漫出省份数目
    manchu =  process1_data['漫出省份']
    dd = defaultdict()
    for idx,i in manchu.iteritems():
        if ',' in i:
            dd[idx] = len(i.split(','))
        elif i == '无':
            dd[idx] = 0
        else:
            dd[idx] = 1
            
    dd1 = DataFrame([dd]).T
    process1_data['漫出省份'] = dd1
    
    #所有软件的使用总次数 (加1是为了防止除数为零)
    soft_use_total = process1_data.iloc[:,7:304].sum(1)+1
    
    #网站的总点击次数
    web_click_num = process1_data.iloc[:,308:338].sum(1)+1
    #构造特征：美食类
    process1_data['美食类'] = (process1_data['星巴克中国'] + process1_data['美团外卖']+process1_data['百度外卖']+process1_data['百度糯米']+process1_data['大众点评']+process1_data['饿了么']+process1_data['肯德基']+process1_data['厨房故事食谱']+1)
    process1_data['美食类软件使用率'] = process1_data['美食类']/soft_use_total
    process1_data['美食类加权1'] =  process1_data['美食类'] * process1_data['访问餐饮网站的次数'] 
    process1_data['美食类加权2'] =  process1_data['美食类'] + process1_data['访问餐饮网站的次数'] 
    # process1_data['美食类总占比'] = (process1_data['美食类'] + process1_data['访问餐饮网站的次数'])/(soft_use_total+web_click_num-1)
    #构造特征： 美食类占比
    process1_data['星巴克中国占比'] = process1_data['星巴克中国']/process1_data['美食类']    
    process1_data['美团占比'] =  process1_data['美团外卖']/process1_data['美食类']
    process1_data['糯米团购占比'] = process1_data['糯米团购']/ process1_data['美食类']   
    process1_data['百度占比'] = (process1_data['百度外卖']+process1_data['百度糯米'])/process1_data['美食类']
    process1_data['大众点评占比'] = process1_data['大众点评']/process1_data['美食类']
    process1_data['饿了么占比'] = process1_data['饿了么']/process1_data['美食类']
    process1_data['肯德基占比'] = process1_data['肯德基']/process1_data['美食类']
    #构造特征： 爱买
    process1_data['爱买'] = process1_data['什么值得买']+process1_data['闲鱼']+process1_data['58同城']+process1_data['亚马逊']+process1_data['小米商城']+process1_data['当当网']+process1_data['美团团购']+process1_data['美团网']+process1_data['蘑菇街']+process1_data['糯米团购']+process1_data['天猫']+process1_data['手机天猫']+process1_data['购物大厅']+process1_data['支付宝']+process1_data['大麦']+process1_data['京东']+process1_data['京东到家']+process1_data['手机淘宝']+process1_data['好又多商城']+process1_data['唯品会']+process1_data['一号店']+process1_data['美团']+process1_data['格瓦拉']+process1_data['去哪儿生活']+process1_data['赶集网']+process1_data['篱笆社区']+process1_data['支付宝钱包']+process1_data['壹钱包']+process1_data['中国批发市场']+process1_data['中国工商银行']+process1_data['工行手机银行']+process1_data['东亚银行']+process1_data['交通银行']+process1_data['农行掌上银行']+process1_data['中国建设银行']
    process1_data['爱买软件使用率'] = process1_data['爱买']/soft_use_total   
    process1_data['爱买加权'] = process1_data['爱买']* process1_data['访问购物网站的次数']
   # process1_data['爱买总占比'] = (process1_data['爱买'] + process1_data['访问购物网站的次数'])/(soft_use_total+web_click_num-1)
    
    #构造特征： 喜欢使用哪种购物app
    process1_data['手机天猫占比'] = (process1_data['手机天猫']+process1_data['天猫']+process1_data['手机淘宝'])/(process1_data['爱买']+1)
    process1_data['京东占比'] = (process1_data['京东']+process1_data['京东到家'])/(process1_data['爱买']+1)
    process1_data['唯品会占比'] = process1_data['唯品会']/(process1_data['爱买']+1)
    process1_data['一号店占比'] = process1_data['一号店']/(process1_data['爱买']+1)
   
    #构造特征： 爱美
    process1_data['爱美'] = process1_data['女孩相机']+process1_data['潮自拍']+process1_data['美柚']+process1_data['拍客']+process1_data['myotee 脸萌']+process1_data['百度魔图']+process1_data['天天P图']+process1_data['拼立得']+process1_data['美图秀秀']+process1_data['gif 快手']
    process1_data['爱美软件使用率'] = (process1_data['百度魔图']+process1_data['天天P图']+process1_data['拼立得']+process1_data['美图秀秀'])/soft_use_total
           
    #构造特征： 爱玩
    process1_data['爱玩'] = process1_data['格瓦拉']+process1_data['去哪儿生活']+process1_data['携程旅行']+process1_data['易游人']+process1_data['大麦']+process1_data['阿里旅行']+process1_data['去哪儿旅行']+process1_data['同程旅游']
    process1_data['爱玩软件使用率'] = (process1_data['格瓦拉']+process1_data['去哪儿生活']+process1_data['携程旅行']+process1_data['易游人']+process1_data['大麦']+process1_data['阿里旅行']+process1_data['去哪儿旅行']+process1_data['同程旅游'])/soft_use_total   
    process1_data['爱玩加权'] = process1_data['爱玩']* process1_data['访问旅游网站的次数']
    # process1_data['爱玩总占比'] = (process1_data['爱玩'] + process1_data['访问旅游网站的次数'])/(soft_use_total+web_click_num-1)
    
    #构造特征：爱使用哪种爱玩app
    process1_data['格瓦拉占比'] = process1_data['格瓦拉']/(process1_data['爱玩']+1)
    process1_data['去哪儿占比'] = (process1_data['去哪儿生活']+process1_data['去哪儿旅行']) /(process1_data['爱玩']+1)
    process1_data['携程旅行占比'] = process1_data['携程旅行']/(process1_data['爱玩']+1)
    process1_data['易游人占比'] = process1_data['易游人']/(process1_data['爱玩']+1)
    process1_data['大麦占比'] = process1_data['大麦']/(process1_data['爱玩']+1)
    process1_data['阿里旅行占比'] = process1_data['阿里旅行']/(process1_data['爱玩']+1)
    process1_data['同程旅游占比'] = process1_data['同程旅游']/(process1_data['爱玩']+1)
    
    #构造特征： 吃喝玩乐
    process1_data['吃喝玩乐'] = process1_data['美食类']+process1_data['爱买']+process1_data['爱玩']
    process1_data['吃喝玩乐软件使用率'] = (process1_data['美食类']+process1_data['爱买']+process1_data['爱玩'])/soft_use_total
    
    #构造特征： 衣食住行
    process1_data['购物旅游'] = process1_data['访问购物网站的次数']+process1_data['访问旅游网站的次数']   
    process1_data['购物旅游点击率'] = (process1_data['访问购物网站的次数']+process1_data['访问旅游网站的次数'])/web_click_num
    #构造特征： 吃喝玩乐衣食住行
    #process1_data['吃喝玩乐衣食住行'] = process1_data['吃喝玩乐']+process1_data['旅游购物']
    
    process1_data['打车'] = process1_data['嘀嘀打车']+process1_data['滴滴司机-专车']+process1_data['嘀嗒拼车']+process1_data['快的打车']+process1_data['易到用车']+process1_data['快的打车'] + process1_data['访问汽车网站的次数']+1
    process1_data['打车软件使用率'] = (process1_data['嘀嘀打车']+process1_data['滴滴司机-专车']+process1_data['嘀嗒拼车']+process1_data['快的打车']+process1_data['易到用车']+process1_data['快的打车'])/soft_use_total
    process1_data['打车加权'] = (process1_data['嘀嘀打车']+process1_data['滴滴司机-专车']+process1_data['嘀嗒拼车']+process1_data['快的打车']+process1_data['易到用车']+process1_data['快的打车']) * (process1_data['访问汽车网站的次数'])
    #process1_data['打车总占比'] = (process1_data['打车'] + process1_data['汽车之家'] + process1_data['访问汽车网站的次数'])/(soft_use_total+web_click_num-1)
    
    #构造特征，喜欢使用哪种打车app
    Da_che = process1_data['嘀嘀打车']+process1_data['滴滴司机-专车']+process1_data['嘀嗒拼车']+process1_data['快的打车']+process1_data['易到用车']+process1_data['快的打车']+1
    process1_data['嘀嘀打车占比'] = process1_data['嘀嘀打车']/Da_che
    process1_data['滴滴司机-专车占比'] = process1_data['滴滴司机-专车']/Da_che
    process1_data['嘀嗒拼车占比'] = process1_data['嘀嗒拼车']/Da_che
    process1_data['快的打车占比'] = process1_data['快的打车']/Da_che
    process1_data['易到用车占比'] = process1_data['易到用车']/Da_che
    process1_data['快的打车占比'] = process1_data['快的打车']/Da_che
    
    process1_data['通信'] = process1_data['固定联络圈规模'] + process1_data['访问通话网站的次数']
    process1_data['通信加权'] = (process1_data['固定联络圈规模']) * (process1_data['访问通话网站的次数'])
    
    process1_data['地图'] = process1_data['腾讯地图'] + process1_data['高德地图']+process1_data['百度地图']+process1_data['苹果地图']+process1_data['谷歌地图']+process1_data['GoogleMap']+1
    process1_data['地图软件使用率'] = process1_data['地图']/ soft_use_total
    process1_data['腾讯地图占比'] = process1_data['腾讯地图']/process1_data['地图']
    process1_data['高德地图占比'] = process1_data['高德地图']/process1_data['地图']
    process1_data['百度地图占比'] = process1_data['百度地图']/process1_data['地图']
    process1_data['苹果地图占比'] = process1_data['苹果地图']/process1_data['地图']
    process1_data['谷歌地图占比'] = (process1_data['谷歌地图']+process1_data['GoogleMap'])/process1_data['地图']
    
    process1_data['地图加权'] = (process1_data['地图']) * (process1_data['访问地图网站的次数'])    
    process1_data['地图网站点击率'] = process1_data['访问地图网站的次数']/web_click_num
    process1_data['地图总占比'] = (process1_data['地图'] + process1_data['访问地图网站的次数'])/(soft_use_total+web_click_num-1)
        
    process1_data['汇总'] = process1_data['打车']+process1_data['通信']+process1_data['地图']+process1_data['爱玩']+process1_data['大致消费汇总']
			
    #构造均值特征
    process1_data['通信均值'] = (process1_data['通信']-process1_data['通信'].mean())/ process1_data['通信'].mean()
    process1_data['打车均值'] = (process1_data['打车']-process1_data['打车'].mean())/ process1_data['打车'].mean()
    process1_data['爱玩均值'] = (process1_data['爱玩']-process1_data['爱玩'].mean())/ process1_data['爱玩'].mean()
    process1_data['大致消费汇总均值'] = (process1_data['大致消费汇总']-process1_data['大致消费汇总'].mean())/ process1_data['大致消费汇总'].mean()
    process1_data['访问通话网站的次数均值'] = (process1_data['访问通话网站的次数']-process1_data['访问通话网站的次数'].mean())/ process1_data['访问通话网站的次数'].mean()
    process1_data['访问教育网站的次数均值'] = (process1_data['访问教育网站的次数']-process1_data['访问教育网站的次数'].mean())/ process1_data['访问教育网站的次数'].mean()
    process1_data['访问旅游网站的次数均值'] = (process1_data['访问旅游网站的次数']-process1_data['访问旅游网站的次数'].mean())/ process1_data['访问旅游网站的次数'].mean()
    process1_data['访问汽车网站的次数均值'] = (process1_data['访问汽车网站的次数']-process1_data['访问汽车网站的次数'].mean())/ process1_data['访问汽车网站的次数'].mean()    
    #process1_data['固定联络圈规模均值'] = (process1_data['固定联络圈规模']-process1_data['固定联络圈规模'].mean())/ process1_data['固定联络圈规模'].mean()    
    
    #构造特征：股票财经   苹果iphone股票  广发证券易淘金  海通e海通财  赢家理财高端版  涨乐财富通  中信证券高端版  财经杂志  股市热点  新浪财经  大智慧免费炒股软件
    # 同花顺  智远一户通  访问股票网站的次数  访问金融网站的次数
    process1_data['股票财经类'] = process1_data['苹果iphone股票']+process1_data['广发证券易淘金']+process1_data['海通e海通财']+process1_data['赢家理财高端版']+process1_data['涨乐财富通']+process1_data['中信证券高端版']+process1_data['财经杂志']+process1_data['股市热点']+process1_data['新浪财经']+process1_data['大智慧免费炒股软件']
    process1_data['股票财经类占比'] = process1_data['股票财经类']/(soft_use_total+1)
    

    #构造特征： app覆盖率=安装的app软件数目/总的软件数目    （这里认为只要用户使用过某软件就代表该用户安装了该软件）
    #          app活跃度=总的app软件使用次数/总的app软件安装数目
    process1_data['用户安装的app个数'] = (process1_data.iloc[:,7:304]!=0).sum(1)
    process1_data['app覆盖率'] = process1_data['用户安装的app个数']/297
    process1_data['app活跃度'] = (soft_use_total-1) / (process1_data['用户安装的app个数']+1)
	
    process1_data['通话_教育_旅游_汽车_procentage'] = (process1_data['访问通话网站的次数']+process1_data['访问教育网站的次数']+process1_data['访问旅游网站的次数']+process1_data['访问汽车网站的次数'])/web_click_num
    process1_data['通话网站点击率'] = process1_data['访问通话网站的次数']/web_click_num
    process1_data['教育网站点击率'] = process1_data['访问教育网站的次数']/web_click_num
    process1_data['汽车网站点击率'] = process1_data['访问汽车网站的次数']/web_click_num
    
    process1_data['股票网站点击率'] = process1_data['访问股票网站的次数']/web_click_num
    process1_data['新闻网站点击率'] = process1_data['访问新闻网站的次数']/web_click_num
    process1_data['育儿网站点击率'] = process1_data['访问育儿网站的次数']/web_click_num
    
    process1_data['消费汇总'] =(process1_data['大致消费水平']+process1_data['用户更换手机频次']) * process1_data['每月的大致刷卡消费次数']
    process1_data['网页类型覆盖率'] = (process1_data.iloc[:,308:338]!=0).sum(1)/30
    process1_data['网页类型活跃度'] = (web_click_num-1)/((process1_data.iloc[:,308:338]!=0).sum(1)+1)
      
    process1_data['美食类app活跃度占比'] = (process1_data['美食类']-process1_data['app活跃度'])/(process1_data['app活跃度']+1)
    process1_data['爱买app活跃度占比'] = (process1_data['爱买']-process1_data['app活跃度'])/(process1_data['app活跃度']+1)
    process1_data['爱美app活跃度占比'] = (process1_data['爱美']-process1_data['app活跃度'])/(process1_data['app活跃度']+1)
    process1_data['爱玩app活跃度占比'] = (process1_data['爱玩']-process1_data['app活跃度'])/(process1_data['app活跃度']+1)
    process1_data['吃喝玩乐app活跃度占比'] = (process1_data['吃喝玩乐']-process1_data['app活跃度'])/(process1_data['app活跃度']+1)
    process1_data['购物旅游app活跃度占比'] = (process1_data['购物旅游']-process1_data['app活跃度'])/(process1_data['app活跃度']+1)
    process1_data['打车app活跃度占比'] = (process1_data['打车']-process1_data['app活跃度'])/(process1_data['app活跃度']+1)
    process1_data['通信app活跃度占比'] = (process1_data['通信']-process1_data['app活跃度'])/(process1_data['app活跃度']+1)
    process1_data['地图app活跃度占比'] = (process1_data['地图']-process1_data['app活跃度'])/(process1_data['app活跃度']+1)
    process1_data['汇总app活跃度占比'] = (process1_data['汇总']-process1_data['app活跃度'])/(process1_data['app活跃度']+1)
    process1_data['出行相关'] = process1_data['车轮查违章']+process1_data['高铁管家']+process1_data['航旅纵横']
    process1_data['出行相关占比'] = process1_data['出行相关']/soft_use_total 
 
    process1_data['美食类占比'] = process1_data['美食类']/(process1_data['用户安装的app个数']+1)
    process1_data['爱买占比'] = process1_data['爱买']/(process1_data['用户安装的app个数']+1)
    process1_data['爱美占比'] = process1_data['爱美']/(process1_data['用户安装的app个数']+1)
    process1_data['爱玩占比'] = process1_data['爱玩']/(process1_data['用户安装的app个数']+1)
    process1_data['吃喝玩乐占比'] = process1_data['吃喝玩乐']/(process1_data['用户安装的app个数']+1)
    process1_data['购物旅游占比'] = process1_data['购物旅游']/(process1_data['用户安装的app个数']+1)
    process1_data['打车占比'] = process1_data['打车']/(process1_data['用户安装的app个数']+1)
    process1_data['通信占比'] = process1_data['通信']/(process1_data['用户安装的app个数']+1)
    process1_data['地图占比'] = process1_data['地图']/(process1_data['用户安装的app个数']+1)
    process1_data['汇总占比'] = process1_data['汇总']/(process1_data['用户安装的app个数']+1)
         
    return process1_data

input = r"C:\competition\liantong\diyiti\data\raw_data\train_data.csv"
output = r"C:\competition\liantong\第1题：算法题数据\data\raw_data\数据集1_用户标签_本地_训练集_process.csv"
process_train_data = process(input, output,save_data = False)
