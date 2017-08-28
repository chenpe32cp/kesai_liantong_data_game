# kesai_liantong_data_game
https://www.kesci.com/apps/home/#!/competition/59682b887284f10ace46baf3/content/0
	中国联通“沃+海创开放数据应用大赛”算法题（top2）
	src文件夹主要包含三个脚本
	1> data_processing.py 数据预处理脚本
	2> phone_dict_info.py 类别属性映射脚本
注：由于数据集中存在类别变量（分类变量），通常可以进行onehot编码或者将类别变量直接映射为一个整型数字，由于这里的类别变量取值太多，所以采用第二种方法。
直接将类别变量映射为数字时，需要注意类别变量的大小是否有意义。这里由于使用lightGBM模型，可以直接使用类别属性，但是因为没有考虑变量的大小，所以可能会对模型产生一定影响。
	3> train_and_validation.py 模型训练和预测脚本
