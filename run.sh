#!/bin/bash
. ./path.sh


stage=$1

tsv_dir=/data/megastore/Projects/DuJing/data/ASR_8k
split=train   #生成kmeans聚类模型使用的数据
#这个模型是WenetSpeech_l数据集训练得到的中文hubert模型
ckpt_path=/data/megastore/Projects/DuJing/model/Wenet-pretrain/hubert_base/chinese-hubert-base-fairseq-ckpt.pt
layer=9   #参考hubert论文
nshard=1
rank=1   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/hubert
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi
lab_dir=$feat_dir
km_path=/data/megastore/Projects/DuJing/data/km_mdl/hubert_iter1/km.mdl
n_cluster=500   #聚类数量，参考hubert

job=10
split_dir=$tsv_dir/"split"$job  #对列表文件进行分割，便于多job运行

# 先进行音频采样点数的计算
if [ $stage -eq -2 ] ; then 

for split in valid train; do 
	mkdir -p $split_dir
	split_scp=""
	for((i=1;i<=$job;i++)); do
		split_scp=$split_scp" "$split_dir"/"$split.$i
	done
	split_scp.pl $tsv_dir/$split.list $split_scp
	for((i=1;i<=$job;i++)); do
		cat $split_dir/$split.$i | awk '{print "soxi -s " $0 | " sh " }' > $split_dir/$split.$i.nsample &
	done
done
fi 


#等采样点数都计算完，再合并生成单个tsv文件，需要首先准备好list文件
if [ $stage -eq -1 ]; then
for  split in valid train; do
	for((i=1;i<=$job;i++)); do
		cat $split_dir/$split.$i.nsample  
	done > $tsv_dir/$split.nsample
	echo "/" > $tsv_dir/$split.tsv
	paste $tsv_dir/$split.list $tsv_dir/$split.nsample >> $tsv_dir/$split.tsv
done
fi

# prepare tsv file
if [ $stage -eq -1 ] ; then 
python -u examples/wav2vec/wav2vec_manifest.py  \
	/data/megastore/Datasets/ASR \
	--dest /data/megastore/Projects/DuJing/data/ASR_8k \
	--ext wav --valid-percent 0.001 #> prep_voice_mega_tsv.log 2>&1 
fi

#prepare hubert feature 注:无预训练的序列级别模型(ctc-loss)，lr=2e-4训练过程发生梯度爆炸，故采用2e-5学习率
if [ $stage -eq 0 ]; then
for split in valid train; do
CUDA_VISIBLE_DEVICES=6 \
	python -u ./examples/hubert/simple_kmeans/dump_hubert_feature.py \
		$tsv_dir  $split  $ckpt_path  $layer   $nshard   $rank  $feat_dir  # || exit 1;
done
fi

#等特征提取完毕，进行kmeans聚类，得到聚类参数模型, 这里只用5% train数据集聚类得到模型
if [ $stage -eq 1 ]; then
	python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
		${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 0.05 
fi

##kmeans模型得到之后，对生成特征进行聚类，帧级别的标签
if [ $stage -eq -100 ] ; then
#lab_dir=/data/megastore/Projects/DuJing/data/ASR_8k/hubert
	for split in valid train ; do
		python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for rank in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${rank}_${nshard}.km
		done > $lab_dir/${split}.km
	done
	#生成聚类标签词典
	for x in $(seq 0 $((n_cluster - 1))); do
		echo "$x 1"
	done > $lab_dir/dict.km.txt  
fi

#不需要保存特征，生成特征之后，直接进行聚类，得到标签：
if [ $stage -eq 2 ] ; then
	for split in  valid; do
		python -u examples/hubert/simple_kmeans/forward_hubert_dump_km_label.py \
			${ckpt_path} ${layer} ${tsv_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for i in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${i}_${nshard}.km
		done > $lab_dir/${split}.km
	done
	#生成聚类标签词典
	for x in $(seq 0 $((n_cluster - 1))); do
		echo "$x 1"
	done > $lab_dir/dict.km.txt  
fi

#准备好了输入文件和列表，同时也生成了特征和聚类标签，可以进行训练了, 这里使用的是hubert模型提取的特征
if [ $stage -eq 3 ]; then 
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7 "  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/pretrain \
		--config-name hubert_base_xmov_8gpu \
		hydra.run.dir=outputs/ASR_8k-hubert-base \
		task.data=$tsv_dir \
		task.label_dir=$lab_dir \
		task.labels=["km"] \
		model.label_rate=50  
fi

#使用ppg模型生成的特征标签训练,PPG特征帧率是16k/256=62.5
if [ $stage -eq 4 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/ASR_8k/
feat_dir=/data/megastore/Projects/DuJing/data/ASR_8k/ppg
lab_dir=$feat_dir
CUDA_VISIBLE_DEVICES="5 "  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/pretrain \
		--config-name hubert_base_xmov_8gpu \
		hydra.run.dir=outputs/ASR_8k-ppg-base \
		task.data=$tsv_dir \
		task.label_dir=$lab_dir \
		task.labels=["km"] \
		model.label_rate=62.5
fi

#使用ppg和hubert模型提取特征聚类得到的标签进行训练
if [ $stage -eq 5 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/ASR_8k/
feat_dir=
lab_dir=
CUDA_VISIBLE_DEVICES="5 "  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/pretrain \
		--config-name hubert_base_xmov_8gpu \
		hydra.run.dir=outputs/ASR_8k-ppg+hubert-base \
		task.data=$tsv_dir \
		task.labels=["km", "km"] \
		+task.label_dir_list=['/data/megastore/Projects/DuJing/data/ASR_8k/ppg', '/data/megastore/Projects/DuJing/data/ASR_8k/hubert'] \
		+task.label_rate_list=["62.5", "50"] \
		+task.use_multiple_frame_rate_label=true  \
		checkpoint.finetune_from_model=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_best.pt  \
		checkpoint.load_state_dict_strict=false 
fi


##下面对ASR_8k数据集，得到帧级别音素标签，进行finetune
##对textgrid采样，得到帧级别的标签
if [ $stage -eq 6 ] ; then
tsv_dir=/data/megastore/Projects/DuJing/data/ASR_8kft
split=train   #生成kmeans聚类模型使用的数据
down_sample_rate=200    #降采样后跟模型的帧率保持一致
nshard=20
rank=0
lab_dir=$tsv_dir
frame_wise=1
label=phn
	for split in   train ; do
		python -u ./examples/hubert/simple_kmeans/dump_frame_level_phone_label.py \
			${tsv_dir} ${split} ${down_sample_rate} ${nshard} ${lab_dir} ${frame_wise} ${label} 
		
		#合并聚类标签
		for rank in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${rank}_${nshard}.${label}
		done > $lab_dir/${split}.${label}
	done
	#生成聚类标签词典
	cat $lab_dir/valid.${label} $lab_dir/train.${label} \|\
		awk '{for(i=1;i<=NF;i++){if($i in dict)dict[$i]+=1; else dict[$i]=1}} END{for(k in dict) print(k" "dict[k]) }' \
	 > $lab_dir/dict.${label}.txt  
fi


#使用预训练过的网络进行音素识别finetune
if [ $stage -eq 7 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/ASR_8kft
lab_dir=$tsv_dir
CUDA_VISIBLE_DEVICES="7 "  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/finetune \
		--config-name hubert_base_xmov_pho_ft \
		hydra.run.dir=outputs/ASR_8k-hubert-base-phn-ft \
		task.data=$tsv_dir \
		task.label_dir=$lab_dir \
		task.labels=["phn"] \
		+task.label_rate=80.0 \
		model.w2v_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_best.pt
fi


#使用预训练过的网络进行音素识别finetune, 添加intermediate_layer_loss
if [ $stage -eq 8 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/ASR_8kft
lab_dir=$tsv_dir
CUDA_VISIBLE_DEVICES="7 "  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/finetune \
		--config-name hubert_base_xmov_pho_ft_inter \
		hydra.run.dir=outputs/ASR_8k-hubert-base-phn-ft-inter \
		task.data=$tsv_dir \
		task.label_dir=$lab_dir \
		task.labels=["phn"] \
		+task.label_rate=80.0 \
		model.w2v_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_best.pt
fi

