#!/bin/bash
. ./path.sh


stage=$1

tsv_dir=/data/megastore/Projects/DuJing/data/ZhEn_25k
split=train   #生成kmeans聚类模型使用的数据
#这个模型是LibriSpeech数据集训练得到的英文hubert模型
ckpt_path=/data/megastore/Projects/DuJing/model/HuBERT/hubert_large_ll60k.pt
layer=18   #参考hubert论文
nshard=20
rank=20   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/hubert-ll-eng
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi
n_cluster=500  
lab_dir=$feat_dir/cluster_${n_cluster}
km_dir=${lab_dir}  
km_path=$km_dir/km.mdl
if [ ! -d  $km_dir ]; then 
	mkdir -p $km_dir
fi
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
if [ $stage -le 0 ]; then
for split in valid ; do
CUDA_VISIBLE_DEVICES=1 \
	python -u ./examples/hubert/simple_kmeans/dump_hubert_feature.py \
		$tsv_dir  $split  $ckpt_path  $layer   $nshard   $rank  $feat_dir  # || exit 1;
done
fi

#等特征提取完毕，进行kmeans聚类，得到聚类参数模型, 这里只用valid数据集聚类得到模型
if [ $stage -le 1 ]; then
split=valid
	python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
		${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 1 
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
if [ $stage -le 2 ] ; then
	for split in  valid train; do
	CUDA_VISIBLE_DEVICES=1 \
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
		hydra.run.dir=outputs/ZhEn_25k-hubert-base \
		task.data=$tsv_dir \
		task.label_dir=$lab_dir \
		task.labels=["km"] \
		model.label_rate=50  
fi

#使用ppg模型生成的特征标签训练,PPG特征帧率是16k/256=62.5
if [ $stage -eq 4 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/ZhEn_25k/
feat_dir=/data/megastore/Projects/DuJing/data/ZhEn_25k/ppg
lab_dir=$feat_dir
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/pretrain \
		--config-name hubert_base_xmov_8gpu \
		hydra.run.dir=outputs/ZhEn_25k-ppg-base \
		task.data=$tsv_dir \
		task.label_dir=$lab_dir \
		task.labels=["km"] \
		model.label_rate=62.5
fi

#使用ppg和hubert模型提取特征聚类得到的标签进行训练
if [ $stage -eq 5 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/ZhEn_25k
feat_dir=
lab_dir=
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/pretrain \
		--config-name hubert_base_xmov_8gpu_2lbl \
		hydra.run.dir=outputs/ZhEn_25k-ppg+hubert-base \
		task.data=$tsv_dir \
		task.labels=[km,km] \
		+task.label_dir_list=['/data/megastore/Projects/DuJing/data/ZhEn_25k/ppg','/data/megastore/Projects/DuJing/data/ZhEn_25k/hubert'] \
		+task.label_rate_list=[62.5,50] \
		+task.use_multiple_frame_rate_label=true 
		#checkpoint.finetune_from_model=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_best.pt \
		#checkpoint.load_state_dict_strict=false \
		#checkpoint.remove_state_dict_keys=[label_embs_concat,final_proj.weight,final_proj.bias]
		#使用中文预训练的参数初始化，发现容易发生梯度爆炸
#备注，双标签的情况下,使用单标签的预训练模型参数，需要移除label_embs_concat和final_proj
#checkpoint.remove_state_dict_keys=[label_embs_concat,final_proj.weight,final_proj.bias]
#checkpoint.load_state_dict_strict=false  #设定load预训练模型参数时不需要严格对应参数名称
#model.untie_final_proj=false  #多个标签的词表embedding和最终线性层都进行共享,这样性能会下降
fi


#使用ppg和hubert（中文和英文）模型提取特征聚类得到的标签进行训练：三标签模型
if [ $stage -eq 6 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/ZhEn_25k
feat_dir=
lab_dir=
CUDA_VISIBLE_DEVICES="0,1,2,3"  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/pretrain \
		--config-name hubert_base_xmov_8gpu_3lbl \
		hydra.run.dir=outputs/ZhEn_25k-ppg+hubert+ll_eng-base \
		task.data=$tsv_dir \
		task.labels=[km,km,km] \
		+task.label_dir_list=['/data/megastore/Projects/DuJing/data/ZhEn_25k/ppg','/data/megastore/Projects/DuJing/data/ZhEn_25k/hubert','/data/megastore/Projects/DuJing/data/ZhEn_25k/hubert-ll-eng/cluster_500'] \
		+task.label_rate_list=[62.5,50,50] \
		+task.use_multiple_frame_rate_label=true 


fi


#使用ppg和hubert（中文和英文）模型提取特征聚类得到的标签进行训练：三标签模型
if [ $stage -eq 7 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/ZhEn_25k
feat_dir=
lab_dir=
CUDA_VISIBLE_DEVICES="4,5,6,7"  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/pretrain \
		--config-name hubert_base_xmov_8gpu_4lbl_fp32 \
		hydra.run.dir=outputs/ZhEn_25k-ppg+hubert+ll+ls-base-fp32 \
		task.data=$tsv_dir \
		task.labels=[km,km,km,km] \
		+task.label_dir_list=['/data/megastore/Projects/DuJing/data/ZhEn_25k/ppg','/data/megastore/Projects/DuJing/data/ZhEn_25k/hubert','/data/megastore/Projects/DuJing/data/ZhEn_25k/hubert-ll-eng/cluster_500','/data/megastore/Projects/DuJing/data/ZhEn_25k/hubert-ls-eng/cluster_500'] \
		+task.label_rate_list=[62.5,50,50,50] \
		+task.use_multiple_frame_rate_label=true 


fi