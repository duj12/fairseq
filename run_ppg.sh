#!/bin/bash
. ./path.sh


stage=2

tsv_dir=/data/megastore/Projects/DuJing/data/ASR_8k
split=train   #生成kmeans聚类模型使用的数据
#这个模型是WenetSpeech_l数据集训练得到的中文hubert模型
ckpt_path=/data/megastore/Projects/DuJing/model/PPG/CHN_phoasr_offline.pt
layer=9
nshard=1
rank=1          #rank不等于nshard时，只运行一个线程，即第(rank+1)个job
feat_dir=$tsv_dir/ppg
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi
batch_size=16
lab_dir=$feat_dir
km_path=/data/megastore/Projects/DuJing/data/km_mdl/ppg_iter1/km.mdl
n_cluster=500   #聚类数量，参考hubert

ppg_model_path=/data/megastore/Projects/DuJing/model/PPG
fairseq_path=/data/megastore/Projects/DuJing/code/fairseq
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
if [ $stage -eq -2 ]; then
for  split in valid train; do
	for((i=1;i<=$job;i++)); do
		cat $split_dir/$split.$i.nsample  
	done > $tsv_dir/$split.nsample
	echo "/" > $tsv_dir/$split.tsv
	paste $tsv_dir/$split.list $tsv_dir/$split.nsample >> $tsv_dir/$split.tsv
done
fi

# 不用上面两步，直接用fairseq自带的准备脚本，prepare tsv file
if [ $stage -eq -1 ] ; then 
python -u examples/wav2vec/wav2vec_manifest.py  \
	/data/megastore/Datasets/ASR 
	--dest /data/megastore/Projects/DuJing/data/ASR_8k 
	--ext wav --valid-percent 0.001 #> prep_voice_mega_tsv.log 2>&1 
fi

#prepare ppg feature，，，只需要提取少量的特征，用于得到一个km模型，不需要对全部训练音频都提取特征
if [ $stage -eq 0 ]; then
#we need to split the train.tsv into $nshard, named as train.$i.tsv
scp="" ;for((i=0;i<$nshard;i++)); do scp=$scp" "$tsv_dir/train.$i.tsv ; done
split_scp.pl $tsv_dir/train.tsv $scp
cd $ppg_model_path
for split in valid train; do
for((i=0;i<$nshard;i++)); do
CUDA_VISIBLE_DEVICES=$i  \
	python -u batch_ppg.py \
		$tsv_dir  $split.$i  $ckpt_path  $layer   $nshard   $i  $feat_dir $batch_size &# || exit 1;
done
done
for split in valid; do
CUDA_VISIBLE_DEVICES=0  \
	python -u batch_ppg.py \
		$tsv_dir  $split  $ckpt_path  $layer   1  0  $feat_dir $batch_size &# || exit 1;
done
cd $fairseq_path

fi

#等特征提取完毕，进行kmeans聚类，得到聚类参数模型, 这里只用train数据集聚类得到模型
if [ $stage -eq 1 ]; then
for ((i=0;i<$nshard;i++)); do 
	ln -s ${feat_dir}/train.${i}_${i}_$nshard.len ${feat_dir}/train_${i}_$nshard.len; 
	ln -s ${feat_dir}/train.${i}_${i}_$nshard.npy ${feat_dir}/train_${i}_$nshard.npy; 
done

	python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
		${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 0.1 
fi

#kmeans模型得到之后，对生成特征进行聚类，帧级别的标签
if [ $stage -eq -10000 ] ; then
	# for split in valid ; do
		# python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			# ${feat_dir} ${split} ${km_path} 1 ${rank} ${lab_dir}
		
		# #合并聚类标签
		# cat $lab_dir/${split}_${rank}_1.km > $lab_dir/${split}.km
	# done
	for split in train; do
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
cd $ppg_model_path
	for split in valid; do
		CUDA_VISIBLE_DEVICES=3  python -u ./forward_ppg_dump_km_label.py \
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
cd $fairseq_path
fi


#准备好了输入文件和列表，同时也生成了特征和聚类标签，可以进行训练了, 这里使用的是hubert模型提取的特征
if [ $stage -eq 3 ]; then 
CUDA_VISIBLE_DEVICES="6 "  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/pretrain \
		--config-name hubert_base_xmov \
		hydra.run.dir=outputs/ASR_8k-hubert-base \
		task.data=$tsv_dir \
		task.label_dir=$lab_dir \
		task.labels=["km"] \
		mdoel.label_rate=50
fi


#使用ppg模型生成的特征标签训练,PPG特征帧率是16k/256=62.5
if [ $stage -eq 4 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/ASR_8k/
feat_dir=/data/megastore/Projects/DuJing/data/ASR_8k/ppg
lab_dir=$feat_dir
CUDA_VISIBLE_DEVICES="5 "  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/pretrain \
		--config-name hubert_base_xmov \
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
		--config-name hubert_base_xmov \
		hydra.run.dir=outputs/ASR_8k-ppg+hubert-base \
		task.data=$tsv_dir \
		task.labels=["km", "km"] \
		+task.label_dir_list=['/data/megastore/Projects/DuJing/data/ASR_8k/ppg', '/data/megastore/Projects/DuJing/data/ASR_8k/hubert'] \
		+task.label_rate_list=["62.5", "50"] \
		+task.use_multiple_frame_rate_label=true
fi
