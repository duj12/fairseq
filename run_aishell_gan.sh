#!/bin/bash
. ./path.sh

FAIRSEQ_ROOT=/data/megastore/Projects/DuJing/code/fairseq
stage=$1
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_gan
# prepare tsv file
if [ $stage -eq -2 ] ; then 
python -u examples/wav2vec/wav2vec_manifest.py  \
	/data/megastore/Datasets/ASR \
	--dest $tsv_dir  \
	--ext TextGrid --valid-percent 0.001 #> prep_voice_mega_tsv.log 2>&1 
fi


##转换textgrid为音素标签
if [ $stage -eq -1 ] ; then
tsv_dir=/data/megastore/Projects/DuJing/data/ASR_8kft
down_sample_rate=200
frame_wise=0
nshard=10
lab_dir=$tsv_dir
label=ltr
	for split in  valid  train ; do
		python -u ./examples/hubert/simple_kmeans/dump_frame_level_phone_label.py \
			${tsv_dir} ${split} ${down_sample_rate} ${nshard} ${lab_dir} ${frame_wise} ${label} 
		
		#合并聚类标签
		for rank in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${rank}_${nshard}.${label}
		done > $lab_dir/${split}.${label}
	done
	#生成聚类标签词典
	cat $lab_dir/valid.${label} $lab_dir/train.${label} |    \
		awk '{for(i=1;i<=NF;i++){if($i in dict)dict[$i]+=1; else dict[$i]=1}} END{for(k in dict) print(k" "dict[k]) }' \
	 > $lab_dir/dict.${label}.txt  
fi

##对textgrid采样，得到帧级别的标签
if [ $stage -eq 0 ] ; then
	for split in valid  test  train ; do
		python -u ./examples/hubert/simple_kmeans/dump_frame_level_phone_label.py \
			${tsv_dir} ${split} ${down_sample_rate} ${nshard} ${lab_dir}
		
		#合并聚类标签
		for rank in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${rank}_${nshard}.ph
		done > $lab_dir/${split}.ph
	done
	#生成聚类标签词典
	cat $lab_dir/valid.ph $lab_dir/train.ph |    \
		awk '{for(i=1;i<=NF;i++){if($i in dict)dict[$i]+=1; else dict[$i]=1}} END{for(k in dict) print(k" "dict[k]) }' \
	 > $lab_dir/dict.ph.txt  
fi

#对音素文件进行预处理
if [ $stage -eq 1 ]; then
tsv_dir=/data/megastore/Projects/DuJing/data/ASR_8kgan
	python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap \
		--trainpref $tsv_dir/train.phn \
		--validpref $tsv_dir/valid.phn \
		--workers 1 --only-source \
		--destdir $tsv_dir/phones \
		--srcdict $tsv_dir/dict.phn.txt

fi

#准备特征文件
if [ $stage -eq 2 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_gan
split=train   #生成kmeans聚类模型使用的数据
#这个模型是hubert-conformer模型，使用8k小时中文语音数据预训练
ckpt_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_best.pt

layer=9   
nshard=1
rank=0   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/hubert-cfm_$layer
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi
lab_dir=$feat_dir
km_dir=${feat_dir}   #/data/megastore/Projects/DuJing/data/km_mdl/
km_path=$km_dir/km.mdl
if [ ! -d  $km_dir ]; then 
	mkdir -p $km_dir
fi
n_cluster=500
#先用hubert-conformer模型提取特征
for split in train valid; do
CUDA_VISIBLE_DEVICES=7 \
	python -u ./examples/hubert/simple_kmeans/dump_hubert_feature.py \
		$tsv_dir  $split  $ckpt_path  $layer   $nshard   $rank  $feat_dir  # || exit 1;
done
#生成聚类模型 
#split=train
#CUDA_VISIBLE_DEVICES=7 \
#	python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
#		${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 1.0

#提取标签  #提取特征的情况下，直接dump_km_label，否则使用forward_hubert_dump_km_label
#	for split in valid ; do
#		CUDA_VISIBLE_DEVICES=7  python -u examples/hubert/simple_kmeans/forward_hubert_dump_km_label.py \
#			${ckpt_path} ${layer} ${tsv_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
	for split in train valid; do
		CUDA_VISIBLE_DEVICES=7  python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for i in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${i}_${nshard}.km
		done > $lab_dir/${split}.km
		ln -s $lab_dir/${split}_${rank}_${nshard}.npy $lab_dir/${split}.npy
		ln -s $lab_dir/${split}_${rank}_${nshard}.len $lab_dir/${split}.len
	done
	
	#生成聚类标签词典
	for x in $(seq 0 $((n_cluster - 1))); do
		echo "$x 1"
	done > $lab_dir/dict.km.txt  



fi

#训练
if [ $stage -eq 3 ]; then
PREFIX=unsup_gan
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_gan
layer=9
feat_dir=$tsv_dir/hubert-cfm_$layer

# For wav2vec-U, audio features are pre-segmented
#CONFIG_NAME=w2vu
#TASK_DATA=/path/to/features/precompute_unfiltered_pca512_cls128_mean_pooled

# For wav2vec-U 2.0, use raw audio features
CONFIG_NAME=w2vu2
TASK_DATA=$feat_dir  #/path/to/features/

# Unpaired text input
TEXT_DATA=/data/megastore/Projects/DuJing/data/aishell_gan/phones   #/path/to/data/phones  # path to fairseq-preprocessed GAN data (phones dir)
KENLM_PATH=     #/path/to/data/phones/kenlm.phn.o4.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

CUDA_VISIBLE_DEVICES=7   \
python -u fairseq_cli/hydra_train.py \
    --config-dir examples/wav2vec/unsupervised/config/gan \
    --config-name $CONFIG_NAME \
    task.data=${TASK_DATA} \
    task.text_data=${TEXT_DATA} \
    hydra.run.dir=outputs/aishell-hubert-gan \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    model.code_penalty=2  \
    model.gradient_penalty=1.5 \
    model.smoothness_weight=0.5 \
    common.seed=0
fi

if [ $stage -eq 4 ]; then
CUDA_VISIBLE_DEVICES=6  \
python -u  examples/wav2vec/unsupervised/w2vu_generate.py \
	--config-dir examples/wav2vec/unsupervised/config/generate --config-name argmax \
	fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
	fairseq.task.data=/data/megastore/Projects/DuJing/data/aishell_gan/hubert-cfm_9 \
	fairseq.common_eval.path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-gan/checkpoints/checkpoint_best.pt \
	fairseq.dataset.gen_subset=valid \
	hydra.run.dir=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-gan/decode_argmax \
	results_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-gan/decode_argmax

fi


#对generator之前的特征进行聚类评估
if [ $stage -eq 5 ]; then
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_gan
split=train1w   #生成kmeans聚类模型使用的数据
#这个模型是WenetSpeech_l数据集训练得到的中文hubert模型
ckpt_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_best.pt

for l in $(seq 9 9); do
layer=$l   
nshard=1
rank=0   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/hubert-ori_${layer}   
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi

#先用hubert-conformer模型提取特征
# for split in train1w test_100; do
# CUDA_VISIBLE_DEVICES=6 \
	# python -u ./examples/hubert/simple_kmeans/dump_hubert_feature.py \
		# $tsv_dir  $split  $ckpt_path  $layer   $nshard   $rank  $feat_dir  # || exit 1;
# done
		# ln -s $feat_dir/${split}_${rank}_${nshard}.npy $feat_dir/${split}.npy
		# ln -s $feat_dir/${split}_${rank}_${nshard}.len $feat_dir/${split}.len

for cluster in 2 500; do 
if [ $cluster -eq 2 ]; then 
	phn_name=vad
else
	phn_name=phn
fi
n_cluster=$cluster  
lab_dir=$feat_dir/c${n_cluster}
km_dir=${lab_dir}   #/data/megastore/Projects/DuJing/data/km_mdl/
km_path=$km_dir/km.mdl
if [ ! -d  $km_dir ]; then 
	mkdir -p $km_dir
fi

split=train1w
CUDA_VISIBLE_DEVICES=6 \
	python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
		${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 1.0
	#生成聚类标签词典
	for x in $(seq 0 $((n_cluster - 1))); do
		echo "$x 1"
	done > $lab_dir/dict.km.txt  
	
#提取标签  #提取特征的情况下，直接dump_km_label，否则使用forward_hubert_dump_km_label
#	for split in test_100 ; do
#		CUDA_VISIBLE_DEVICES=6  python -u examples/hubert/simple_kmeans/forward_hubert_dump_km_label.py \
#			${ckpt_path} ${layer} ${tsv_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
for split in test_100; do
		CUDA_VISIBLE_DEVICES=6  python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for i in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${i}_${nshard}.km
		done > $lab_dir/${split}.km

		
#最后评估标签和音素映射的吻合程度
#split=test_100
	CUDA_VISIBLE_DEVICES=6 \
		python -u ./examples/hubert/measure_teacher_quality.py \
		$tsv_dir   $lab_dir  km  \
		--lab_sets $split  \
		--phn_name  $phn_name  \
		--phn_dir  $tsv_dir  \
		--phn_sets  $split  \
		--upsample  1  > $lab_dir/${split}_${layer}.result
		
done
done
done

fi


#提取generator特征并评估
if [ $stage -eq 6 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_gan
split=train1w   #生成kmeans聚类模型使用的数据
#这个模型是WenetSpeech_l数据集训练得到的中文hubert模型
ckpt_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_last.pt

for l in $(seq 9 9); do
layer=$l   
nshard=1
rank=0   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/hubert-gan1_${layer}   
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi

#先用hubert-conformer-gan模型提取特征
for split in train1w test_100; do
CUDA_VISIBLE_DEVICES=3  \
python -u  examples/wav2vec/unsupervised/w2vu_generate.py \
	--config-dir examples/wav2vec/unsupervised/config/generate --config-name argmax \
	fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
	fairseq.task.data=/data/megastore/Projects/DuJing/data/aishell_gan/hubert-ori_9 \
	fairseq.common_eval.path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-gan/checkpoints/checkpoint_best.pt \
	fairseq.dataset.gen_subset=$split \
	hydra.run.dir=$feat_dir \
	results_path=$feat_dir
	
	ln -s $feat_dir/${split}.npy $feat_dir/${split}_${rank}_${nshard}.npy
	ln -s $feat_dir/${split}.len $feat_dir/${split}_${rank}_${nshard}.len
done

for cluster in 2 500; do 
if [ $cluster -eq 2 ]; then 
	phn_name=vad
else
	phn_name=phn
fi
n_cluster=$cluster  
lab_dir=$feat_dir/c${n_cluster}
km_dir=${lab_dir}   #/data/megastore/Projects/DuJing/data/km_mdl/
km_path=$km_dir/km.mdl
if [ ! -d  $km_dir ]; then 
	mkdir -p $km_dir
fi

split=train1w
CUDA_VISIBLE_DEVICES=6 \
	python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
		${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 1.0
	#生成聚类标签词典
	for x in $(seq 0 $((n_cluster - 1))); do
		echo "$x 1"
	done > $lab_dir/dict.km.txt  
	
#提取标签  #提取特征的情况下，直接dump_km_label，否则使用forward_hubert_dump_km_label
#	for split in test_100 ; do
#		CUDA_VISIBLE_DEVICES=6  python -u examples/hubert/simple_kmeans/forward_hubert_dump_km_label.py \
#			${ckpt_path} ${layer} ${tsv_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
for split in test_100; do
		CUDA_VISIBLE_DEVICES=3  python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for i in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${i}_${nshard}.km
		done > $lab_dir/${split}.km

		
#最后评估标签和音素映射的吻合程度
#split=test_100
	CUDA_VISIBLE_DEVICES=3 \
		python -u ./examples/hubert/measure_teacher_quality.py \
		$tsv_dir   $lab_dir  km  \
		--lab_sets $split  \
		--phn_name  $phn_name  \
		--phn_dir  $tsv_dir  \
		--phn_sets  $split  \
		--upsample  1  > $lab_dir/${split}_${layer}.result
		
done
done
done
fi

#下面进行二分类的GAN训练
#对音素文件进行预处理, 先将音素处理成sil和spch
if [ $stage -eq 7 ]; then
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_gan
#for x in train ; do 
#	cat $tsv_dir/$x.phn | awk '{for(i=1;i<NF;i++){if($i=="sil"||$i=="sp")printf "sil "; else printf "spch ";}   if($NF=="sil"||$NF=="sp")print "sil"; else print "spch";}' > $.vad
#done

	python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap \
		--trainpref /data/megastore/Projects/DuJing/data/ASR_8kft/train.vad \
		--workers 1 --only-source \
		--destdir $tsv_dir/vads \
		--srcdict $tsv_dir/dict.vad.txt
#--validpref $tsv_dir/valid.vad 
fi

#训练
if [ $stage -eq 8 ]; then
PREFIX=unsup_gan
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_gan
layer=9
feat_dir=$tsv_dir/hubert-cfm_$layer

# For wav2vec-U, audio features are pre-segmented
#CONFIG_NAME=w2vu
#TASK_DATA=/path/to/features/precompute_unfiltered_pca512_cls128_mean_pooled

# For wav2vec-U 2.0, use raw audio features
CONFIG_NAME=w2vu2_vad
TASK_DATA=$feat_dir  #/path/to/features/

# Unpaired text input
TEXT_DATA=/data/megastore/Projects/DuJing/data/aishell_gan/vads   #/path/to/data/phones  # path to fairseq-preprocessed GAN data (phones dir)
KENLM_PATH=     #/path/to/data/phones/kenlm.phn.o4.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

CUDA_VISIBLE_DEVICES=6   \
python -u fairseq_cli/hydra_train.py \
    --config-dir examples/wav2vec/unsupervised/config/gan \
    --config-name $CONFIG_NAME \
    task.data=${TASK_DATA} \
    task.text_data=${TEXT_DATA} \
    hydra.run.dir=outputs/aishell-hubert-gan-vad \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    model.code_penalty=2  \
    model.gradient_penalty=1.5 \
    model.smoothness_weight=0.5 \
    common.seed=0
fi


#下面进行音素序列（非帧级别）的GAN训练
#对音素文件进行预处理, 先将音素处理成sil和spch
if [ $stage -eq 9 ]; then
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_gan
#for x in train ; do 
#	cat $tsv_dir/$x.phn | awk '{for(i=1;i<NF;i++){if($i=="sil"||$i=="sp")printf "sil "; else printf "spch ";}   if($NF=="sil"||$NF=="sp")print "sil"; else print "spch";}' > $.vad
#done

	python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap \
		--trainpref /data/megastore/Projects/DuJing/data/ASR_8kft/train.sym \
		--validpref $tsv_dir/valid.sym  \
		--workers 1 --only-source \
		--destdir $tsv_dir/syms \
		--srcdict $tsv_dir/dict.sym.txt
#--validpref $tsv_dir/valid.vad 
fi

#训练
if [ $stage -eq 10 ]; then
PREFIX=unsup_gan
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_gan
layer=9
feat_dir=$tsv_dir/hubert-cfm_$layer

# For wav2vec-U, audio features are pre-segmented
#CONFIG_NAME=w2vu
#TASK_DATA=/path/to/features/precompute_unfiltered_pca512_cls128_mean_pooled

# For wav2vec-U 2.0, use raw audio features
CONFIG_NAME=w2vu2_sym
TASK_DATA=$feat_dir  #/path/to/features/

# Unpaired text input
TEXT_DATA=/data/megastore/Projects/DuJing/data/aishell_gan/syms   #/path/to/data/phones  # path to fairseq-preprocessed GAN data (phones dir)
KENLM_PATH=     #/path/to/data/phones/kenlm.phn.o4.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

CUDA_VISIBLE_DEVICES=7  \
python -u fairseq_cli/hydra_train.py \
    --config-dir examples/wav2vec/unsupervised/config/gan \
    --config-name $CONFIG_NAME \
    task.data=${TASK_DATA} \
    task.text_data=${TEXT_DATA} \
    hydra.run.dir=outputs/aishell-hubert-gan-sym \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    model.code_penalty=2  \
    model.gradient_penalty=1.5 \
    model.smoothness_weight=0.5 \
    common.seed=0
fi

#提取generator特征并评估
if [ $stage -eq 11 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_gan
split=train1w   #生成kmeans聚类模型使用的数据
#这个模型是WenetSpeech_l数据集训练得到的中文hubert模型
ckpt_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_last.pt

for l in $(seq 9 9); do
layer=$l   
nshard=1
rank=0   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/hubert-gan-vad_${layer}   
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi

#先用hubert-conformer-gan模型提取特征
for split in train1w test_100; do
CUDA_VISIBLE_DEVICES=7  \
python -u  examples/wav2vec/unsupervised/w2vu_generate.py \
	--config-dir examples/wav2vec/unsupervised/config/generate --config-name argmax_vad \
	fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
	fairseq.task.data=/data/megastore/Projects/DuJing/data/aishell_gan/hubert-ori_9 \
	fairseq.common_eval.path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-gan-vad/checkpoints/checkpoint_best.pt \
	fairseq.dataset.gen_subset=$split \
	hydra.run.dir=$feat_dir \
	results_path=$feat_dir
	
	ln -s $feat_dir/${split}.npy $feat_dir/${split}_${rank}_${nshard}.npy
	ln -s $feat_dir/${split}.len $feat_dir/${split}_${rank}_${nshard}.len
done

for cluster in 2 500; do 
if [ $cluster -eq 2 ]; then 
	phn_name=vad
else
	phn_name=phn
fi
n_cluster=$cluster  
lab_dir=$feat_dir/c${n_cluster}
km_dir=${lab_dir}   #/data/megastore/Projects/DuJing/data/km_mdl/
km_path=$km_dir/km.mdl
if [ ! -d  $km_dir ]; then 
	mkdir -p $km_dir
fi

split=train1w
CUDA_VISIBLE_DEVICES=7 \
	python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
		${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 1.0
	#生成聚类标签词典
	for x in $(seq 0 $((n_cluster - 1))); do
		echo "$x 1"
	done > $lab_dir/dict.km.txt  
	
#提取标签  #提取特征的情况下，直接dump_km_label，否则使用forward_hubert_dump_km_label
#	for split in test_100 ; do
#		CUDA_VISIBLE_DEVICES=6  python -u examples/hubert/simple_kmeans/forward_hubert_dump_km_label.py \
#			${ckpt_path} ${layer} ${tsv_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
for split in test_100; do
		CUDA_VISIBLE_DEVICES=7 python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for i in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${i}_${nshard}.km
		done > $lab_dir/${split}.km

		
#最后评估标签和音素映射的吻合程度
#split=test_100
	CUDA_VISIBLE_DEVICES=7 \
		python -u ./examples/hubert/measure_teacher_quality.py \
		$tsv_dir   $lab_dir  km  \
		--lab_sets $split  \
		--phn_name  $phn_name  \
		--phn_dir  $tsv_dir  \
		--phn_sets  $split  \
		--upsample  1  > $lab_dir/${split}_${layer}.result
		
done
done
done
fi

#提取generator特征并评估
if [ $stage -eq 12 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_gan
split=train1w   #生成kmeans聚类模型使用的数据
#这个模型是WenetSpeech_l数据集训练得到的中文hubert模型
ckpt_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_last.pt

for l in $(seq 9 9); do
layer=$l   
nshard=1
rank=0   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/hubert-gan-sym_${layer}   
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi

#先用hubert-conformer-gan模型提取特征
for split in train1w test_100; do
CUDA_VISIBLE_DEVICES=6  \
python -u  examples/wav2vec/unsupervised/w2vu_generate.py \
	--config-dir examples/wav2vec/unsupervised/config/generate --config-name argmax \
	fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
	fairseq.task.data=/data/megastore/Projects/DuJing/data/aishell_gan/hubert-ori_9 \
	fairseq.common_eval.path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-gan-sym/checkpoints/checkpoint_best.pt \
	fairseq.dataset.gen_subset=$split \
	hydra.run.dir=$feat_dir \
	results_path=$feat_dir
	
	ln -s $feat_dir/${split}.npy $feat_dir/${split}_${rank}_${nshard}.npy
	ln -s $feat_dir/${split}.len $feat_dir/${split}_${rank}_${nshard}.len
done

for cluster in 2 500; do 
if [ $cluster -eq 2 ]; then 
	phn_name=vad
else
	phn_name=phn
fi
n_cluster=$cluster  
lab_dir=$feat_dir/c${n_cluster}
km_dir=${lab_dir}   #/data/megastore/Projects/DuJing/data/km_mdl/
km_path=$km_dir/km.mdl
if [ ! -d  $km_dir ]; then 
	mkdir -p $km_dir
fi

split=train1w
CUDA_VISIBLE_DEVICES=6 \
	python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
		${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 1.0
	#生成聚类标签词典
	for x in $(seq 0 $((n_cluster - 1))); do
		echo "$x 1"
	done > $lab_dir/dict.km.txt  
	
#提取标签  #提取特征的情况下，直接dump_km_label，否则使用forward_hubert_dump_km_label
#	for split in test_100 ; do
#		CUDA_VISIBLE_DEVICES=6  python -u examples/hubert/simple_kmeans/forward_hubert_dump_km_label.py \
#			${ckpt_path} ${layer} ${tsv_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
for split in test_100; do
		CUDA_VISIBLE_DEVICES=6  python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for i in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${i}_${nshard}.km
		done > $lab_dir/${split}.km

		
#最后评估标签和音素映射的吻合程度
#split=test_100
	CUDA_VISIBLE_DEVICES=6 \
		python -u ./examples/hubert/measure_teacher_quality.py \
		$tsv_dir   $lab_dir  km  \
		--lab_sets $split  \
		--phn_name  $phn_name  \
		--phn_dir  $tsv_dir  \
		--phn_sets  $split  \
		--upsample  1  > $lab_dir/${split}_${layer}.result
		
done
done
done
fi
