#!/bin/bash
. ./path.sh


stage=$1


tsv_dir=/data/megastore/Projects/DuJing/data/aishell
split=train   #生成kmeans聚类模型使用的数据
down_sample_rate=200    #降采样后跟模型的帧率保持一致
nshard=20
rank=0
lab_dir=$tsv_dir

# prepare tsv file
if [ $stage -eq -2 ] ; then 
python -u examples/wav2vec/wav2vec_manifest.py  \
	/data/megastore/Datasets/ASR \
	--dest $tsv_dir  \
	--ext TextGrid --valid-percent 0.001 #> prep_voice_mega_tsv.log 2>&1 
fi


##转换textgrid为音素标签
if [ $stage -eq -1 ] ; then
frame_wise=0
label=ltr
	for split in  valid  test  train ; do
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

#训练无预训练的音素识别，直接使用和预训练时一样的backbone, 使用的是CE-loss, 而不是CTC
if [ $stage -eq 1 ]; then 
CUDA_VISIBLE_DEVICES="3 "  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/finetune \
		--config-name hubert_base_xmov_pho \
		hydra.run.dir=outputs/aishell-hubert-base-ph2 \
		task.data=$tsv_dir \
		task.label_dir=$lab_dir \
		task.labels=["ph"] \
		+task.label_rate=80.0 \
		model.w2v_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_best.pt  
fi

# 推理测试
if [ $stage -eq 2 ]; then 
CUDA_VISIBLE_DEVICES="6 "  \
python -u examples/speech_recognition/new/infer.py \
  --config-dir examples/hubert/config/decode \
  --config-name infer_argmax \
  hydra.run.dir=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph/decode \
  task.data=/data/megastore/Projects/DuJing/data/aishell \
  +task.labels=["ph"] \
  +task.label_rate=80.0 \
  task.normalize=false \
  decoding.results_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph/decode \
  common_eval.path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph/checkpoints/checkpoint_best.pt \
  dataset.gen_subset=test  
  
fi


#使用预训练过的网络进行音素识别finetune
if [ $stage -eq 3 ]; then 
CUDA_VISIBLE_DEVICES="4 "  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/finetune \
		--config-name hubert_base_xmov_pho_ft \
		hydra.run.dir=outputs/aishell-hubert-base-ph-ft2 \
		task.data=$tsv_dir \
		task.label_dir=$lab_dir \
		task.labels=["ph"] \
		+task.label_rate=80.0 \
		model.w2v_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_best.pt
fi

# 推理测试
if [ $stage -eq 4 ]; then 
CUDA_VISIBLE_DEVICES="6 "  \
python -u examples/speech_recognition/new/infer.py \
  --config-dir examples/hubert/config/decode \
  --config-name infer_argmax \
  hydra.run.dir=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph-ft/decode \
  task.data=/data/megastore/Projects/DuJing/data/aishell \
  +task.labels=["ph"] \
  +task.label_rate=80.0 \
  task.normalize=false \
  decoding.results_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph-ft/decode \
  common_eval.path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph-ft/checkpoints/checkpoint_best.pt \
  dataset.gen_subset=test 
  
fi



#训练无预训练的音素识别，直接使用和预训练时一样的backbone, 使用CTC-loss
if [ $stage -eq 5 ]; then 
CUDA_VISIBLE_DEVICES="1 "  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/finetune \
		--config-name hubert_base_xmov_pho_ctc \
		hydra.run.dir=outputs/aishell-hubert-base-ph-ctc \
		task.data=$tsv_dir \
		task.label_dir=$lab_dir \
		task.labels=["ltr"] \
		model.w2v_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_best.pt  
fi

#训练有预训练的音素识别，直接使用和预训练时一样的backbone, 使用CTC-loss
if [ $stage -eq 6 ]; then 
CUDA_VISIBLE_DEVICES="2 "  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/finetune \
		--config-name hubert_base_xmov_pho_ctc_ft \
		hydra.run.dir=outputs/aishell-hubert-base-ph-ctc-ft \
		task.data=$tsv_dir \
		task.label_dir=$lab_dir \
		task.labels=["ltr"] \
		model.w2v_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_best.pt  
fi

# 推理测试，帧级别预测
if [ $stage -eq 7 ]; then 
CUDA_VISIBLE_DEVICES="7 "  \
python -u examples/speech_recognition/new/infer.py \
  --config-dir examples/hubert/config/decode \
  --config-name infer_argmax \
  hydra.run.dir=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph-ctc-ft/decode \
  task.data=/data/megastore/Projects/DuJing/data/aishell \
  +task.labels=["ph"] \
  +task.label_rate=80.0 \
  task.normalize=false \
  decoding.results_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph-ctc-ft/decode \
  common_eval.path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph-ctc-ft/checkpoints/checkpoint_best.pt \
  dataset.gen_subset=test 
  
fi

# 推理测试,序列维特比解码
if [ $stage -eq 8 ]; then 
CUDA_VISIBLE_DEVICES="6 "  \
python -u examples/speech_recognition/new/infer.py \
  --config-dir examples/hubert/config/decode \
  --config-name infer_viterbi \
  hydra.run.dir=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph-ctc-ft/decode_viterbi \
  task.data=/data/megastore/Projects/DuJing/data/aishell \
  +task.labels=["ltr"] \
  +task.label_rate=-1 \
  task.normalize=false \
  decoding.results_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph-ctc-ft/decode_viterbi \
  common_eval.path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph-ctc-ft/checkpoints/checkpoint_best.pt \
  dataset.gen_subset=test 
  
fi


#使用预训练过的网络进行音素识别finetune
if [ $stage -eq 9 ]; then 
tsv_dir=/data/megastore/Projects/DuJing/data/aishell
lab_dir=$tsv_dir
CUDA_VISIBLE_DEVICES="7 "  \
	python -u fairseq_cli/hydra_train.py \
		--config-dir examples/hubert/config/finetune \
		--config-name hubert_base_xmov_pho_ft \
		hydra.run.dir=outputs/aishell-hubert-base-phn-ft \
		task.data=$tsv_dir \
		task.label_dir=$lab_dir \
		task.labels=["phn"] \
		+task.label_rate=80.0 \
		model.w2v_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_best.pt
fi

#prepare hubert feature, then dump labels, then mapping labels to phonemes
if [ $stage -eq 10 ]; then
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_10h
split=train   #生成kmeans聚类模型使用的数据
#这个模型是WenetSpeech_l数据集训练得到的中文hubert模型
ckpt_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base/checkpoints/checkpoint_best.pt

for l in $(seq 6 12); do
layer=$l   
nshard=1
rank=0   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/hubert-cfm_$layer
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi
for cluster in 2 500;do 
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
#先用hubert-conformer模型提取特征
#for split in train valid; do
#CUDA_VISIBLE_DEVICES=6 \
#	python -u ./examples/hubert/simple_kmeans/dump_hubert_feature.py \
#		$tsv_dir  $split  $ckpt_path  $layer   $nshard   $rank  $feat_dir  # || exit 1;
#done
#
# split=train
# CUDA_VISIBLE_DEVICES=6 \
	# python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
		# ${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 1.0
	# #生成聚类标签词典
	# for x in $(seq 0 $((n_cluster - 1))); do
		# echo "$x 1"
	# done > $lab_dir/dict.km.txt  
	
#提取标签  #提取特征的情况下，直接dump_km_label，否则使用forward_hubert_dump_km_label
	for split in test_100 ; do
		CUDA_VISIBLE_DEVICES=6  python -u examples/hubert/simple_kmeans/forward_hubert_dump_km_label.py \
			${ckpt_path} ${layer} ${tsv_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
	# for split in test_100; do
		# CUDA_VISIBLE_DEVICES=6  python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			# ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for i in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${i}_${nshard}.km
		done > $lab_dir/${split}.km
		
			#最后评估标签和音素映射的吻合程度
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


#prepare ppg feature, then dump labels, then mapping labels to phonemes
if [ $stage -eq 11 ]; then
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_10h
split=train   #生成kmeans聚类模型使用的数据
ckpt_path=/data/megastore/Projects/DuJing/model/PPG/CHN_phoasr_offline.pt

for l in $(seq 6 12); do
layer=$l   
nshard=1
rank=0   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/ppg_$layer
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi
for cluster in 2 500;do 
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
split=train
#先用hubert-conformer模型提取特征
ppg_model_path=/data/megastore/Projects/DuJing/model/PPG
fairseq_path=/data/megastore/Projects/DuJing/code/fairseq
# cd $ppg_model_path
# for split in train valid; do
# CUDA_VISIBLE_DEVICES=0 \
	# python -u forward_ppg.py \
		# $ckpt_path  $layer  $tsv_dir  $split   $nshard   $rank  $feat_dir  || exit 1;
# done
# #hubert-conformer聚类
# cd $fairseq_path

#CUDA_VISIBLE_DEVICES=0 \
#	python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
#		${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 1.0

	# #生成聚类标签词典
	# for x in $(seq 0 $((n_cluster - 1))); do
		# echo "$x 1"
	# done > $lab_dir/dict.km.txt  

#提取标签 #提取特征的情况下，直接dump_km_label，否则使用forward_hubert_dump_km_label
cd $ppg_model_path
for split in test_100 ; do
		CUDA_VISIBLE_DEVICES=0  python -u ./forward_ppg_dump_km_label.py \
			${ckpt_path} ${layer} ${tsv_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
	# for split in test_100; do
		# CUDA_VISIBLE_DEVICES=0  python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			# ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for i in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${i}_${nshard}.km
		done > $lab_dir/${split}.km
	
	cd $fairseq_path
#最后评估标签和音素映射的吻合程度
	CUDA_VISIBLE_DEVICES=0 \
		python -u ./examples/hubert/measure_teacher_quality.py \
		$tsv_dir   $lab_dir  km  \
		--lab_sets $split  \
		--phn_dir  $tsv_dir  \
		--phn_name  $phn_name  \
		--phn_sets  $split  \
		--upsample  1  > $lab_dir/${split}_${layer}.result
		
done

done
done
fi

#prepare hubert feature, then dump labels, then mapping labels to phonemes
if [ $stage -eq 12 ]; then
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_10h
split=train   #生成kmeans聚类模型使用的数据
#这个模型是WenetSpeech_l数据集训练得到的中文hubert模型
ckpt_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-ph-ft2/checkpoints/checkpoint_best.pt

for l in $(seq 6 12); do
layer=$l   
nshard=1
rank=0   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/hubert-cfm_phft_$layer
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi
for cluster in 2 500;do 
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
#先用hubert-conformer模型提取特征
#for split in train valid; do
#CUDA_VISIBLE_DEVICES=6 \
#	python -u ./examples/hubert/simple_kmeans/dump_hubert_feature.py \
#		$tsv_dir  $split  $ckpt_path  $layer   $nshard   $rank  $feat_dir  # || exit 1;
#done
#
# split=train
# CUDA_VISIBLE_DEVICES=6 \
	# python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
		# ${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 1.0
	# #生成聚类标签词典
	# for x in $(seq 0 $((n_cluster - 1))); do
		# echo "$x 1"
	# done > $lab_dir/dict.km.txt  
	
#提取标签  #提取特征的情况下，直接dump_km_label，否则使用forward_hubert_dump_km_label
	for split in test_100 ; do
		CUDA_VISIBLE_DEVICES=6  python -u examples/hubert/simple_kmeans/forward_hubert_dump_km_label.py \
			${ckpt_path} ${layer} ${tsv_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
	# for split in test_100; do
		# CUDA_VISIBLE_DEVICES=6  python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			# ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for i in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${i}_${nshard}.km
		done > $lab_dir/${split}.km
		
			#最后评估标签和音素映射的吻合程度
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

#prepare hubert feature, then dump labels, then mapping labels to phonemes
if [ $stage -eq 13 ]; then
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_10h
split=train   #生成kmeans聚类模型使用的数据
#这个模型是WenetSpeech_l数据集训练得到的中文hubert模型
ckpt_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/aishell-hubert-base-phn-ft/checkpoints/checkpoint_best.pt

for l in $(seq 6 12); do
layer=$l   
nshard=1
rank=0   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/hubert-cfm_phnft_$layer
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi
for cluster in 2 500;do 
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
#先用hubert-conformer模型提取特征
#for split in train valid; do
#CUDA_VISIBLE_DEVICES=6 \
#	python -u ./examples/hubert/simple_kmeans/dump_hubert_feature.py \
#		$tsv_dir  $split  $ckpt_path  $layer   $nshard   $rank  $feat_dir  # || exit 1;
#done
#
# split=train
# CUDA_VISIBLE_DEVICES=6 \
	# python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
		# ${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 1.0
	# #生成聚类标签词典
	# for x in $(seq 0 $((n_cluster - 1))); do
		# echo "$x 1"
	# done > $lab_dir/dict.km.txt  
	
#提取标签  #提取特征的情况下，直接dump_km_label，否则使用forward_hubert_dump_km_label
	for split in test_100 ; do
		CUDA_VISIBLE_DEVICES=6  python -u examples/hubert/simple_kmeans/forward_hubert_dump_km_label.py \
			${ckpt_path} ${layer} ${tsv_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
	# for split in test_100; do
		# CUDA_VISIBLE_DEVICES=6  python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			# ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for i in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${i}_${nshard}.km
		done > $lab_dir/${split}.km
		
			#最后评估标签和音素映射的吻合程度
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


#prepare hubert feature, then dump labels, then mapping labels to phonemes
if [ $stage -eq 14 ]; then
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_10h
split=train   #生成kmeans聚类模型使用的数据
#这个模型是WenetSpeech_l数据集训练得到的中文hubert模型
ckpt_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base-phn-ft/checkpoints/checkpoint_best.pt

for l in $(seq 6 12); do
layer=$l   
nshard=1
rank=0   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/hubert-cfm_phnft1_$layer
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi
for cluster in 2 500;do 
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
#先用hubert-conformer模型提取特征
for split in train ; do
CUDA_VISIBLE_DEVICES=7 \
	python -u ./examples/hubert/simple_kmeans/dump_hubert_feature.py \
		$tsv_dir  $split  $ckpt_path  $layer   $nshard   $rank  $feat_dir  # || exit 1;
done

split=train
CUDA_VISIBLE_DEVICES=7 \
	python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
		${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 1.0
	#生成聚类标签词典
	for x in $(seq 0 $((n_cluster - 1))); do
		echo "$x 1"
	done > $lab_dir/dict.km.txt  
	
#提取标签  #提取特征的情况下，直接dump_km_label，否则使用forward_hubert_dump_km_label
	for split in test_100 ; do
		CUDA_VISIBLE_DEVICES=7  python -u examples/hubert/simple_kmeans/forward_hubert_dump_km_label.py \
			${ckpt_path} ${layer} ${tsv_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
	# for split in test_100; do
		# CUDA_VISIBLE_DEVICES=6  python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			# ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for i in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${i}_${nshard}.km
		done > $lab_dir/${split}.km
		
			#最后评估标签和音素映射的吻合程度
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


#prepare hubert feature, then dump labels, then mapping labels to phonemes
if [ $stage -eq 15 ]; then
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_10h
split=train   #生成kmeans聚类模型使用的数据
#这个模型是WenetSpeech_l数据集训练得到的中文hubert模型
ckpt_path=/data/megastore/Projects/DuJing/code/fairseq/outputs/ASR_8k-hubert-base-phn-ft-inter/checkpoints/checkpoint_last.pt

for l in $(seq 6 12); do
layer=$l   
nshard=1
rank=0   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程
feat_dir=$tsv_dir/hubert-cfm_phnft_inter1_$layer
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi

#先用hubert-conformer模型提取特征
for split in train ; do
CUDA_VISIBLE_DEVICES=1 \
	python -u ./examples/hubert/simple_kmeans/dump_hubert_feature.py \
		$tsv_dir  $split  $ckpt_path  $layer   $nshard   $rank  $feat_dir  # || exit 1;
done


for cluster in 2 500;do 
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

split=train
CUDA_VISIBLE_DEVICES=1 \
	python -u ./examples/hubert/simple_kmeans/learn_kmeans.py \
		${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 1.0
	#生成聚类标签词典
	for x in $(seq 0 $((n_cluster - 1))); do
		echo "$x 1"
	done > $lab_dir/dict.km.txt  
	
#提取标签  #提取特征的情况下，直接dump_km_label，否则使用forward_hubert_dump_km_label
	for split in test_100 valid ; do
		CUDA_VISIBLE_DEVICES=1  python -u examples/hubert/simple_kmeans/forward_hubert_dump_km_label.py \
			${ckpt_path} ${layer} ${tsv_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
	# for split in test_100; do
		# CUDA_VISIBLE_DEVICES=6  python -u ./examples/hubert/simple_kmeans/dump_km_label.py \
			# ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
		
		#合并聚类标签
		for i in $(seq 0 $(($nshard - 1))); do
			cat $lab_dir/${split}_${i}_${nshard}.km
		done > $lab_dir/${split}.km
		
			#最后评估标签和音素映射的吻合程度
	CUDA_VISIBLE_DEVICES=1 \
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


if [ $stage -eq 16 ]; then
tsv_dir=/data/megastore/Projects/DuJing/data/aishell_10h
split=train   #生成kmeans聚类模型使用的数据
for l in $(seq 6 12); do
layer=$l   
nshard=1
rank=0   #rank==nshard时表示多线程并行，rank<nshard表示只运行其中一个线程  hubert-cfm_phnft_inter1
for name in  hubert-cfm_phnft_inter1; do 
feat_dir=$tsv_dir/${name}_$layer
if [ ! -d  $feat_dir ]; then 
	mkdir -p $feat_dir
fi
for cluster in 2 ; do 
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
for split in test_100 valid ; do
			#最后评估标签和音素映射的吻合程度
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
done

fi


