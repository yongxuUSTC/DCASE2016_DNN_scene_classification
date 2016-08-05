use strict;

my $i;
my $j;
my $line;
my $curacc;
my $preacc;
#my $threshold=0.1;


my $numlayers=4;

	my $lrate=0.005; ####dropout的学习速率，大10倍
	#my $layersizes = "468,2048,2048,39,2048,183"; # (39)*11+(39)
	my $layersizes = "440,500,500,15"; # (25)*11
#	for(my $i=0;$i<$numlayers -2;$i++)
#	{
#		$layersizes	  .= ",2048";
#	}	
#	$layersizes	  .= ",183";#
	
	my $node=500;
	
#	my $hidname = "";
#	for(my $i=0;$i<$numlayers -2;$i++)
#	{
#		$hidname	  .= "_h500";
#	}	

	my $exe 						= "./code_BP_GPU_Dropout_ReLU_GPU2_multiClassRec_149900/BPtrain";
	my $gpu_used				= 1;
#	my $numlayers				= 4;
#	my $layersizes			= "429,1024,1024,183";

	my $bunchsize				= 100;#128
	my $momentum				= 0.9;
	my $weightcost			= 0;
	my $fea_dim					= 40;#(257+39)
	my $fea_context			= 11;
	my $traincache			= 149900;  ############ how many samples per chunk #102400
	my $init_randem_seed= 27863875;   ############ every epoch must change
	my $targ_offset			= 5;
	my $dpflag=1;
	my $v_omit=0.1;
	my $h_omit=0.3;
	
	my $CF_DIR					= "/user/HS103/yx0001/yongxu/sc_dnn/lib";
	my $norm_file				= "/user/HS103/yx0001/yongxu/sc_dnn/fea/dcase2016_FBankNoCMN_1ch_TrainRand_and_evaluateRand_40d.fea_norm";
	my $fea_file				= "/vol/vssp/datasets/audio/dcase2016/pfile/dcase2016_FBankNoCMN_1ch_TrainRand_and_evaluateRand_40d.pfile";
	my $targ_file				= "/user/HS103/yx0001/yongxu/sc_dnn/get_label/lab_dcase2016_fold1_final.pfile";#########
		
	my $train_sent_range		= "0-879";#截取200H，共165720句，约265H
	#my $cv_sent_range				= "880-1169";#截取最后面的1000句作为CV集
	my $cv_sent_range				= "600-879";#截取最后面的1000句作为CV集
	
	my $MLP_DIR					= "models/resubmission_fold1_40d_dcase2016_scTask1_random_batch$bunchsize\_momentum$momentum\_frContext$fea_context\_lrate$lrate\_node$node\_numlayer$numlayers\-Rand_440_2h500_15-ReLU-DropoutV0.1H0.3-GPU1";###########################################################################
	
	system("mkdir $MLP_DIR");
	my $outwts_file			= "$MLP_DIR/mlp.1.wts";
	my $log_file				= "$MLP_DIR/mlp.1.log";
	my $initwts_file		= "$CF_DIR/Rand_440_2h500_15.belta0.5.wts";###########
	#my $initwts_file		= "$CF_DIR/mlp.23.wts_3HLpretrCV93_Random15out";
	
	#printf("2");
	print "iter 1 lrate is $lrate\n"; 
	system("$exe" .
		" gpu_used=$gpu_used".
		" numlayers=$numlayers".
		" layersizes=$layersizes".
		" bunchsize=$bunchsize".
		" momentum=$momentum".
		" weightcost=$weightcost".
		" lrate=$lrate".
		" fea_dim=$fea_dim".
		" fea_context=$fea_context".
		" traincache=$traincache".
		" init_randem_seed=$init_randem_seed".
		" targ_offset=$targ_offset".
		" initwts_file=$initwts_file".
		" norm_file=$norm_file".
		" fea_file=$fea_file".
		" targ_file=$targ_file".
		" outwts_file=$outwts_file".
		" log_file=$log_file".
		" train_sent_range=$train_sent_range".
		" cv_sent_range=$cv_sent_range".
	  " dropoutflag=$dpflag".
		" visible_omit=$v_omit".
		" hid_omit=$h_omit"
		);
#		
##	die;
##	
##		my $success=open LOG, "$log_file";
##		if(!$success)
##		{
##			printf "open log fail\n";
##		}
##		while(<LOG>)
##	  {
##	  	chomp;
##	  	if(/CV over.*/)
##	  	{
##	  	  s/CV over\. right num: \d+, ACC: //; 
##	  	  s/%//; 
##	  	  $curacc=$_;
##	  	}	  	
##	  }
##	  close LOG;
##	  
#  $preacc=$curacc;
#	my $destep=0;
#	########################################
##	$init_randem_seed=27865600;
##	$momentum=0.7;
#	########################################
	for($i= 2;$i <= 10;$i++){

		$j = $i -1;
		$initwts_file		= "$MLP_DIR/mlp.$j.wts";
		$outwts_file		= "$MLP_DIR/mlp.$i.wts";
		$log_file				= "$MLP_DIR/mlp.$i.log";
		$init_randem_seed  += 345;
		#$momentum=$momentum+0.04;
    #
    print "iter $i lrate is $lrate\n"; 
		system("$exe" .
		" gpu_used=$gpu_used".
		" numlayers=$numlayers".
		" layersizes=$layersizes".
		" bunchsize=$bunchsize".
		" momentum=$momentum".
		" weightcost=$weightcost".
		" lrate=$lrate".
		" fea_dim=$fea_dim".
		" fea_context=$fea_context".
		" traincache=$traincache".
		" init_randem_seed=$init_randem_seed".
		" targ_offset=$targ_offset".
		" initwts_file=$initwts_file".
		" norm_file=$norm_file".
		" fea_file=$fea_file".
		" targ_file=$targ_file".
		" outwts_file=$outwts_file".
		" log_file=$log_file".
		" train_sent_range=$train_sent_range".
		" cv_sent_range=$cv_sent_range".
		" dropoutflag=$dpflag".
		" visible_omit=$v_omit".
		" hid_omit=$h_omit"
		);
	}
		
#		my $success=open LOG, "$log_file";
#		if(!$success)
#		{
#			printf "open log fail\n";
#		}
#		while(<LOG>)
#	  {
#	  	chomp;
#	  	if(/CV over.*/)
#	  	{
#	  	  s/CV over\. right num: \d+, ACC: //; 
#	  	  s/%//; 
#	  	  $curacc=$_;
#	  	}	  	
#	  }
#	  close LOG;
#
#	  if($curacc<$preacc+$threshold)	
#	  {
#	  	print "iter $i ACC $curacc < iter $j ACC $preacc+threshold($threshold)\n";
#	  	$destep++;
#	  	print "destep is $destep\n";
#	  	if($destep>=3)
#	  	{
#	  		
#	  		unlink($outwts_file) or warn "can not delete weights file";
#	  		unlink($log_file) or warn "can not delete log file";
#	  		$i+100;
#	  		#print "finetune end\n";
#	  		last;
#	  	}
#	  	else
#	  	{
#	  	$i--;	  	
#	  	$lrate *=0.5;
# 	    }
#	  }
#	  else
#	  {
#	  	$destep=0;
#	  	$preacc=$curacc;
#	  	print "1\n\n\n\n\n\n\n\n";
#	  }
#
#	}
#	
#	########################################
#	$init_randem_seed=27872155;
#	$momentum=0.9;
#	$lrate=0.002059;
#	########################################
	for($i= 11;$i <= 100;$i++){
		$j = $i -1;
		$initwts_file		= "$MLP_DIR/mlp.$j.wts";
		$outwts_file		= "$MLP_DIR/mlp.$i.wts";
		$log_file				= "$MLP_DIR/mlp.$i.log";
		#$lrate *= 0.9;
		$momentum=0.9;
		$init_randem_seed  += 345;
		print "iter $i lrate is $lrate\n"; 
		system("$exe" .
		" gpu_used=$gpu_used".
		" numlayers=$numlayers".
		" layersizes=$layersizes".
		" bunchsize=$bunchsize".
		" momentum=$momentum".
		" weightcost=$weightcost".
		" lrate=$lrate".
		" fea_dim=$fea_dim".
		" fea_context=$fea_context".
		" traincache=$traincache".
		" init_randem_seed=$init_randem_seed".
		" targ_offset=$targ_offset".
		" initwts_file=$initwts_file".
		" norm_file=$norm_file".
		" fea_file=$fea_file".
		" targ_file=$targ_file".
		" outwts_file=$outwts_file".
		" log_file=$log_file".
		" train_sent_range=$train_sent_range".
		" cv_sent_range=$cv_sent_range".
		" dropoutflag=$dpflag".
		" visible_omit=$v_omit".
		" hid_omit=$h_omit"
		);
	}
