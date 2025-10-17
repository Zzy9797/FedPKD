dataset="cifar100"
backbone='simplecnn'
type='mode'
skew_class=20
iternum=400
beta=0.1
alpha3=0.5
ratio=1.0
attack_type='gauss'
attack_ratio=0.0
seed=3



python train.py --gpu "0" --dataset $dataset --backbone $backbone --type $type --skew_class $skew_class --num_local_iterations $iternum --beta $beta --alpha3 $alpha3 --ratio $ratio --attack_type $attack_type --attack_ratio $attack_ratio --init_seed $seed   
