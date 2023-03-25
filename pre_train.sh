# declare -a arr=('masonscnn' 'lstm' 'textcnn' 'ag_fast_parapred' 'pipr' 'resppi' 'pesi')
declare -a arr=('pesi')

for model_name in "${arr[@]}"
do
    echo "python pre_train.py $model_name > ./logs/log_pretrain_$model_name 2>&1;"

    python pre_train.py $model_name > ./logs/log_pretrain_$model_name 2>&1;
done