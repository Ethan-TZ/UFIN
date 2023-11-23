for da in 'Movies_and_TV' 'Grocery_and_Gourmet_Food' 'Electronics' 'Books' 'Musical_Instruments' 'Office_Products' 'Toys_and_Games'; do
    python -u train.py --config_files="$da"_EulerNet.yaml
done

