import os 




for gnn in ['GCN']:
    for fet in ['efficientnet-b0']:
        for sp in [150,300]:
            print(f"{gnn}   {fet}   {sp}")
            os.system(f"python main.py  --cnn_model_name {fet} --gnn_model {gnn}  --use_saved_state yes   --superpixel_number {sp}  --train yes")
            os.system(f"python main.py  --cnn_model_name {fet} --gnn_model {gnn}  --use_saved_state yes   --superpixel_number {sp}  --train no")