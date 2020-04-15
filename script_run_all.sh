

############
# GNNs
############

#MLP
#MLP_GATED
#GCN
#GAT
#GraphSage
#GIN
#MoNet
#GatedGCN
#GatedGCN_E

##################
# IEMOCAP DATASET
##################


code=main_emotion_node_classification.py 
python $code --config 'configs/emotion_node_classification_MLP_IEMOCAP.json' &
wait
python $code --config 'configs/emotion_node_classification_MLP_GATED_IEMOCAP.json' &
wait
python $code --config 'configs/emotion_node_classification_GCN_IEMOCAP.json' &
wait
python $code --config 'configs/emotion_node_classification_GAT_IEMOCAP.json' &
wait
python $code --config 'configs/emotion_node_classification_GraphSage_IEMOCAP.json' &
wait
python $code --config 'configs/emotion_node_classification_GIN_IEMOCAP.json' &
wait
python $code --config 'configs/emotion_node_classification_MoNet_IEMOCAP.json' & 
wait
python $code --config 'configs/emotion_node_classification_GatedGCN_IEMOCAP.json' & 
wait
python $code --config 'configs/emotion_node_classification_GatedGCN_E_IEMOCAP.json' &

# tmux new -s emotion_gnn_all -d
# tmux send-keys "conda activate emotion_gnn" C-m
# tmux send-keys "
# python $code --config 'configs/emotion_node_classification_MLP_IEMOCAP.json' &
# wait" C-m
# tmux send-keys "
# python $code --config 'configs/emotion_node_classification_MLP_GATED_IEMOCAP.json' &
# wait" C-m
# tmux send-keys "
# python $code --config 'configs/emotion_node_classification_GCN_IEMOCAP.json' &
# wait" C-m
# tmux send-keys "
# python $code --config 'configs/emotion_node_classification_GAT_IEMOCAP.json' &
# wait" C-m
# tmux send-keys "
# python $code --config 'configs/emotion_node_classification_GraphSage_IEMOCAP.json' &
# wait" C-m
# tmux send-keys "
# python $code --config 'configs/emotion_node_classification_GIN_IEMOCAP.json' &
# wait" C-m
# tmux send-keys "
# python $code --config 'configs/emotion_node_classification_MoNet_IEMOCAP.json' & 
# wait" C-m
# tmux send-keys "
# python $code --config 'configs/emotion_node_classification_GatedGCN_IEMOCAP.json' & 
# wait" C-m
# tmux send-keys "
# python $code --config 'configs/emotion_node_classification_GatedGCN_E_IEMOCAP.json' &
# wait" C-m

# tmux send-keys "tmux kill-session -t emotion_gnn_all" C-m
