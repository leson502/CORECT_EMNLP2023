# !bin/bash
# train from begin
python train.py --dataset="iemocap" --modalities="atv" --from_begin --epochs=50 --learning_rate=0.00025 --optimizer="adam" --drop_rate=0.5 --batch_size=10 --rnn="transformer" --use_speaker  --edge_type="temp_multi" --wp=11 --wf=5  --gcn_conv="rgcn" --use_graph_transformer --graph_transformer_nheads=7  --use_crossmodal --num_crossmodal=2 --num_self_att=3 --crossmodal_nheads=2 --self_att_nheads=2

