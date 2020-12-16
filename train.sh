NEPTUNE=0
WIDTH=9
HEIGHT=9
MINES=10
EPISODES=100000
MODEL_NAME='WHATEVER'
UPDATE=20 #computing values in neptune
BSIZE=64
CONVDIM=64
LINDIM=128
GAMMA=0.05
EPS_DECAY=10000 #the higher, the slower it converges
LR_DECAY=0.999999975


python3.8 main.py \
            --neptune=$NEPTUNE --update_every=$UPDATE --batch_size=$BSIZE --conv_dim=$CONVDIM \
	    --lin_dim=$LINDIM --gamma=$GAMMA --eps_decay=$EPS_DECAY --lr_decay=$LR_DECAY \
	    --episodes=$EPISODES

