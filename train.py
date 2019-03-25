
import os

import tensorflow as tf
tf.enable_eager_execution()

import data
import model

WEIGHT_DECAY = 0.0005
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
TRAIN_EPOCHS = 45
INPUT_SIZE = 384
LEARNING_RATE = 0.0001
LOG_DIR = 'logs/person_coco_aspp'

COCO_dataset_train = data.COCO_Dataset('train')
COCO_dataset_val = data.COCO_Dataset('val')

train_ds = COCO_dataset_train.train_dataset(
    TRAIN_BATCH_SIZE, TRAIN_EPOCHS, INPUT_SIZE)
val_ds = COCO_dataset_val.val_dataset(
    VAL_BATCH_SIZE, INPUT_SIZE)

net = model.Model(weight_decay=WEIGHT_DECAY)

def loss(logits, labels):
    return tf.losses.sigmoid_cross_entropy(labels, logits)

def evaluate_model(net, val_ds):
    val_ds_iterator = val_ds.make_one_shot_iterator()
    mean_loss = 0.0
    counter = 0
    for (img, gt) in val_ds_iterator:
        logits = net(img, is_training=False)
        mean_loss += loss(logits, gt)
        counter += 1
    mean_loss /= counter
    return mean_loss

train_ds_iterator = train_ds.make_one_shot_iterator()

optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
writer = tf.contrib.summary.create_file_writer(LOG_DIR)
global_step = tf.train.get_or_create_global_step()

ckpt_prefix = os.path.join(LOG_DIR, 'ckpt')
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=net, gs=global_step)
ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_prefix, max_to_keep=5)
if ckpt_manager.latest_checkpoint is not None:
    print('Restoring from checkpoint: {}'.format(ckpt_manager.latest_checkpoint))
    ckpt.restore(ckpt_manager.latest_checkpoint)

for (img, gt) in train_ds_iterator:

    gs = global_step.numpy()

    # Forward
    with tf.GradientTape() as tape:
        logits = net(img, is_training=True)
        loss_value = loss(logits, gt) + sum(net.losses)

    # Bacward
    grads = tape.gradient(loss_value, net.variables)
    optimizer.apply_gradients(zip(grads, net.variables), global_step=global_step)

    # Display loss and images
    if gs % 100 == 0:
        with writer.as_default():
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('loss', loss_value)
                tf.contrib.summary.image('seg', tf.concat([gt, tf.sigmoid(logits)], axis=2))
        print("[%4d] Loss: %2.4f" % (gs, loss_value))

    # Calc validation loss
    if gs % 200 == 0:
        val_loss = evaluate_model(net, val_ds)
        with writer.as_default():
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('val_loss', val_loss)
        print("[%4d] Validation Loss: %2.4f" % (gs, val_loss))

    # Save checkpoint
    if gs % 1000 == 0:
        ckpt_manager.save()

ckpt_manager.save()
