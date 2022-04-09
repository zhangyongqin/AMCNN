from keras_preprocessing.image import  np
import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred):
    y_pred[y_pred > 0.5] = np.float32(1)
    y_pred[y_pred < 0.5] = np.float32(0)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.00001) / (K.sum(y_true_f) + K.sum(y_pred_f) + 0.00001)


def dice_score(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """
    seg = seg.flatten()
    gt = gt.flatten()
    seg[seg > ratio] = np.float32(1)
    seg[seg < ratio] = np.float32(0)
    gt[gt > ratio] = np.float(1)
    gt[gt < ratio] = np.float(0)
    dice = float(2 * (gt * seg).sum()) / float(gt.sum() + seg.sum())
    return dice