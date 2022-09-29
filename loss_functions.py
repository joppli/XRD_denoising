#!/usr/bin/env python3
import tensorflow as tf

class mae_ssim_loss(tf.keras.losses.Loss):
    """Combination of mean absolute error (MAE) and structural similarity index (SSIM).

    Attributes
    ----------
    max_val_ssim : float
        Structural similarity index maximum value
    alpha : float
        Defines trade-off between SSIM and MAE

    Methods
    -------
    call(y_true, y_pred)
        Returns the loss as a mix of SSIM and MAE
    get_config()
        Updates config
    """

    def __init__(self, max_val_ssim=1.0, alpha = 0.7, **kwargs):
        super(mae_ssim_loss, self).__init__(name="mae_ssim_loss")
        self.max_val_ssim = max_val_ssim
        self.alpha = alpha
        super(mae_ssim_loss, self).__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        mae = tf.reduce_mean(tf.keras.losses.mae(y_true, y_pred))
        ssim = tf.reduce_mean(tf.image.ssim(img1=y_true, img2=y_pred, max_val=self.max_val_ssim))
        return (1. - self.alpha)*mae + self.alpha*(1-ssim)/2

    def get_config(self):
        config = super(mae_ssim_loss, self).get_config()
        config.update({"max_val_ssim": self.max_val_ssim, "alpha": self.alpha})
        return config

class mae_ssim_ms_loss(tf.keras.losses.Loss):
    """Combination of mean absolute error (MAE) and multiscale structural similarity index (MSSIM).

    Attributes
    ----------
    max_val_ssim : float
        Multiscale structural similarity index maximum value
    alpha : float
        Defines trade-off between MSSIM and MAE
    filter_size : int
        Size of the Gaussian filter to be used for MSSIM
    power_factors : list[int]
        Power factors for each level of downscaling as defined by MSSIM

    Methods
    -------
    call(y_true, y_pred)
        Returns the loss as a mix of MSSIM and MAE
    get_config()
        Updates config
    """

    def __init__(self, max_val_ssim=1.0, alpha=0.7, filter_size=11, power_factors=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], **kwargs):
        super(mae_ssim_ms_loss, self).__init__(name="mae_ssim_ms_loss")
        self.max_val_ssim = max_val_ssim
        self.alpha = alpha
        self.filter_size = filter_size
        self.power_factors = power_factors
        super(mae_ssim_ms_loss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        mae = tf.reduce_mean(tf.keras.losses.mae(y_true, y_pred))
        ssim_ms = tf.reduce_mean(tf.image.ssim_multiscale(img1=y_true, img2=y_pred, max_val=self.max_val_ssim, filter_size=self.filter_size, power_factors=self.power_factors))
        return (1. - self.alpha)*mae + self.alpha*(1-ssim_ms)

    def get_config(self):
        config = super(mae_ssim_ms_loss, self).get_config()
        config.update({"max_val_ssim": self.max_val_ssim, "alpha": self.alpha, "filter_size": self.filter_size, "power_factors": self.power_factors})
        return config
