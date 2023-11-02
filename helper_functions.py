#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
import random
import warnings
import h5py

from PIL import Image
from scipy.ndimage import gaussian_filter
from shutil import copy2
from datetime import datetime

import loss_functions as lf

class CustomDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator to be used during training of the networks.

    Attributes
    ----------
    df : pandas dataframe
        Dataframe containing the names of the files to be used
    batch_size : int
        Batch size to be used during training
    target_shape : tuple[int]
        Input shape in the form of (# pixels in x, # pixels in y, 1)
    simulate_lc : None or string
        Wether to simulate the low count data (Poisson or Gaussian noise)
    augmentation : bool
        Whether to apply data augmentation (flipping frames left-right)

    Methods
    -------
    __get_normalized_image(path)
        Loads an image specified by path and returns it as an array with values between 0 and 1
    __normalize_image(img)
        Normalizes an image (in the form of a numpy array) to have values between 0 and 1
    __augment_data(lc, hc, id)
        Performs potential data augmentation on both low- and high count depending on the given index id
    __getitem__(index)
        Returns a complete batch at position defined by index
    __len__()
        Returns the number of batches
    on_epoch_end
        Executed at the end of every epoch (shuffling the order of the batches)
    """
    
    def __init__(self, df, batch_size, target_shape, simulate_lc, augmentation=False, scale_range=[5000,15000]):
        if isinstance(df, pd.DataFrame):
            self.df = df.copy()
            self.n = len(self.df)
            assert simulate_lc is None
            self.simulate_lc = False
            # Check whether to crop the image (assuming all images in the folder have the same size)
            self.cb = get_cropping_box(self.df.iloc[0,0], target_shape[0:2])
        else:
            # In this case the files are already cropped
            self.df = df # df[0] is a HDF5 file name, df[1] is the dataset name
            assert simulate_lc.lower() == 'poisson' or simulate_lc.lower() == 'gaussian'
            self.simulate_lc = True
            with h5py.File(self.df[0], 'r') as file:
                self.n = file[self.df[1]].shape[0]
            self.cb = None

        self.batch_size = batch_size
        self.indices = np.arange(self.n)
        self.target_shape = target_shape
        self.augmentation = augmentation
        self.scale_range = scale_range
    
    def __get_normalized_image(self, path):
        img = np.asarray(Image.open(path).crop(box=self.cb), dtype=np.int32).astype(np.float32)
        return self.__normalize_image(img)

    def __normalize_image(self, img):
        scale = np.random.randint(low=self.scale_range[0], high=self.scale_range[1])
        img = np.clip(img, a_min=0., a_max=None)
        img = img / np.sum(img) * scale
        return np.clip(img, a_min=0., a_max=1.)

    def __augment_data(self, lc, hc, id):
        if id == 0:
            # Flip both LC and HC left-right
            lc_aug = np.fliplr(lc)
            hc_aug = np.fliplr(hc)
        else:
            lc_aug = lc.copy()
            hc_aug = hc.copy()
        return self.__normalize_image(lc_aug), self.__normalize_image(hc_aug)

    def __getitem__(self, index):
        inds = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        if self.augmentation is True:
            X = []
            y = []
            ids_aug = [np.random.randint(0,2) for _ in range(len(inds))] # list of random integers in range [0,2) defining data augmentation operations
            if self.simulate_lc is False:
                batches = self.df.iloc[inds]
                for a, b, id in zip(batches.iloc[:,0], batches.iloc[:,1], ids_aug):
                    hc = np.asarray(Image.open(b).crop(box=self.cb), dtype=np.int32).astype(np.float32)
                    lc = np.asarray(Image.open(a).crop(box=self.cb), dtype=np.int32).astype(np.float32)
                    lc, hc = self.__augment_data(lc, hc, id)
                    X.append(lc)
                    y.append(hc)
            else:
                # Load the data from the HDF5 file
                with h5py.File(self.df[0], 'r') as file:
                    dset = file[self.df[1]]
                    for i, id in zip(inds, ids_aug):
                        hc = dset[i,:,:,1]
                        lc = dset[i,:,:,0]
                        lc, hc = self.__augment_data(lc, hc, id)
                        X.append(lc)
                        y.append(hc)
        else:
            if self.simulate_lc is False:
                batches = self.df.iloc[inds]
                X = np.asarray([self.__get_normalized_image(x) for x in batches.iloc[:,0]])
                y = np.asarray([self.__get_normalized_image(x) for x in batches.iloc[:,1]])
            else:
                # Load the data from the HDF5 file
                X = []
                y = []
                with h5py.File(self.df, 'r') as file:
                    dset = file[self.df[1]]
                    for i in inds:
                        X.append(self.__normalize_image(dset[i,:,:,0]))
                        y.append(self.__normalize_image(dset[i,:,:,1]))
        X = np.asarray(X)
        y = np.asarray(y)
        return tf.expand_dims(X, -1), tf.expand_dims(y, -1)

    def __len__(self):
        return self.n // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def convert_zenodo_hdf5_to_tif(file, save_folder):
    """Converts an HDF5 file (containing stacked low- and high-count data) to individual TIF files.

    The HDF5 files for training, validation, and test set can be found at:
    https://zenodo.org/records/8237173

    Parameters
    ----------
    file : str
        HDF5 filename
    save_folder : str
        Folder where to store the TIF files

    Returns
    -------
    Within save_folder two sub-folders "LC" and "HC" are created where
    the individual TIF files are saved in sorted order.
    """
    # Load raw data into RAM
    with h5py.File(file, 'r') as fid:
        lc = fid['low_count']['data'][:,:,:]
        hc = fid['high_count']['data'][:,:,:]

    # Create low- and high-count subfolders
    lc_folder = os.path.join(save_folder, "LC")
    hc_folder = os.path.join(save_folder, "HC")

    if not os.path.exists(lc_folder):
        os.makedirs(lc_folder)

    if not os.path.exists(hc_folder):
        os.makedirs(hc_folder)

    # Conversion to TIF files
    for i in range(lc.shape[-1]):
        Image.fromarray(lc[:,:,i].astype(np.int32)).save(os.path.join(lc_folder, f"{i:05d}_lc.tif"))
        Image.fromarray(hc[:,:,i].astype(np.int32)).save(os.path.join(hc_folder, f"{i:05d}_hc.tif"))

def get_dataframes_separate_datasets(train_folder, valid_folder, save_folder, batch_size, target_shape, shuffle=True, simulate_lc=None, augmentation_train=False, augmentation_valid=False):
    """Returns a train and validation dataframe from two given folders.

    Parameters
    ----------
    train_folder : string
        Folder containing the files to be used for training
    valid_folder : string
        Folder containing the files to be used for validation
    save_folder : string
        Folder where the simulated low count frames are saved
    batch_size : int
        Batch size to be used during training
    target_shape : tuple[int]
        Input shape in the form of (# pixels in x, # pixels in y, 1)
    shuffle : bool
        Whether to shuffle the training and validation data
    simulate_lc : None or string
        Wether to simulate the low count data (Poisson or Gaussian noise)
    augmentation_train : bool
        Whether to perform data augmentation on the training dataset
    augmentation_valid : bool
        Whether to perform data augmentation on the validation dataset

    Returns
    -------
    data_train : CustomDataGenerator instance
    data_valid : CustomDataGenerator instance
    """

    def get_train_and_valid_names(name):
        """Returns the names of the low- or high count training and validation data files

        Parameters
        ----------
        name : string
            low count ('LC') or high count ('HC')

        Returns
        -------
        content_train : sorted list of training data files
        content_valid : sorted list of validation data files
        """
        train = os.path.join(train_folder, name)
        valid = os.path.join(valid_folder, name)
        content_train = np.sort(np.asarray([os.path.join(train, file) for file in os.listdir(train)]))
        content_valid = np.sort(np.asarray([os.path.join(valid, file) for file in os.listdir(valid)]))
        return content_train, content_valid

    # Get the filenames
    train_hc_files, valid_hc_files = get_train_and_valid_names("HC")
    train_lc_files, valid_lc_files = get_train_and_valid_names("LC")

    # Shuffle the data if required
    if shuffle is True:            
        ids_train = np.random.permutation(len(train_hc_files)) # random permutation of indices for shuffling
        ids_valid = np.random.permutation(len(valid_hc_files)) # random permutation of indices for shuffling
        train_hc_files = train_hc_files[ids_train]
        valid_hc_files = valid_hc_files[ids_valid]
        train_lc_files = train_lc_files[ids_train]
        valid_lc_files = valid_lc_files[ids_valid]
        assert len(train_hc_files) == len(train_lc_files) and len(valid_hc_files) == len(valid_lc_files) # sanity check

    n_train_files = len(train_hc_files)
    n_valid_files = len(valid_hc_files)

    print(f"Training set size: {n_train_files}")
    print(f"Validation set size: {n_valid_files}")

    # Create custom data generator object
    if simulate_lc['type'] is None:
        train_df = pd.DataFrame(np.asarray([[train_lc_files], [train_hc_files]]).squeeze().transpose(), columns=['low_count_train', 'high_count_train'])
        valid_df = pd.DataFrame(np.asarray([[valid_lc_files], [valid_hc_files]]).squeeze().transpose(), columns=['low_count_valid', 'high_count_valid'])
        data_train = CustomDataGenerator(df=train_df, batch_size=batch_size, target_shape=target_shape, simulate_lc=None, augmentation=augmentation_train)
        data_valid = CustomDataGenerator(df=valid_df, batch_size=batch_size, target_shape=target_shape, simulate_lc=None, augmentation=augmentation_valid)
    else:
        # Check whether to crop the image (assuming all images in the folder have the same size)
        cb = get_cropping_box(train_hc_files[0], target_shape[0:2])
        print(f"Simulating LC images ...")
        train = np.empty(shape=(n_train_files, target_shape[0], target_shape[1], 2))
        valid = np.empty(shape=(n_valid_files, target_shape[0], target_shape[1], 2))
        if simulate_lc['type'].lower() == 'poisson': # simulate low count using Poisson noise statistics
            for i in range(n_train_files):
                hc_train = np.asarray(Image.open(train_hc_files[i]).crop(box=cb), dtype=np.int32).astype(np.float32)
                lc_train = np.asarray(Image.open(train_lc_files[i]).crop(box=cb), dtype=np.int32).astype(np.float32)
                train[i,:,:,0] = simulate_poisson_low_count_data(hc_train, lc_train, simulate_lc['constant'])
                train[i,:,:,1] = hc_train
            for i in range(n_valid_files):
                hc_valid = np.asarray(Image.open(valid_hc_files[i]).crop(box=cb), dtype=np.int32).astype(np.float32)
                lc_valid = np.asarray(Image.open(valid_lc_files[i]).crop(box=cb), dtype=np.int32).astype(np.float32)
                valid[i,:,:,0] = simulate_poisson_low_count_data(hc_valid, lc_valid, simulate_lc['constant'])
                valid[i,:,:,1] = hc_valid
        elif simulate_lc['type'].lower() == 'gaussian': # simulate low count using normal Gaussian white noise statistics
            for i in range(n_train_files):
                hc_train = np.asarray(Image.open(train_hc_files[i]).crop(box=cb), dtype=np.int32).astype(np.float32)
                lc_train = np.asarray(Image.open(train_lc_files[i]).crop(box=cb), dtype=np.int32).astype(np.float32)
                train[i,:,:,0] = simulate_gaussian_low_count_data(hc_train, lc_train, simulate_lc['constant'])
                train[i,:,:,1] = hc_train
            for i in range(n_valid_files):
                hc_valid = np.asarray(Image.open(valid_hc_files[i]).crop(box=cb), dtype=np.int32).astype(np.float32)
                lc_valid = np.asarray(Image.open(valid_lc_files[i]).crop(box=cb), dtype=np.int32).astype(np.float32)
                valid[i,:,:,0] = simulate_gaussian_low_count_data(hc_valid, lc_valid, simulate_lc['constant'])
                valid[i,:,:,1] = hc_valid
        
        temp_folder = os.path.join(save_folder, "simulated_noise")
        fn = os.path.join(temp_folder, "sim_data.hdf5")

        print(f"Saving simulated LC train and validation data to '{temp_folder}'.")
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)
            
        with h5py.File(fn, 'w') as file:
            _ = file.create_dataset('simulated_lc_train', data=train, dtype=np.float32)
            _ = file.create_dataset('simulated_lc_valid', data=valid, dtype=np.float32)
        print(f"Done.")
        data_train = CustomDataGenerator(df=[fn, "simulated_lc_train"], batch_size=batch_size, target_shape=target_shape, simulate_lc=simulate_lc['type'], augmentation=augmentation_train, scale_range=[5000,15000])
        data_valid = CustomDataGenerator(df=[fn, "simulated_lc_valid"], batch_size=batch_size, target_shape=target_shape, simulate_lc=simulate_lc['type'], augmentation=augmentation_valid, scale_range=[10000,10001])

    return data_train, data_valid

def load_model(model, folder, shape, loss=lf.mae_ssim_ms_loss(max_val_ssim=1., alpha=0.7)):
    """Loads a given model with pre-trained weights and biases.

    Parameters
    ----------
    model : keras.Model
        Instantiated model object
    chk_fn : str
        Path to a checkpoint file (.hdf5) containing the model weights and biases
    shape : tuple[int]
        Input shape in the form of (# pixels in x, # pixels in y, 1)
    loss : keras.Loss
        Loss function used during training (only needed for compiling step)

    Returns
    -------
    model : model with trained weights and biases
    history : dataframe containing the models training history
    """
    
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(), metrics=['mae'])
    model.build(input_shape=(None,shape[0],shape[1],shape[2]))

    # Load the model weights if model is given
    chk_fn = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".hdf5")][0]
    model.load_weights(chk_fn)

    # Check if a training .log file are present
    base_folder, _ = os.path.split(chk_fn)
    log_file = [os.path.join(base_folder, file) for file in os.listdir(base_folder) if file.endswith(".log")][0]
    try:
        history = pd.read_csv(log_file)
    except:
        history = None
        print("No .log file with training history found.")

    return model, history

def make_or_restore_model(ckpt_folder, new_model=None, input_shape=None):
    """Restores the model saved within a checkpoint folder or creates a fresh one if no checkpoints are found.

    Parameters
    ----------
    chkpt_folder : string
        Checkpoint folder to be searched for checkpoint files
    new_model : keras.Model
        (Compiled) model to be used (throws error if None is provided)
    input_shape : tuple[int]
        Input shape in the form of (# pixels in x, # pixels in y, 1)

    Returns
    -------
    new_model : compiled model with loaded weights
    val_loss : previous (best) validation loss (or infinity if no checkpoint files are found)
    """

    if not os.path.exists(ckpt_folder) and new_model is None:
        print(f"Checkpoint folder not found. Either enter an existing folder or pass a new model.")
        sys.exit(0)
    
    # Create checkpoint folder in case if it doesn't exist
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    
    ckpts = [os.path.join(ckpt_folder, name) for name in os.listdir(ckpt_folder) if name.endswith(".hdf5")]
    if ckpts:
        latest_ckpt, ext = os.path.splitext(max(ckpts, key=os.path.getctime)) # split filename into name and file extension
        val_loss = np.float32(latest_ckpt.split("=",1)[1])
        latest_ckpt = latest_ckpt+ext # re-combine filename
        if new_model is not None and input_shape is not None:
            print(f"Restoring model from {latest_ckpt} ...")
            new_model.build(input_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
            new_model.load_weights(latest_ckpt) # load latest model weights
            return new_model, val_loss
    else:
        print(f"No checkpoints found in the given folder. Initialize given new model ...")
        if new_model is None:
            print(f"Please specify the model using the keyword 'new_model'.")
            sys.exit(0)
        else:
            return new_model, np.Inf

def prepare_datasets(lc_folder, hc_folder, target_folder, fraction=[0.7,0.2,0.1]):
    """Splits the low- and high-count data according to fraction into a training, a validation and a test set.
    
    Parameters
    ----------
    lc_folder : string
        Folder containing the low-count data files
    hc_folder : string
        Folder containing the high-count data files
    target_folder : string
        Folder where the files belonging to the individual sets are saved to
    fraction : list[float]
        List of fractional values [training, validation, test]

    Returns
    -------
    creates new folders containing the training, validation and test files
    for both low- and high-count data files
    """

    n_files = len(os.listdir(lc_folder))
    if n_files != len(os.listdir(hc_folder)):
        print(f"Number of LC files ({n_files}) is not equal to the number of HC files ({len(os.listdir(hc_folder))}).")
        sys.exit(0)

    # Define the random order of occurrence
    order = np.random.permutation(range(n_files))
    lc_files = np.sort(np.asarray([os.path.join(lc_folder, fn) for fn in os.listdir(lc_folder)]))[order]
    hc_files = np.sort(np.asarray([os.path.join(hc_folder, fn) for fn in os.listdir(hc_folder)]))[order]
    
    # Split the data into training, validation and test set
    N_frac = np.round(np.multiply(fraction, n_files)).astype(np.int16)
    lc_train, lc_valid, lc_test = lc_files[:N_frac[0]], lc_files[N_frac[0]:N_frac[0]+N_frac[1]], lc_files[N_frac[0]+N_frac[1]:]
    hc_train, hc_valid, hc_test = hc_files[:N_frac[0]], hc_files[N_frac[0]:N_frac[0]+N_frac[1]], hc_files[N_frac[0]+N_frac[1]:]

    # Store the data into the target folder with appropriate names
    train_folder = os.path.join(target_folder, "train")
    valid_folder = os.path.join(target_folder, "validation")
    test_folder = os.path.join(target_folder, "test")
    os.makedirs(os.path.join(train_folder, "LC"))
    os.makedirs(os.path.join(valid_folder, "LC"))
    os.makedirs(os.path.join(test_folder, "LC"))
    os.makedirs(os.path.join(train_folder, "HC"))
    os.makedirs(os.path.join(valid_folder, "HC"))
    os.makedirs(os.path.join(test_folder, "HC"))
    
    # Copy train files
    lc_tmp_target = os.path.join(train_folder, "LC")
    hc_tmp_target = os.path.join(train_folder, "HC")
    print(f"Copying training files ...")
    for i in range(N_frac[0]):
        copy2(lc_train[i], lc_tmp_target)
        copy2(hc_train[i], hc_tmp_target)
    
    # Copy validation files
    print(f"Copying validation files ...")
    lc_tmp_target = os.path.join(valid_folder, "LC")
    hc_tmp_target = os.path.join(valid_folder, "HC")
    for i in range(N_frac[1]):
        copy2(lc_valid[i], lc_tmp_target)
        copy2(hc_valid[i], hc_tmp_target)

    # Copy test files
    print(f"Copying test files ...")
    lc_tmp_target = os.path.join(test_folder, "LC")
    hc_tmp_target = os.path.join(test_folder, "HC")
    for i in range(N_frac[2]):
        copy2(lc_test[i], lc_tmp_target)
        copy2(hc_test[i], hc_tmp_target)

    print(f"Successfully copied data to {target_folder}.")

def train_model(base_folder, model_name, model, input_shape, train_folder, valid_folder, n_epochs, batch_size, simulate_lc, rng_seed, lrs=None):
    """Trains a given model on a given set of low-count data ('inputs') and high-count data ('labels').
       Saves the results in the form of checkpoints to a given folder.
    
    Parameters
    ----------
    base_folder : string
        Folder for storing the output files
    model_name : string
        Name of the model resp. the output folder
    model : keras.Model
        Compiled neural network model
    input_shape : tuple[int]
        Input shape in the form of (# pixels in x, # pixels in y, 1)
    train_folder : string
        Folder containing the training data (needs to contain separate HC and LC folders!)
    valid_folder : string
        Folder containing the validation data (needs to contain separate HC and LC folders!)
    n_epochs : int
        Number of epochs to train the model
    batch_size : int
        Batch size to be used during training
    simulate_lc : dict
        Dictionary defining whether and how to simulate the low-count data
    rng_seed : int
        Random number seed
    lrs : None or function
        Optional learning rate scheduler function for e.g. decreasing the learning rate during training

    Returns
    -------
    trains the given model on the given input data and stores the output files
    """

    # Reset the random seed
    reset_random_seeds(rng_seed)

    # Define the save folder and create it if it doesn't exist yet
    save_folder = os.path.join(base_folder, model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Get the data in the form of batches
    gen_train, gen_valid = get_dataframes_separate_datasets(train_folder=train_folder, valid_folder=valid_folder, 
                                                                save_folder=save_folder, batch_size=batch_size,
                                                                target_shape=input_shape, shuffle=True,
                                                                simulate_lc=simulate_lc, augmentation_train=True, augmentation_valid=True)
    
    # Get the already compiled model if it exists (will create new folder if it doesn't already)
    model, prev_val_loss = make_or_restore_model(ckpt_folder=save_folder, new_model=model, input_shape=input_shape)

    # Setup a few callbacks
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_folder, "tensorboard_logs", datetime.now().strftime("%Y%m%d-%H%M%S")), 
                                                                histogram_freq=1)
        
    model_ckpt_cb = SaveModelCheckpoint(filepath=os.path.join(save_folder, "ckpt_loss={val_loss:.8f}.hdf5"), 
                                           save_best_only=True, mode='min', monitor='val_loss', verbose=1, prev_best=prev_val_loss)
        
    csv_logger_cb = tf.keras.callbacks.CSVLogger(filename=os.path.join(save_folder, model_name+"_history.log"), append=True)
  
    if lrs is not None:
        lr_cb = tf.keras.callbacks.LearningRateScheduler(lrs)
        model_cbs = [tb_cb, model_ckpt_cb, csv_logger_cb, lr_cb]
    else:
        model_cbs = [tb_cb, model_ckpt_cb, csv_logger_cb]

    # Train the model
    model.fit(gen_train, epochs=n_epochs, verbose=2, validation_data=gen_valid, callbacks=model_cbs)

def reset_random_seeds(seed):
    """Initializes a global random number generator using the given seed.

    Parameters
    ----------
    seed : int
        Random number seed to be used
    
    Returns
    -------
    global seed it set
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_cropping_box(fn, target_shape):
    """Returns the boundary box for cropping images.

    Parameters
    ----------
    target_shape : tuple[int]
    """
    width, height = Image.open(fn).size
    if width > target_shape[1]:
        rmv_px_x = width - target_shape[1]
    else:
        rmv_px_x = 0
    
    if height > target_shape[0]:
        rmv_px_y = height - target_shape[0]
    else:
        rmv_px_y = 0

    cb = (rmv_px_x//2, rmv_px_y//2, width - (rmv_px_x//2+rmv_px_x%2), height - (rmv_px_y//2+rmv_px_y%2))
    return cb

def simulate_poisson_low_count_data(hc, lc, const=None):
    """Simulates a low-count frame from a given high-count frame according to Poisson noise statistics.

    Parameters
    ----------
    hc : 2D array[float]
        High-count data frame
    lc : 2D array[float]
        Low-count data frame
    const : None or int
        Optional constant lambda value to be used for scaling

    Returns
    -------
    2D array[float]
        Simulated low-count data frame
    """

    # Shift the data into positive range as Poisson noise can only be generated for positive numbers
    hc = hc - np.amin(hc)
    lc = lc - np.amin(lc)

    if const is None:
        # Calculate the noisy high count based on the true LC and HC statistics
        hc_noise = np.random.poisson(hc*np.sum(lc)/np.sum(hc))
    else:
        hc_noise = np.random.poisson(hc*const)

    return gaussian_smooth(hc_noise)

def simulate_gaussian_low_count_data(hc, lc, const=None):
    """Simulates a low-count frame from a given high-count frame according to normal Gaussian noise statistics.

    Parameters
    ----------
    hc : 2D array[float]
        High-count data frame
    lc : 2D array[float]
        Low-count data frame
    const : None or list[int]
        Optional constant mu and sigma value to be used for scaling

    Returns
    -------
    2D array[float]
        Simulated low-count data frame
    """

    # Normalize the input
    hc = (hc - np.amin(hc))/(np.amax(hc) - np.amin(hc))
    lc = (lc - np.amin(lc))/(np.amax(lc) - np.amin(lc))

    # Calculate the noisy high count
    if const is None:
        mu = np.mean(lc)
        sigma = np.std(lc)
    else:
        mu = const[0]
        sigma = const[1]

    # mu should be the pixel intensity value of the HC? Not a constant value
    hc_noise = hc + np.random.normal(mu, sigma, hc.shape)

    return gaussian_smooth(hc_noise)

def gaussian_smooth(img, width_min=0.3, width_max=0.5):
    """Convolutes an input image with a Gaussian kernel of random with.

    Parameters
    ----------
    img : 2D array[float]
        Input data frame
    width_min : float
        Minimum width (standard deviation of the Gaussian kernel)
    width_max : float
        Maximum width

    Returns
    -------
    2D array[float]
        Simulated low-count data frame
    """

    # Randomly select the Gaussian kernel FWHM
    width = width_min + (width_max - width_min)*np.random.rand()

    return gaussian_filter(img, sigma=width)

def calculate_global_mean_and_std(folder):
    """Calculates the global mean and standard deviation value of all the low-count data files contained in folder.

    Parameters
    ----------
    folder : string
        Folder containing the low-count data (.tif format)

    Returns
    -------
    mean : float
        Mean value of the original data
    mean_norm : float
        Mean value of the normalized data
    std : float
        Standard deviation of the original data
    std_norm : float
        Standard deviation of the normalized data
    """

    mean = 0.
    mean_norm = 0.
    std = 0.
    std_norm = 0.

    normalize_image = lambda I: (I - np.amin(I))/(np.amax(I) - np.amin(I))

    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".tif")]
    n = len(files)

    for i in range(n):
        img = np.asarray(Image.open(files[i]), dtype=np.int32).astype(np.float32)
        img_norm = normalize_image(img)
        mean += np.mean(img)
        mean_norm += np.mean(img_norm)
        std += np.std(img)
        std_norm += np.std(img_norm)
    
    return mean/n, mean_norm/n, std/n, std_norm/n

def calculate_global_lambda(lc_folder, hc_folder, type='median'):
    """Calculates multiplicative factor lambda as the mean or median over the sum of 
       low-count intensity divided by the sum of high-count intensity for all frame pairs.

    Parameters
    ----------
    lc_folder : string
        Folder containing the low-count training data (.tif format)
    hc_folder : string
        Folder containing the high-count training data (.tif format)

    Returns
    -------
    val : float
        Mean or median value over all (low-count, high-count) data pairs
    """

    files_lc = np.sort([os.path.join(lc_folder, file) for file in os.listdir(lc_folder)])
    files_hc = np.sort([os.path.join(hc_folder, file) for file in os.listdir(hc_folder)])
    assert len(files_lc) == len(files_hc) # sanity check
    
    n_files = len(files_lc)
    arr = np.zeros((n_files,1))

    for i in range(n_files):
        lc = np.asarray(Image.open(files_lc[i]), dtype=np.int32).astype(np.float32)
        hc = np.asarray(Image.open(files_hc[i]), dtype=np.int32).astype(np.float32)
        hc = hc - np.amin(hc)
        lc = lc - np.amin(lc)
        arr[i] = np.sum(lc)/np.sum(hc)

    if type.lower() == 'mean':
        return np.mean(arr)
    elif type.lower() == 'median':
        return np.median(arr)
    else:
        raise ValueError("'type' has to be either 'mean' or 'median'.")

class SaveModelCheckpoint(tf.keras.callbacks.Callback):
    """Callback class allowing to ressume training of a model from the last checkpoint.

    Source: 
    https://github.com/keras-team/keras/issues/12803

    """
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=True,
                 mode='auto', period=1, prev_best=None):
        super(SaveModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('SaveModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'
        
        if(prev_best == None):
            if mode == 'min':
                self.monitor_op = np.less
                self.best = np.Inf
            elif mode == 'max':
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                    self.monitor_op = np.greater
                    self.best = -np.Inf
                else:
                    self.monitor_op = np.less
                    self.best = np.Inf
        else:
            if mode == 'min':
                self.monitor_op = np.less
                self.best = prev_best
            elif mode == 'max':
                self.monitor_op = np.greater
                self.best = prev_best
            else:
                if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                    self.monitor_op = np.greater
                    self.best = prev_best
                else:
                    self.monitor_op = np.less
                    self.best = prev_best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
        