# 开发者：蔡志鹏
# 开发时间：2023/7/15  13:32
import datetime
from pathlib import Path

import os
from typing import Union, Dict, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PREPROCESS_SCALE_LSDSSIMR_SAT = [1., 1., 1., 1., 1., 1., 1 / 500., 1 / 500., 1 / 500., 1 / 500., 1 / 500., 1 / 500.,
                                 1 / 500., 1. / 90., 1. / 24.]
PREPROCESS_SCALE_LSDSSIMR_METE = [1 / 25., 1 / 25., 1 / 25., 1 / 25., 1 / 25., 1 / 25.,
                                  1 / 100., 1 / 100., 1 / 100, 1 / 100., 1 / 100., 1 / 100.,
                                  1 / 5.]
PREPROCESS_OFFSET_LSDSSIMR_SAT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
PREPROCESS_OFFSET_LSDSSIMR_METE = [0, 0, 0, 0, 0, 0, -240, -240, -240, -240, -240, -240, 0]

# LSDSSIMR_DIR_PATH = Path('/opt/data/private')
# LSDSSIMR_CATALOG = os.path.join(LSDSSIMR_DIR_PATH, 'catalog.csv')


class LSDSSIMRDataset(Dataset):
    def __init__(self,
                 dir_out,
                 use_sat_channels=np.array([0, 1, 6, 9, 10, 11]),
                 use_mete_channels=np.array([]),
                 img_size: tuple = (640, 1280),
                 stride: int = 1,
                 lsdssimr_catalog: Union[str, pd.DataFrame] = None,
                 lsdssimr_data_dir: str = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 datetime_filter=None,
                 catalog_filter='default',
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 output_type=np.float32,
                 preprocess: bool = False,
                 # transform=None,
                 rescale_mode: str = 'mm',
                 verbose: bool = False,
                 mode: str = None
                 ):
        super(LSDSSIMRDataset, self).__init__()
        self.mode = mode
        if img_size is None:
            img_size = [640, 1280]
        print(f"loading {mode} dataset, from {start_date} to {end_date}")
        self.lsdssimr_dataloader = LSDSSIMRDataLoader(
            dir_out=dir_out,
            use_sat_channels=use_sat_channels,
            use_mete_channels=use_mete_channels,
            img_size=img_size,
            stride=stride,
            lsdssimr_catalog=lsdssimr_catalog,
            lsdssimr_data_dir=lsdssimr_data_dir,
            start_date=start_date,
            end_date=end_date,
            datetime_filter=datetime_filter,
            catalog_filter=catalog_filter,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            output_type=output_type,
            preprocess=preprocess,
            # transform=transform,
            rescale_mode=rescale_mode,
            verbose=verbose,
            mode=mode
        )

    def __getitem__(self, index):
        data_in, data_out = self.lsdssimr_dataloader.idx_sample(index=index)
        return data_in, data_out

    def __len__(self):
        return self.lsdssimr_dataloader.__len__()


class LSDSSIMRDataLoader:
    def __init__(self,
                 dir_out,
                 use_sat_channels=np.array([0, 1, 6, 9, 10, 11]),
                 use_mete_channels=np.array([]),
                 img_size: tuple = (640, 1280),
                 stride: int = 1,
                 lsdssimr_catalog: Union[str, pd.DataFrame] = None,
                 lsdssimr_data_dir: str = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 datetime_filter=None,
                 catalog_filter='default',
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 output_type=np.float32,
                 preprocess: bool = True,
                 transform=None,
                 down_sample_dict: Dict[str, Sequence[int]] = None,
                 rescale_mode: str = 'mm',
                 verbose: bool = False,
                 mode: str = None
                 ):
        """

        Parameters
        ----------
        use_sat_channels
            list, Which channels to use
        lsdssimr_catalog
            Name or path of LSDSSIMR catalog CSV file.
        lsdssimr_data_dir
            Directory path to LSDSSIMR data.
        start_date
            Start time of LSDSSIMR samples to generate.
        end_date
            End time of LSDSSIMR samples to generate.
        datetime_filter
            function
            Mask function applied to time_utc column of catalog (return true to keep the row).
            Pass function of the form   lambda t : COND(t)
            Example:  lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21)  # Generate only day-time events
        catalog_filter
            function or None or 'default'
            Mask function applied to entire catalog dataframe (return true to keep row).
            Pass function of the form lambda catalog:  COND(catalog)
            Example:  lambda c:  [s[0]=='S' for s in c.id]   # Generate only the 'S' events
        shuffle
            bool, If True, data samples are shuffled before each epoch.
        shuffle_seed
            int, Seed to use for shuffling.
        output_type
            np.dtype, dtype of generated tensors
        preprocess
            bool, If True, self.preprocess_data_dict(data_dict) is called before each sample generated
        down_sample_dict
            dict, downsample_dict.keys() == data_types. downsample_dict[key] is a Sequence of (t_factor, h_factor, w_factor),
            representing the downsampling factors of all dimensions.
        rescale_mode
            str,
            'std': mean std, Standard Transform
            'mm': max min, MinMax TransForm
        verbose
            bool, verbose when opening raw data files
        """
        super(LSDSSIMRDataLoader, self).__init__()
        # if lsdssimr_catalog is None:
        #     lsdssimr_catalog = LSDSSIMR_CATALOG
        # if lsdssimr_data_dir is None:
        #     lsdssimr_data_dir = LSDSSIMR_DIR_PATH

        dir_cata_out = os.path.join(dir_out, "catalog")
        os.makedirs(dir_cata_out, exist_ok=True)
        self.dir_cata_out = dir_cata_out
        self.use_sat_channels = use_sat_channels
        self.use_mete_channels = use_mete_channels
        self.img_size = img_size
        # self.raw_seq_len = raw_seq_len
        # assert seq_len <= self.raw_seq_len, f'seq_len must not be larger than raw_seq_len = {raw_seq_len}, got {seq_len}.'
        if isinstance(lsdssimr_catalog, str):
            self.catalog = pd.read_csv(lsdssimr_catalog, parse_dates=['time_utc'], low_memory=False)
        else:
            self.catalog = lsdssimr_catalog
        self.lsdssimr_data_dir = lsdssimr_data_dir
        self.datetime_filter = datetime_filter
        self.catalog_filter = catalog_filter
        self.start_date = start_date
        self.end_date = end_date
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.output_type = output_type
        self.preprocess = preprocess
        self.transform = transform
        self.down_sample_dict = down_sample_dict
        self.rescale_method = rescale_mode
        self.verbose = verbose
        self.mode = mode
        self._samples = None
        self._hdf_files = {}
        # self._sample_count = None
        # self._curr_seq_idx = 0

        if self.start_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc >= self.start_date]
        if self.end_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc < self.end_date]
        assert mode in ['train', 'val', 'test'], f"mode must be one of ('train', 'val', 'test')."
        self.catalog.to_csv(os.path.join(self.dir_cata_out, f'catalog_{mode}.csv'), index=False)

        self.get_samples()
        # self.open_files()  # open_files写在打开一个序列时，在那时调用
        # self.reset()

    def get_samples(self):
        """Compute and get the list(index) of samples by start_idx and end_idx, save in self._samples"""
        samples = {'sample_idx': [],
                   'file_index': []}
        sample_idx = 0
        for index, row in self.catalog.iterrows():
            samples['sample_idx'].append(sample_idx)
            samples['file_index'].append(index)
            sample_idx += 1
        self._samples = pd.DataFrame(columns=['sample_idx', 'file_index'], data=samples)
        self._samples.to_csv(os.path.join(self.dir_cata_out, f'samples_{self.mode}.csv'), index=False)  # 保存下来看看对不对
        if self.shuffle:
            self.shuffle_samples("get_samples")

    def shuffle_samples(self, s):
        self._samples = self._samples.sample(frac=1, random_state=self.shuffle_seed)
        self._samples.to_csv(os.path.join(self.dir_cata_out, f'{s}_samples_shuffle_{self.mode}.csv'), index=False)

    def open_files(self, sample_idx, verbose=False):
        """
        Open HDF files TODO: of one sequence, sevir of one event(date).
        """
        self._hdf_files = {}
        hdf_file_idx = self._samples[self._samples.sample_idx == sample_idx].file_index.values[0]
        file_path = self.catalog.loc[hdf_file_idx, 'file_path_relative']
        if verbose:
            print(f'Opening HDF5 file for reading: {file_path}')
        # print(self.lsdssimr_data_dir, file_path)
        # print(os.path.join(self.lsdssimr_data_dir, file_path))
        self._hdf_files[hdf_file_idx] = h5py.File(os.path.join(self.lsdssimr_data_dir, file_path), 'r')

    def close_files(self):
        """
        Close all opened files.
        """
        for f in self._hdf_files:
            self._hdf_files[f].close()
        self._hdf_files = {}

    @property
    def total_num_samples(self):
        return self._samples.sample_idx.count()

    def _read_data(self, row):
        """
        Iteratively read data into data dict. 一row就是self._samples里的一行 ('seq_idx', 'file_index')
        Finally data[imgt] gets shape (tmp_batch_size + 1, seq_len, height, width, channel).

        Parameters
        row
            A series (one row) fields of index, date, time_utc, file_path
        data
            Dict, data is a data tensor with shape = ‘tmp_batch_size’: (seq_len, height, width, channel)
        Returns
        -------
        data
            Updated data. data.shape = 'tmp_batch_size + 1': (seq_len, height, width, channel)
            # 经过一次read_data后data第一维多了一个，可能相当于读进去了一个date的数据，所以这里tmp_batch_size要+1
            # 经过一次read_data后读取进去一个sequence。
        """
        get_which_sat = list(['R0.47', 'R0.65', 'R0.83', 'R1.37', 'R1.61', 'R2.22', 'BT3.72', 'BT6.25',
                              'BT7.10', 'BT8.50', 'BT10.8', 'BT12.0', 'BT13.5', "SOZ"])
        get_which_mete = list(["10m_eastward_wind_component", "10m_northward_wind_component",
                               "2m_eastward_wind_component", "2m_northward_wind_component",
                               "50m_eastward_wind_component", "50m_northward_wind_component",
                               "10m_air_temperature", "2m_air_temperature",
                               "2m_dewpoint_temperature", "mean_sea_level_pressure",
                               "skin_temperature", "surface_pressure",
                               "total_precipitation"])
        _sample_idx = row['sample_idx']
        _file_index = row['file_index']

        self.open_files(_sample_idx)
        _input_data = []
        # _output_data = []
        # input data

        # print(self._hdf_files[_file_index][f"Meteorological_Data/{get_which_mete[0]}"].shape)

        for i in self.use_mete_channels:
            _data = np.array(
                self._hdf_files[_file_index][f"Meteorological_Data/{get_which_mete[i]}"][:self.img_size[0],
                90:90 + self.img_size[1]].astype(self.output_type))  # _data.shape: 640*1280
            _data = np.expand_dims(_data, axis=0)  # _data.shape: 1*H*W
            _input_data += [_data, ]
        for i in self.use_sat_channels:
            _data = np.array(
                self._hdf_files[_file_index][f'Satellite_Source_Data/{get_which_sat[i]}'][:self.img_size[0],
                90:90 + self.img_size[1]].astype(self.output_type))  # _data.shape: 640*1280
            _data = np.expand_dims(_data, axis=0)  # _data.shape: 1*H*W
            _input_data += [_data, ]

        # output data
        # # 输入所有数据，输出 DST 数据
        # _data = np.array(
        #     self._hdf_files[_file_index]['others/DST_binary'][:self.img_size[0], 90:90 + self.img_size[1]]).astype(
        #     self.output_type)  # dst.shape: 640*1280

        _data = np.array(
            # self._hdf_files[_file_index]['others/DST_binary'][:self.img_size[0], 90:90 + self.img_size[1]]).astype(
            # self.output_type)  # dst.shape: 640*1280
            self._hdf_files[_file_index]['others/DST'][:self.img_size[0], 90:90 + self.img_size[1]]).astype(
            self.output_type)  # dst.shape: 640*1280


        # Occurrence detection
        # _data[_data < 12] = 0.
        # _data[_data >= 12] = 1.

        # Intensity detection
        _data[_data < 12] = 0.
        _data[(_data >= 12) & (_data < 15)] = 1.
        _data[(_data >= 15) & (_data < 17)] = 2.
        _data[(_data >= 17) & (_data < 19)] = 3.
        _data[(_data >= 19) & (_data < 21)] = 4.
        _data[(_data >= 21) & (_data < 23)] = 5.
        _data[(_data >= 23) & (_data <= 24)] = 6.

        _data = _data.astype(self.output_type)

        output_data = _data
        input_data = np.concatenate([_input_data[i] for i in range(len(_input_data))], axis=0)
        input_data[input_data == -99.] = 0  # data.shape: 6*640*1280
        output_data[output_data == -99.] = 0  # data.shape: 640*1280



        return input_data, output_data

    def idx_sample(self, index):
        """
        Parameters
        ----------
        index
            the index of the batch to sample.
        Returns
        -------
        ret
            tensor
            if self.preprocess == False:
                ret.shape == (batch_size, height, width, seq_len) TODO:
        """
        # seq_idx_curr = index
        # seqs_batch_in, seqs_batch_out = self.load_seqs_batch(seq_start_idx=seq_idx_curr, seq_batch_size=1)
        batch_in, batch_out = self._read_data(row=self._samples.iloc[index])

        if self.preprocess:  # todo: 将‘NTCHW’， ‘TNCHW’ 等之间的互相转换，以及rescale和offset
            batch_in, batch_out = self.preprocess_data(batch_in, batch_out, rescale=self.rescale_method)
            # pass

        return batch_in, batch_out

    def __len__(self):
        """Use only when self.sample_mode == 'sequent'"""
        return self.total_num_samples

    def preprocess_data(self, data_in, data_out, rescale: str = 'mm'):
        if rescale == 'mm':
            '''scale_dict = PREPROCESS_SCALE_LSDSSIMR
            offset_dict = PREPROCESS_OFFSET_LSDSSIMR
            scale_list = [scale_dict[self.scale_list[i]] for i in range(len(self.scale_list))]
            offset_list = [offset_dict[self.scale_list[i]] for i in range(len(self.scale_list))]'''
            scale_list = np.array([PREPROCESS_SCALE_LSDSSIMR_METE[i] for i in self.use_mete_channels] +
                                  [PREPROCESS_SCALE_LSDSSIMR_SAT[j] for j in self.use_sat_channels])
            offset_list = np.array(
                [PREPROCESS_OFFSET_LSDSSIMR_METE[i] for i in self.use_mete_channels] +
                [PREPROCESS_OFFSET_LSDSSIMR_SAT[j] for j in self.use_sat_channels]
            ).reshape([1, 1, len(self.use_mete_channels) + len(self.use_sat_channels)])
            # scale = PREPROCESS_SCALE_LSDSSIMR_SAT[-1]
            # offset = PREPROCESS_OFFSET_LSDSSIMR_SAT[-1]
            # if isinstance(seqs_data, np.ndarray):
            # [C, H, W] to [H, W, C]
            data_in = (data_in.astype(np.float32).transpose(1, 2, 0) + offset_list) * scale_list
            data_in = np.transpose(data_in, axes=(2, 0, 1))
            data_in[data_in > 1.] = 1.
            data_in[data_in < 0.] = 0.
            # data_out = (data_out.astype(np.float32) + offset) * scale

            # data_out[data_out > 1.] = 1.
            # data_out[data_out < 0.] = 0.

        elif rescale == 'std':
            # todo: 没完全写完
            pass
        elif rescale is None:
            pass
        else:
            raise ValueError(f'Invalid rescale option: {rescale}.')
        seqs_data_in = torch.from_numpy(data_in).float()
        seqs_data_out = torch.from_numpy(data_out).float()
        return seqs_data_in, seqs_data_out
