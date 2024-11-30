Models = {
    "TCN-effKAN": {
        "model": "TCN-effKAN",
        'e_layers': '2',
        'd_layers': '1',
        'task_name': 'conv',
        'target': 'y',
        'enc_in': '6',
        'c_out': '1',
        'data': 'conv_ETTh1',
        'features': 'MS',
    },
    "Transformer":{
        'task_name': 'short_term_forecast', 'is_training': '1',
        'seasonal_patterns': "Monthly",
        'model': 'Transformer', 'data': 'former',
        'features': 'MS', 'e_layers': '2',
        'd_layers': '1', 'factor': '3',
        'batch_size': '16', 'd_model': '16', 'des': 'Exp',
        "loss": 'SMAPE'
    }
}

Datasets = {
    "Inland Wind Farm Dataset1(WT1)": {
        'root_path': '/kaggle/input/zenodo5516552-wt1',
        'data_path': 'Inland Wind Farm Dataset1(WT1).csv',
        'date_column': 'Sequence',
        'target': 'y',
        'enc_in': '6', 'dec_in': '6', 'c_out': '1',
    },
    "QLD":{
        'root_path': '/kaggle/input/zenodo5516552-wt1',
        'data_path': 'Inland Wind Farm Dataset1(WT1).csv',
        'date_column': 'Sequence',
        'target': 'y',
        'enc_in': '6', 'dec_in': '6', 'c_out': '1',
    }
}

Base = {
    'learning_rate': '0.001',
    'is_training': '1',
    'seq_len': '96',
    'label_len': '48',
    'pred_len': '1',
    'result_rpath': '/kaggle/working',
    'num_workers': '10',
}

Inland_Wind_Farm_Dataset = {
    "Transformer":
        {'task_name': 'short_term_forecast', 'is_training': '1', 'root_path': '/kaggle/input/zenodo5516552-wt1',
         'seasonal_patterns': "Monthly", 'model_id': '[Inland-Wind-Farm-Dataset1(WT1)]_Transformer_conv_96_1',
         'model': 'Transformer', 'data': 'former', 'features': 'MS', 'e_layers': '2', 'd_layers': '1', 'factor': '3',
         'enc_in': '6', 'dec_in': '6', 'c_out': '1', 'batch_size': '16', 'd_model': '16', 'des': 'Exp',
         'learning_rate': '0.001', 'date_column': 'Sequence',
         'data_path': 'Inland Wind Farm Dataset1(WT1).csv', 'seq_len': '96', 'label_len': '48', 'pred_len': '1',
         'target': 'y', 'result_rpath': '/kaggle/working', 'num_workers': '10',
         "loss": 'SMAPE'},
}
