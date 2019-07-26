from resnet_2d3d import * 

def select_resnet(network, sample_size, seq_len, 
                  batchnorm, affine=True, shortcut_type='B',
                  track_running_stats=True, expand_factor=1,):
    param = {'feature_size': 1024 * expand_factor}
    if network == 'resnet18':
        model = resnet18_2d3d_full(shortcut_type=shortcut_type,
                                   sample_size=sample_size,
                                   sample_duration=seq_len,
                                   batchnorm=batchnorm,
                                   affine=affine,
                                   track_running_stats=track_running_stats,
                                   expand_factor=expand_factor)
        param['feature_size'] = 256 * expand_factor
    elif network == 'resnet34':
        model = resnet34_2d3d_full(shortcut_type=shortcut_type,
                                   sample_size=sample_size,
                                   sample_duration=seq_len,
                                   batchnorm=batchnorm,
                                   affine=affine,
                                   track_running_stats=track_running_stats,
                                   expand_factor=expand_factor)
        param['feature_size'] = 256 * expand_factor 
    elif network == 'resnet50':
        model = resnet50_2d3d_full(shortcut_type=shortcut_type,
                                   sample_size=sample_size,
                                   sample_duration=seq_len,
                                   batchnorm=batchnorm,
                                   affine=affine,
                                   track_running_stats=track_running_stats,
                                   expand_factor=expand_factor)
    elif network == 'resnet101':
        model = resnet101_2d3d_full(shortcut_type=shortcut_type,
                                    sample_size=sample_size,
                                    sample_duration=seq_len,
                                    batchnorm=batchnorm,
                                    affine=affine,
                                    track_running_stats=track_running_stats,
                                    expand_factor=expand_factor)
    elif network == 'resnet152':
        model = resnet152_2d3d_full(shortcut_type=shortcut_type,
                                    sample_size=sample_size,
                                    sample_duration=seq_len,
                                    batchnorm=batchnorm,
                                    affine=affine,
                                    track_running_stats=track_running_stats,
                                    expand_factor=expand_factor)
    else: raise IOError('model type is wrong')
    return model, param